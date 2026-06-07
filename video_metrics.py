"""모듈형 비디오 metric 라이브러리.

원하는 metric만 골라서 불러와 바로 적용할 수 있는 registry 구조.

입력 규약
---------
- per-video metric: uint8 numpy [T, H, W, C] (0~255)
- mask: [T, H, W] 또는 [T, H, W, C], 값 >127 인 곳이 객체(foreground)
- set-level metric(fvd): 영상 리스트 (각 [T, H, W, C] uint8)

사용 예
-------
    from video_metrics import get_metric, list_metrics

    psnr = get_metric('psnr')(gen, gt)
    bg_psnr = get_metric('masked_psnr')(gen, src, mask, region='background')  # 배경 보존
    fg_lpips = get_metric('masked_lpips')(gen, gt, mask, region='foreground', device='cuda')
    fvd = get_metric('fvd')(gen_videos, ref_videos, device='cuda')            # N>=10 권장

지원 metric: list_metrics() 로 확인.
"""
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_FVD_REPO = os.path.join(_REPO_ROOT, 'submodule', 'frechet_video_distance-pytorch')

METRICS = {}


def register(name):
    def deco(fn):
        METRICS[name] = fn
        return fn
    return deco


def get_metric(name):
    if name not in METRICS:
        raise KeyError(f"unknown metric '{name}'. available: {sorted(METRICS)}")
    return METRICS[name]


def list_metrics():
    return sorted(METRICS)


# ---------------------------------------------------------------- helpers
def _to_bool_mask(mask, region):
    """mask -> [T,H,W] bool (선택한 region이 True)"""
    m = np.asarray(mask)
    if m.ndim == 4:
        m = m[..., 0]
    fg = m > 127 if m.dtype == np.uint8 else m > 0.5
    if region == 'foreground':
        return fg
    elif region == 'background':
        return ~fg
    raise ValueError(f"region must be 'foreground' or 'background', got {region}")


def _align(gen, ref, mask=None):
    T = min(len(gen), len(ref)) if mask is None else min(len(gen), len(ref), len(mask))
    return (gen[:T], ref[:T]) if mask is None else (gen[:T], ref[:T], mask[:T])


_LPIPS_CACHE = {}


def _lpips_model(device, spatial):
    key = (str(device), spatial)
    if key not in _LPIPS_CACHE:
        import lpips as lpips_pkg
        _LPIPS_CACHE[key] = lpips_pkg.LPIPS(net='alex', spatial=spatial).to(device).eval()
    return _LPIPS_CACHE[key]


def _to_lpips_tensor(arr, device):
    import torch
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1


# ---------------------------------------------------------------- 전체 프레임 metric
def _to_f32_tensor(arr, device):
    import torch
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device).float()


def _ssim_map_gpu(gen, ref, device):
    """torchmetrics SSIM (uniform 7x7, skimage 기본과 동일 설정) -> (mean, map [T,H,W,C] numpy)"""
    import torch
    from torchmetrics.functional.image import structural_similarity_index_measure as tm_ssim
    g = _to_f32_tensor(gen, device).permute(0, 3, 1, 2)
    r = _to_f32_tensor(ref, device).permute(0, 3, 1, 2)
    with torch.no_grad():
        score, smap = tm_ssim(g, r, data_range=255.0, gaussian_kernel=False, kernel_size=7,
                              return_full_image=True)
    return float(score.item()), smap.permute(0, 2, 3, 1).cpu().numpy()


@register('psnr')
def psnr(gen, ref, device=None, **kw):
    gen, ref = _align(gen, ref)
    if device:                                   # GPU 경로 (CPU 부하 절감)
        import torch
        g, r = _to_f32_tensor(gen, device), _to_f32_tensor(ref, device)
        with torch.no_grad():
            mse = torch.mean((g - r) ** 2).item()
    else:
        mse = np.mean((gen.astype(np.float64) - ref.astype(np.float64)) ** 2)
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-10)))


@register('ssim')
def ssim(gen, ref, device=None, **kw):
    gen, ref = _align(gen, ref)
    if device:                                   # GPU 경로 (torchmetrics)
        score, _ = _ssim_map_gpu(gen, ref, device)
        return score
    from skimage.metrics import structural_similarity as compute_ssim
    return float(np.mean([compute_ssim(ref[t], gen[t], channel_axis=2) for t in range(len(gen))]))


@register('lpips')
def lpips(gen, ref, device='cuda', chunk=8, **kw):
    import torch
    gen, ref = _align(gen, ref)
    model = _lpips_model(device, spatial=False)
    vals = []
    with torch.no_grad():
        for beg in range(0, len(gen), chunk):
            g = _to_lpips_tensor(gen[beg:beg + chunk], device)
            r = _to_lpips_tensor(ref[beg:beg + chunk], device)
            vals.append(model(g, r).flatten().cpu())
    return float(torch.cat(vals).mean().item())


# ---------------------------------------------------------------- masked metric
@register('masked_psnr')
def masked_psnr(gen, ref, mask, region='foreground', device=None, **kw):
    """선택 영역(foreground=객체 / background=배경)만의 PSNR.
    배경 보존 측정: masked_psnr(gen, src, mask, region='background')"""
    gen, ref, mask = _align(gen, ref, mask)
    sel = _to_bool_mask(mask, region)
    if sel.sum() == 0:
        return float('nan')
    if device:                                   # GPU 경로
        import torch
        g, r = _to_f32_tensor(gen, device), _to_f32_tensor(ref, device)
        s = torch.from_numpy(sel).to(device)
        with torch.no_grad():
            mse = ((g - r) ** 2)[s].mean().item()
    else:
        mse = ((gen.astype(np.float64) - ref.astype(np.float64)) ** 2)[sel].mean()
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-10)))


@register('masked_ssim')
def masked_ssim(gen, ref, mask, region='foreground', device=None, **kw):
    """SSIM map을 구한 뒤 선택 영역만 평균."""
    gen, ref, mask = _align(gen, ref, mask)
    sel = _to_bool_mask(mask, region)
    if sel.sum() == 0:
        return float('nan')
    if device:                                   # GPU 경로 (torchmetrics map)
        _, smap = _ssim_map_gpu(gen, ref, device)
        return float(smap[sel].mean())
    from skimage.metrics import structural_similarity as compute_ssim
    vals = []
    for t in range(len(gen)):
        if sel[t].sum() == 0:
            continue
        _, smap = compute_ssim(ref[t], gen[t], channel_axis=2, full=True)   # [H,W,C]
        vals.append(smap[sel[t]].mean())
    return float(np.mean(vals)) if vals else float('nan')


@register('masked_lpips')
def masked_lpips(gen, ref, mask, region='foreground', device='cuda', chunk=8, **kw):
    """spatial LPIPS map을 구한 뒤 선택 영역만 평균."""
    import torch
    gen, ref, mask = _align(gen, ref, mask)
    sel = _to_bool_mask(mask, region)
    model = _lpips_model(device, spatial=True)
    num, den = 0.0, 0
    with torch.no_grad():
        for beg in range(0, len(gen), chunk):
            g = _to_lpips_tensor(gen[beg:beg + chunk], device)
            r = _to_lpips_tensor(ref[beg:beg + chunk], device)
            lmap = model(g, r)[:, 0].cpu().numpy()                          # [n,H,W]
            s = sel[beg:beg + chunk]
            num += float(lmap[s].sum()); den += int(s.sum())
    return float(num / den) if den > 0 else float('nan')


# ---------------------------------------------------------------- set-level: FVD
_FVD_MOD = None


def _load_fvd_module():
    """fork(cympyc1785/frechet_video_distance-pytorch)를 package alias로 로드
    (폴더명 하이픈 + relative import 때문에 직접 import 불가)."""
    global _FVD_MOD
    if _FVD_MOD is not None:
        return _FVD_MOD
    import importlib.util
    import types
    pkg = types.ModuleType('fvd_pkg')
    pkg.__path__ = [_FVD_REPO]
    sys.modules['fvd_pkg'] = pkg
    spec = importlib.util.spec_from_file_location(
        'fvd_pkg.frechet_video_distance', os.path.join(_FVD_REPO, 'frechet_video_distance.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['fvd_pkg.frechet_video_distance'] = mod
    spec.loader.exec_module(mod)
    _FVD_MOD = mod
    return mod


@register('fvd')
def fvd(gen_videos, ref_videos, device='cuda', batch_size=8, max_frames=None, **kw):
    """Frechet Video Distance (I3D). 입력: 영상 리스트 (각 [T,H,W,C] uint8).
    공분산 추정 때문에 세트당 N>=10 권장. 프레임 수는 두 세트에서 동일하게 맞춤."""
    import torch
    mod = _load_fvd_module()
    T = min(min(len(v) for v in gen_videos), min(len(v) for v in ref_videos))
    if max_frames is not None:
        T = min(T, max_frames)
    g = torch.from_numpy(np.stack([np.asarray(v[:T]) for v in gen_videos])).float()
    r = torch.from_numpy(np.stack([np.asarray(v[:T]) for v in ref_videos])).float()
    weights = os.path.join(_FVD_REPO, 'pytorch_i3d_model', 'models', 'rgb_imagenet.pt')

    i3d = mod.InceptionI3d(400, in_channels=3).to(torch.device(device))
    i3d.load_state_dict(torch.load(weights, map_location=device))
    i3d.train(False)
    with torch.no_grad():
        g_act = mod.get_activations(mod.preprocess(g, (224, 224)), i3d, batch_size=batch_size)
        r_act = mod.get_activations(mod.preprocess(r, (224, 224)), i3d, batch_size=batch_size)
    return float(mod.calculate_fvd_from_activations(g_act, r_act))


# ---------------------------------------------------------------- bbox 기반 (삽입 위치/스케일)
def _bbox_from_bool(m):
    """[H,W] bool -> (x0, y0, x1, y1) or None"""
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _bbox_iou(a, b):
    if a is None or b is None:
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1)


def _bbox_area(b):
    return 0 if b is None else (b[2] - b[0]) * (b[3] - b[1])


def _score_bboxes_vs_gt(pred_boxes, mask):
    """프레임별 예측 bbox vs GT mask bbox -> iou/area_ratio/success 집계"""
    sel = _to_bool_mask(mask, 'foreground')
    ious, ratios, n_eval = [], [], 0
    for t, pb in enumerate(pred_boxes):
        gt_b = _bbox_from_bool(sel[t]) if t < len(sel) else None
        if gt_b is None:
            continue
        n_eval += 1
        ious.append(_bbox_iou(pb, gt_b))
        if pb is not None:
            ratios.append(_bbox_area(pb) / max(_bbox_area(gt_b), 1))
    return {
        'iou': float(np.mean(ious)) if ious else float('nan'),
        'area_ratio': float(np.mean(ratios)) if ratios else float('nan'),
        'success_rate': float(np.mean([i > 0.5 for i in ious])) if ious else float('nan'),
        'detected_rate': float(len(ratios) / max(n_eval, 1)),
    }


@register('diff_bbox')
def diff_bbox(gen, src, mask=None, percentile=75.0, abs_floor=8.0, min_area=800, **kw):
    """RGB diff 기반 삽입 영역 bbox (모델 불필요, deterministic).

    gen vs src 의 |diff|를 percentile+절대값 floor로 이진화, morphology로 정리한 뒤
    최대 connected component의 bbox를 프레임별로 추출.
    mask(GT) 제공 시 iou/area_ratio/success_rate에 더해
    outside_mask_ratio(=GT mask 밖 diff 비율, 배경 오염 지표)도 반환.
    """
    from scipy import ndimage
    gen, src = _align(gen, src)
    diff = np.abs(gen.astype(np.float32) - src.astype(np.float32)).mean(-1)   # [T,H,W]
    boxes, outside = [], []
    sel = _to_bool_mask(mask, 'foreground') if mask is not None else None
    for t in range(len(diff)):
        thr = max(np.percentile(diff[t], percentile), abs_floor)
        binm = diff[t] > thr
        binm = ndimage.binary_opening(binm, iterations=2)
        binm = ndimage.binary_closing(binm, iterations=2)
        lab, n = ndimage.label(binm)
        if n == 0:
            boxes.append(None)
        else:
            sizes = ndimage.sum(binm, lab, range(1, n + 1))
            big = (np.argmax(sizes) + 1)
            comp = lab == big
            boxes.append(_bbox_from_bool(comp) if sizes.max() >= min_area else None)
        if sel is not None and t < len(sel) and binm.sum() > 0:
            outside.append(float((binm & ~sel[t]).sum() / binm.sum()))
    out = {'bboxes': boxes}
    if mask is not None:
        out.update(_score_bboxes_vs_gt(boxes, mask))
        out['outside_mask_ratio'] = float(np.mean(outside)) if outside else float('nan')
    return out


_GDINO_CACHE = {}


def _gdino(device):
    if 'm' not in _GDINO_CACHE:
        from transformers import AutoProcessor, GroundingDinoForObjectDetection
        name = 'IDEA-Research/grounding-dino-base'
        _GDINO_CACHE['p'] = AutoProcessor.from_pretrained(name)
        _GDINO_CACHE['m'] = GroundingDinoForObjectDetection.from_pretrained(name).to(device).eval()
    return _GDINO_CACHE['p'], _GDINO_CACHE['m']


@register('detection_bbox')
def detection_bbox(gen, phrase, mask=None, device='cuda', frame_stride=8,
                   box_threshold=0.3, text_threshold=0.25, **kw):
    """GroundingDINO open-vocab 검출 기반 bbox (frame_stride 간격 샘플 프레임).

    phrase: 검출할 객체 명사구 (예: 'a golden retriever dog').
    mask(GT) 제공 시 iou/area_ratio/success_rate 반환 (샘플 프레임 기준).
    """
    import torch
    from PIL import Image
    proc, model = _gdino(device)
    text = phrase.lower().strip().rstrip('.') + '.'
    frames = list(range(0, len(gen), frame_stride))
    boxes_per_frame = {}
    with torch.no_grad():
        for t in frames:
            img = Image.fromarray(gen[t])
            inputs = proc(images=img, text=text, return_tensors='pt').to(device)
            outputs = model(**inputs)
            res = proc.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=box_threshold,
                text_threshold=text_threshold, target_sizes=[img.size[::-1]])[0]
            if len(res['scores']) > 0:
                best = res['boxes'][res['scores'].argmax()].tolist()
                boxes_per_frame[t] = tuple(int(v) for v in best)        # (x0,y0,x1,y1)
            else:
                boxes_per_frame[t] = None
    out = {'bboxes': boxes_per_frame}
    if mask is not None:
        sel_boxes = [boxes_per_frame[t] for t in frames]
        sub_mask = np.asarray(mask)[frames]
        out.update(_score_bboxes_vs_gt(sel_boxes, sub_mask))
    return out


@register('hybrid_bbox')
def hybrid_bbox(gen, src, phrase, mask=None, device='cuda', frame_stride=8,
                box_threshold=0.25, text_threshold=0.25, overlap_min=0.1,
                diff_kwargs=None, **kw):
    """하이브리드: RGB diff로 '새로 생긴 영역' 후보를 잡고, GroundingDINO 검출 박스 중
    diff 영역과 겹치는 것만 채택 (장면에 원래 있던 동종 객체 오검출 차단).

    프레임별 선택 규칙:
      1) 검출 박스들 중 diff-bbox와 IoU가 가장 큰 박스 (IoU >= overlap_min)
      2) 겹치는 검출이 없으면 diff-bbox 사용
      3) 둘 다 없으면 None
    """
    import torch
    from PIL import Image
    diff_res = diff_bbox(gen, src, mask=None, **(diff_kwargs or {}))
    diff_boxes = diff_res['bboxes']

    proc, model = _gdino(device)
    text = phrase.lower().strip().rstrip('.') + '.'
    frames = list(range(0, len(gen), frame_stride))
    chosen = []
    with torch.no_grad():
        for t in frames:
            img = Image.fromarray(gen[t])
            inputs = proc(images=img, text=text, return_tensors='pt').to(device)
            outputs = model(**inputs)
            res = proc.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=box_threshold,
                text_threshold=text_threshold, target_sizes=[img.size[::-1]])[0]
            det_boxes = [tuple(int(v) for v in b.tolist()) for b in res['boxes']]
            db = diff_boxes[t] if t < len(diff_boxes) else None
            best, best_ov = None, overlap_min
            for cand in det_boxes:
                ov = _bbox_iou(cand, db)
                if ov >= best_ov:
                    best, best_ov = cand, ov
            chosen.append(best if best is not None else db)
    out = {'bboxes': dict(zip(frames, chosen))}
    if mask is not None:
        sub_mask = np.asarray(mask)[frames]
        out.update(_score_bboxes_vs_gt(chosen, sub_mask))
    return out


# ---------------------------------------------------------------- I/O helper
def read_video(path):
    """mp4 -> uint8 [T,H,W,C]"""
    import decord
    vr = decord.VideoReader(path)
    return vr.get_batch(list(range(len(vr)))).asnumpy()


def extract_add_phrase(instruction):
    """'Add a ...' instruction에서 객체 명사구 추출 (GroundingDINO 프롬프트용).
    명사구 내부 쉼표('fluffy, tabby cat')는 유지하고, ', wearing ...' 같은
    절(쉼표+동명사)과 전치사 이하만 잘라낸다."""
    import re
    s = instruction.strip()
    m = re.match(r'^Add\s+(?:a|an|the)?\s*(.+)$', s, flags=re.IGNORECASE)
    s = m.group(1) if m else s
    # 1) 쉼표+동명사 절에서 절단 (', wearing ...' / ', sitting ...')
    s = re.split(r',\s+\w+ing\b', s)[0]
    # 2) 전치사/동작 동명사(공백+ing, 단 선행 형용사 보호 위해 잘 알려진 동작 동사만)에서 절단
    s = re.split(r'\s+(?:on|in|to|at|near|behind|between|over|above|under|from|next)\s+'
                 r'|\s+(?:walking|sitting|standing|flying|swimming|running|kneeling|lying|riding|driving|jumping|dancing|climbing|grazing|perched|peeling|holding|eating|drinking|looking|making)\b', s)[0]
    words = s.rstrip(',').split()
    return ' '.join(words[:8]).strip().rstrip(',')
