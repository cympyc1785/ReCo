"""VBench 방법론 기반 비-VLM video consistency/quality metric (자체 구현).

VBench(arXiv 2311.17982) dimension 중 VLM을 쓰지 않고 가벼운 백본으로 재현:
  - subject_consistency   : DINOv2 프레임 feature 코사인 유사도 (객체 외형 시간 일관성)
  - background_consistency: CLIP 이미지 feature 프레임 유사도
  - temporal_flickering   : 인접 프레임 MAE (낮을수록 안정 → 100-scale로 반전)
  - motion_smoothness     : 프레임 가속도(2차 차분) 기반 매끄러움 근사
  - dynamic_degree        : RAFT optical flow 평균 크기 (정적/동적 판별)
  - imaging_quality       : pyiqa MUSIQ (NR-IQA)
  - overall_consistency   : CLIP text-video 정합 (instruction 필요)

입력: uint8 numpy [T,H,W,C] (생성 영상). registry: get_vmetric(name).
"""
import numpy as np
import torch

_CACHE = {}
METRICS = {}


def register(name):
    def deco(fn): METRICS[name] = fn; return fn
    return deco

def get_vmetric(name): return METRICS[name]
def list_vmetrics(): return sorted(METRICS)


def _frames_tensor(video, device, size=224):
    """uint8 [T,H,W,C] -> float [T,3,size,size] (0~1)"""
    import torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(video)).to(device).float().permute(0, 3, 1, 2) / 255.
    return F.interpolate(t, (size, size), mode='bilinear', align_corners=False)


# ---------------- DINOv2 subject consistency ----------------
def _dino(device):
    if 'dino' not in _CACHE:
        m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()
        _CACHE['dino'] = m
    return _CACHE['dino']


@register('subject_consistency')
def subject_consistency(video, device='cuda', **kw):
    """DINOv2 feature 프레임간 코사인 유사도. VBench: (첫프레임 대비 + 인접프레임) 평균."""
    import torch.nn.functional as F
    m = _dino(device)
    x = _frames_tensor(video, device, 224)  # 14 배수 필요 → 224 ok
    with torch.no_grad():
        f = m(x)                              # [T, D]
        f = F.normalize(f, dim=-1)
    sims = []
    for i in range(1, len(f)):
        sims.append((f[0] @ f[i]).item())     # 첫 프레임 대비
        sims.append((f[i-1] @ f[i]).item())   # 인접 프레임
    return float(np.mean(sims)) if sims else 1.0


# ---------------- CLIP background consistency / overall ----------------
def _clip(device):
    if 'clip' not in _CACHE:
        import clip
        model, _ = clip.load('ViT-B/32', device=device)
        _CACHE['clip'] = model.eval()
    return _CACHE['clip']


@register('background_consistency')
def background_consistency(video, device='cuda', **kw):
    """CLIP image feature 프레임간 유사도 (첫프레임 대비 + 인접)."""
    import torch.nn.functional as F
    m = _clip(device)
    x = _frames_tensor(video, device, 224)
    mean = torch.tensor([0.48145466,0.4578275,0.40821073], device=device).view(1,3,1,1)
    std = torch.tensor([0.26862954,0.26130258,0.27577711], device=device).view(1,3,1,1)
    with torch.no_grad():
        f = m.encode_image((x-mean)/std).float()
        f = F.normalize(f, dim=-1)
    sims=[]
    for i in range(1,len(f)):
        sims.append((f[0]@f[i]).item()); sims.append((f[i-1]@f[i]).item())
    return float(np.mean(sims)) if sims else 1.0


@register('overall_consistency')
def overall_consistency(video, prompt='', device='cuda', **kw):
    """CLIP text-video 정합 (프레임별 CLIPScore 평균). instruction 필요."""
    import clip, torch.nn.functional as F
    if not prompt: return float('nan')
    m = _clip(device)
    x = _frames_tensor(video, device, 224)
    mean = torch.tensor([0.48145466,0.4578275,0.40821073], device=device).view(1,3,1,1)
    std = torch.tensor([0.26862954,0.26130258,0.27577711], device=device).view(1,3,1,1)
    tok = clip.tokenize([prompt[:300]], truncate=True).to(device)
    with torch.no_grad():
        imf = F.normalize(m.encode_image((x-mean)/std).float(), dim=-1)
        txf = F.normalize(m.encode_text(tok).float(), dim=-1)
        sims = (imf @ txf.T).squeeze(-1)
    return float(sims.mean().item())


# ---------------- temporal flickering ----------------
@register('temporal_flickering')
def temporal_flickering(video, **kw):
    """인접 프레임 MAE (0~255). VBench는 정적영역 대상이나 여기선 전체. 반환: 100*(1-MAE/255) 안정도."""
    v = video.astype(np.float64)
    mae = np.mean(np.abs(v[1:]-v[:-1]))
    return float(100.0 * (1.0 - mae/255.0))


# ---------------- motion smoothness (가속도 근사) ----------------
@register('motion_smoothness')
def motion_smoothness(video, **kw):
    """프레임 2차 차분(가속도) 크기. VBench AMT 대신 근사 — 낮을수록 매끄러움.
    반환: 100*(1 - accel/255) (높을수록 매끄러움)."""
    v = video.astype(np.float64)
    if len(v) < 3: return 100.0
    accel = np.mean(np.abs(v[2:] - 2*v[1:-1] + v[:-2]))
    return float(100.0 * (1.0 - accel/255.0))


# ---------------- dynamic degree (RAFT optical flow) ----------------
def _raft(device):
    if 'raft' not in _CACHE:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        m = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(device).eval()
        _CACHE['raft'] = m
    return _CACHE['raft']


@register('dynamic_degree')
def dynamic_degree(video, device='cuda', region=None, stride=2, **kw):
    """RAFT optical flow 평균 크기. region=[T,H,W] bool 주면 그 영역만(객체 모션).
    '수정 안 함/정적' 검출용 — 값이 낮으면 거의 안 움직임."""
    import torch.nn.functional as F
    m = _raft(device)
    T = len(video)
    H, W = video.shape[1:3]
    Hr, Wr = (H//8)*8, (W//8)*8     # RAFT 8배수
    mags = []
    with torch.no_grad():
        for i in range(0, T-1, stride):
            a = torch.from_numpy(video[i]).to(device).float().permute(2,0,1)[None]/255.*2-1
            b = torch.from_numpy(video[i+1]).to(device).float().permute(2,0,1)[None]/255.*2-1
            a = F.interpolate(a,(Hr,Wr),mode='bilinear',align_corners=False)
            b = F.interpolate(b,(Hr,Wr),mode='bilinear',align_corners=False)
            flow = m(a,b)[-1][0]                       # [2,Hr,Wr]
            mag = torch.sqrt(flow[0]**2+flow[1]**2)    # [Hr,Wr]
            if region is not None:
                rm = torch.from_numpy(region[i]).to(device).float()[None,None]
                rm = F.interpolate(rm,(Hr,Wr),mode='nearest')[0,0] > 0.5
                if rm.sum()>0: mags.append(mag[rm].mean().item())
            else:
                mags.append(mag.mean().item())
    return float(np.mean(mags)) if mags else 0.0


# ---------------- imaging quality (MUSIQ) ----------------
def _musiq(device):
    if 'musiq' not in _CACHE:
        import pyiqa
        _CACHE['musiq'] = pyiqa.create_metric('musiq', device=device)
    return _CACHE['musiq']


@register('imaging_quality')
def imaging_quality(video, device='cuda', stride=4, **kw):
    """pyiqa MUSIQ (NR-IQA, GT 불필요). 프레임 샘플 평균 (0~100)."""
    m = _musiq(device)
    scores=[]
    with torch.no_grad():
        for i in range(0, len(video), stride):
            t = torch.from_numpy(video[i]).to(device).float().permute(2,0,1)[None]/255.
            scores.append(m(t).item())
    return float(np.mean(scores)) if scores else float('nan')
