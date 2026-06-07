"""bbox/diff metric 검증용 시각화.

샘플마다 두 영상 생성:
  {idx:02d}_bbox.mp4 : [GT영상(GT파랑 bbox) | edited(GT파랑/diff초록/det빨강 bbox)] width concat
                       — 왼쪽에서 GT 박스가 실제 GT 객체를 감싸는지 확인, 오른쪽에서 생성 위치와 비교
  {idx:02d}_diff.mp4 : [edited | RGB diff 시각화(grayscale + threshold 영역 빨간 틴트)] width concat

Usage:
    python visualize_bbox_metrics.py --gen_dir all_results/val32_baseline \
        --out_dir all_results/metric_viz/baseline --max_items 32 --device cuda
"""
import argparse
import json
import os

import cv2
import numpy as np

from video_metrics import (get_metric, read_video, extract_add_phrase,
                           _to_bool_mask, _bbox_from_bool)

import sys
sys.path.insert(0, 'scripts')
from diffsynth import save_video


def draw_box(frame, box, color, label=None, thick=3):
    if box is None:
        return
    x0, y0, x1, y1 = box
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, thick)
    if label:
        cv2.putText(frame, label, (x0 + 4, max(y0 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--val_json', default='ReCo-Data/ReCo-Data/add/add_val_configs.json')
    ap.add_argument('--data_root', default='ReCo-Data/ReCo-Data')
    ap.add_argument('--max_items', type=int, default=32)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--frame_stride', type=int, default=8, help='detection 프레임 간격 (그 사이는 마지막 박스 유지)')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    val = json.load(open(args.val_json))[:args.max_items]
    diff_m, det_m = get_metric('diff_bbox'), get_metric('detection_bbox')

    GT_C, DIFF_C, DET_C = (60, 120, 255), (0, 220, 0), (255, 50, 50)   # 파랑 / 초록 / 빨강 (RGB)

    for i, it in enumerate(val):
        edited = read_video(os.path.join(args.gen_dir, f'{i:03d}_edited.mp4'))
        src = read_video(os.path.join(args.data_root, it['src_video']))
        gt = read_video(os.path.join(args.data_root, it['tar_video']))
        mask = read_video(os.path.join(args.data_root, 'video_masks/add', os.path.basename(it['tar_video'])))
        phrase = extract_add_phrase(it['instruction_final_refine'])
        T = min(len(edited), len(src), len(gt), len(mask))

        rd = diff_m(edited, src)                                            # 전 프레임 diff bbox
        rt = det_m(edited, phrase, device=args.device, frame_stride=args.frame_stride)
        sel = _to_bool_mask(mask[:T], 'foreground')

        # ---- 1) [edited+박스 | GT+GT박스] concat ----
        frames_bbox = []
        last_det = None
        for t in range(T):
            f = edited[t].copy()
            if t in rt['bboxes']:
                last_det = rt['bboxes'][t]
            gt_box = _bbox_from_bool(sel[t])
            draw_box(f, gt_box, GT_C, 'GT')
            draw_box(f, rd['bboxes'][t] if t < len(rd['bboxes']) else None, DIFF_C, 'diff')
            draw_box(f, last_det, DET_C, 'det')
            g = gt[t].copy()
            draw_box(g, gt_box, GT_C, 'GT')
            cv2.putText(f, 'edited', (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(g, 'GT', (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            frames_bbox.append(np.concatenate([g, f], axis=1))   # [GT | edited]
        save_video(frames_bbox, os.path.join(args.out_dir, f'{i:02d}_bbox.mp4'), fps=16, quality=6)

        # ---- 2) [edited | diff 시각화] concat ----
        diff = np.abs(edited[:T].astype(np.float32) - src[:T].astype(np.float32)).mean(-1)   # [T,H,W]
        frames_diff = []
        for t in range(T):
            g = np.clip(diff[t] * 2.5, 0, 255).astype(np.uint8)            # 시인성 게인
            viz = np.stack([g, g, g], -1)
            thr = max(np.percentile(diff[t], 75.0), 8.0)                    # calibrated threshold
            binm = diff[t] > thr
            viz[binm] = (0.45 * viz[binm] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)
            frames_diff.append(np.concatenate([edited[t], viz], axis=1))
        save_video(frames_diff, os.path.join(args.out_dir, f'{i:02d}_diff.mp4'), fps=16, quality=6)
        print(f'[{i:02d}] saved | {phrase[:40]}', flush=True)

    print('viz done:', os.path.abspath(args.out_dir))


if __name__ == '__main__':
    main()
