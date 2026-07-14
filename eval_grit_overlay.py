"""GRiT(DenseCaptioning) 검출을 우리 edited 영상에 적용 → bbox+label overlay 영상 저장.

각 프레임을 GRiT로 검출(object description + bbox) → 영상에 그림.
우리 diff/det(GroundingDINO) bbox와 시각 비교용.

reco_vbench 환경, CUDA_HOME=/usr/local/cuda-12.8, tools/VBench cwd.

Usage:
  python eval_grit_overlay.py --gen_dir <abs>/all_results/val32_run5_2000 --name run5 \
      --out_dir <abs>/all_results/grit_overlay --max_items 8 --frame_stride 8
"""
import argparse, json, os, sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_VB = os.path.join(_ROOT, 'tools', 'VBench')
sys.path.insert(0, _VB)
import pkg_resources
if not hasattr(pkg_resources, 'packaging'):
    import packaging, packaging.version, packaging.specifiers, packaging.requirements
    pkg_resources.packaging = packaging

import cv2
import numpy as np
import torch
from vbench.third_party.grit_model import DenseCaptioning

GRIT_W = os.path.expanduser('~/.cache/vbench/grit_model/grit_b_densecap_objectdet.pth')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen_dir', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--max_items', type=int, default=8)
    ap.add_argument('--frame_stride', type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import decord
    sys.path.insert(0, '.')  # tools/VBench cwd
    dc = DenseCaptioning(torch.device('cuda'))
    dc.initialize_model_det(GRIT_W)

    det_records = {}
    for i in range(args.max_items):
        vp = os.path.join(args.gen_dir, f'{i:03d}_edited.mp4')
        if not os.path.exists(vp):
            continue
        vr = decord.VideoReader(vp)
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()
        T = len(frames)
        out_frames = []
        last_dets = []
        rec = []
        for t in range(T):
            if t % args.frame_stride == 0:
                preds, _ = dc.run_det_tensor(frames[t][:, :, ::-1])  # GRiT는 BGR 기대
                inst = preds['instances']
                dets = []
                if inst.has('pred_boxes'):
                    boxes = inst.pred_boxes.tensor.cpu().numpy()
                    descs = inst.pred_object_descriptions.data
                    scores = inst.scores.cpu().numpy() if inst.has('scores') else [1.0] * len(boxes)
                    for b, d, s in zip(boxes, descs, scores):
                        dets.append((d, [int(x) for x in b], float(s)))
                last_dets = dets
                rec.append({'frame': t, 'dets': dets})
            f = frames[t].copy()
            for d, box, s in last_dets:
                x0, y0, x1, y1 = box
                cv2.rectangle(f, (x0, y0), (x1, y1), (255, 50, 50), 3)
                cv2.putText(f, f'{d[:18]} {s:.2f}', (x0 + 3, max(y0 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 50), 2)
            cv2.putText(f, args.name, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            out_frames.append(f)
        # save overlay (cv2 VideoWriter, RGB->BGR)
        H, W = out_frames[0].shape[:2]
        vw = cv2.VideoWriter(os.path.join(args.out_dir, f'{i:02d}_grit.mp4'),
                             cv2.VideoWriter_fourcc(*'mp4v'), 16, (W, H))
        for f in out_frames:
            vw.write(f[:, :, ::-1])
        vw.release()
        det_records[i] = rec
        print(f'[{args.name} {i:02d}] frames detected, sample dets:',
              rec[0]['dets'][:2] if rec and rec[0]['dets'] else 'none', flush=True)

    json.dump(det_records, open(os.path.join(args.out_dir, f'{args.name}_grit_dets.json'), 'w'), indent=2)
    print('saved overlays + dets to', args.out_dir)


if __name__ == '__main__':
    main()
