"""VBench(custom_input)로 run별 edited 영상 평가 → 모델별 csv + json 저장.

VLM 미사용, prompt 불필요한 dimension만 사용 (custom_input 지원):
  subject_consistency, background_consistency, temporal_flickering,
  motion_smoothness, dynamic_degree, imaging_quality, aesthetic_quality

reco_vbench 환경 + tools/VBench cwd(또는 PYTHONPATH)에서 실행.

Usage:
  cd tools/VBench
  python /path/eval_vbench.py --gen_dir <abs>/all_results/val32_run5_2000 --name run5 \
      --out_dir <abs>/all_results/vbench
"""
import argparse
import csv
import json
import os
import shutil

import sys
_VBENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools', 'VBench')
if _VBENCH not in sys.path:
    sys.path.insert(0, _VBENCH)

# 최신 setuptools에서 제거된 pkg_resources.packaging을 packaging 모듈로 주입 (clip 등이 참조)
import pkg_resources
if not hasattr(pkg_resources, 'packaging'):
    import packaging, packaging.version, packaging.specifiers, packaging.requirements
    pkg_resources.packaging = packaging

import torch
from vbench import VBench

DIMS = ['subject_consistency', 'background_consistency', 'temporal_flickering',
        'motion_smoothness', 'dynamic_degree', 'imaging_quality', 'aesthetic_quality']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen_dir', required=True, help='*_edited.mp4 들이 있는 폴더 (절대경로)')
    ap.add_argument('--name', required=True, help='모델 이름 (run5 등)')
    ap.add_argument('--out_dir', required=True, help='결과 저장 폴더 (절대경로)')
    ap.add_argument('--dims', nargs='+', default=DIMS)
    ap.add_argument('--full_json', default='vbench/VBench_full_info.json')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda')
    vb = VBench(device, args.full_json, args.out_dir)
    vb.evaluate(videos_path=args.gen_dir, name=args.name,
                dimension_list=args.dims, mode='custom_input')

    # VBench 결과 json: {out_dir}/{name}_eval_results.json  (dimension -> [평균, [per-video dict...]])
    res_path = os.path.join(args.out_dir, f'{args.name}_eval_results.json')
    res = json.load(open(res_path))

    # 집계 json
    agg = {d: (res[d][0] if isinstance(res[d], list) else res[d]) for d in res}
    json.dump(agg, open(os.path.join(args.out_dir, f'{args.name}_summary.json'), 'w'), indent=2)

    # per-sample csv (video별 dimension 점수)
    per = {}  # video_path -> {dim: score}
    for d in res:
        detail = res[d][1] if isinstance(res[d], list) and len(res[d]) > 1 else []
        for item in detail:
            vp = item.get('video_path') or item.get('video_list', [''])[0] if isinstance(item, dict) else ''
            score = item.get('video_results') if isinstance(item, dict) else None
            if vp:
                per.setdefault(os.path.basename(vp), {})[d] = score
    if per:
        def _idx(v):
            b = v.split('_')[0]
            return int(b) if b.isdigit() else -1
        fields = ['idx', 'video'] + args.dims
        rows = []
        for vid in sorted(per, key=_idx):
            row = {'idx': _idx(vid), 'video': vid}
            # dynamic_degree 등 bool -> 0/1
            row.update({k: (int(per[vid][k]) if isinstance(per[vid].get(k), bool) else per[vid].get(k)) for k in args.dims})
            rows.append(row)
        with open(os.path.join(args.out_dir, f'{args.name}_per_sample.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        # 집계 csv도 저장 (재실행 없이 바로 비교용)
        import numpy as np
        agg_row = {'model': args.name}
        for k in args.dims:
            vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
            agg_row[k] = round(float(np.mean(vals)), 4) if vals else None
        summary_csv = os.path.join(args.out_dir, 'vbench_summary.csv')
        write_header = not os.path.exists(summary_csv)
        with open(summary_csv, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['model'] + args.dims)
            if write_header: w.writeheader()
            w.writerow(agg_row)
    print(f'[{args.name}] 집계:', {k: round(v, 4) if isinstance(v, float) else v for k, v in agg.items()})
    print('saved:', res_path)


if __name__ == '__main__':
    main()
