"""생성 결과 폴더에 전체 metric suite를 적용하는 표준 평가 스크립트.

- per-sample 점수를 CSV로 저장 (낮은/높은 샘플을 영상으로 직접 확인 가능하도록 파일 경로 포함)
- 집계값(mean)을 wandb run에 logging (key prefix 지정 가능)
- 마지막에 best/worst 샘플 요약 출력

Usage:
    python eval_metrics_suite.py --gen_dir all_results/val32_baseline \
        --max_items 32 --run_id baseline --ckpt_step 0 [--no_wandb] [--device cuda]

전제: gen_dir 안에 {idx:03d}_edited.mp4 (eval_val8.py가 저장), val config 순서와 정렬.
"""
import argparse
import csv
import json
import os

import numpy as np

from video_metrics import get_metric, read_video, extract_add_phrase

PER_SAMPLE_METRICS = ['psnr', 'ssim', 'lpips',
                      'bg_psnr', 'bg_ssim', 'bg_lpips',
                      'fg_psnr', 'fg_ssim', 'fg_lpips',
                      'diff_iou', 'diff_area_ratio', 'diff_success', 'diff_outside_mask',
                      'det_iou', 'det_area_ratio', 'det_success']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen_dir', required=True, help='{idx:03d}_edited.mp4 가 있는 생성 결과 폴더')
    ap.add_argument('--val_json', default='ReCo-Data/ReCo-Data/add/add_val_configs.json')
    ap.add_argument('--data_root', default='ReCo-Data/ReCo-Data')
    ap.add_argument('--max_items', type=int, default=32)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--run_id', default=None, help='wandb run id (없으면 wandb 로깅 생략)')
    ap.add_argument('--ckpt_step', type=int, default=0)
    ap.add_argument('--key_prefix', default='val32')
    ap.add_argument('--csv_out', default=None, help='기본: <gen_dir>/metrics_full.csv')
    ap.add_argument('--no_wandb', action='store_true')
    args = ap.parse_args()

    val = json.load(open(args.val_json))[:args.max_items]
    csv_path = args.csv_out or os.path.join(args.gen_dir, 'metrics_full.csv')
    DEV = args.device
    M = {n: get_metric(n) for n in ['psnr', 'ssim', 'lpips', 'masked_psnr', 'masked_ssim',
                                    'masked_lpips', 'fvd', 'diff_bbox', 'detection_bbox']}

    rows, edited_set, gt_set = [], [], []
    for i, it in enumerate(val):
        tar_name = os.path.basename(it['tar_video'])
        edited_path = os.path.join(args.gen_dir, f'{i:03d}_edited.mp4')
        edited = read_video(edited_path)
        src = read_video(os.path.join(args.data_root, it['src_video']))
        gt = read_video(os.path.join(args.data_root, it['tar_video']))
        mask = read_video(os.path.join(args.data_root, 'video_masks/add', tar_name))
        phrase = extract_add_phrase(it['instruction_final_refine'])
        edited_set.append(edited); gt_set.append(gt)

        rd = M['diff_bbox'](edited, src, mask=mask)
        rt = M['detection_bbox'](edited, phrase, mask=mask, device=DEV)
        row = {
            'idx': i,
            'edited_video': edited_path,
            'src_video': os.path.join(args.data_root, it['src_video']),
            'gt_video': os.path.join(args.data_root, it['tar_video']),
            'phrase': phrase,
            'psnr': M['psnr'](edited, gt, device=DEV),
            'ssim': M['ssim'](edited, gt, device=DEV),
            'lpips': M['lpips'](edited, gt, device=DEV),
            'bg_psnr': M['masked_psnr'](edited, src, mask, region='background', device=DEV),
            'bg_ssim': M['masked_ssim'](edited, src, mask, region='background', device=DEV),
            'bg_lpips': M['masked_lpips'](edited, src, mask, region='background', device=DEV),
            'fg_psnr': M['masked_psnr'](edited, gt, mask, region='foreground', device=DEV),
            'fg_ssim': M['masked_ssim'](edited, gt, mask, region='foreground', device=DEV),
            'fg_lpips': M['masked_lpips'](edited, gt, mask, region='foreground', device=DEV),
            'diff_iou': rd['iou'], 'diff_area_ratio': rd['area_ratio'],
            'diff_success': rd['success_rate'], 'diff_outside_mask': rd['outside_mask_ratio'],
            'det_iou': rt['iou'], 'det_area_ratio': rt['area_ratio'], 'det_success': rt['success_rate'],
        }
        rows.append(row)
        print(f'[{i:02d}] psnr {row["psnr"]:.2f} diff_iou {row["diff_iou"]:.3f} | {phrase[:40]}', flush=True)

    # ---- per-sample CSV ----
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\nCSV saved: {csv_path}')

    # ---- 집계 + set-level ----
    agg = {f'{args.key_prefix}/{k}': float(np.nanmean([r[k] for r in rows])) for k in PER_SAMPLE_METRICS}
    agg[f'{args.key_prefix}/fvd'] = M['fvd'](edited_set, gt_set, device=DEV, batch_size=8)
    print('\n===== aggregate =====')
    for k in sorted(agg):
        print(f'  {k}: {agg[k]:.4f}')

    # ---- best/worst 샘플 (diff_iou 기준) ----
    by = sorted(rows, key=lambda r: (np.nan_to_num(r['diff_iou'], nan=-1)))
    print('\nWORST 3 (diff_iou):')
    for r in by[:3]:
        print(f'  idx {r["idx"]:02d} iou {r["diff_iou"]:.3f} | {r["edited_video"]} | {r["phrase"][:40]}')
    print('BEST 3 (diff_iou):')
    for r in by[-3:]:
        print(f'  idx {r["idx"]:02d} iou {r["diff_iou"]:.3f} | {r["edited_video"]} | {r["phrase"][:40]}')

    # ---- wandb ----
    if args.run_id and not args.no_wandb:
        import wandb
        agg['trainer/global_step'] = args.ckpt_step
        run = wandb.init(entity='VCAI_Vid', project='ReCo', id=args.run_id, name=args.run_id, resume='allow')
        run.log(agg)
        run.finish()
        print(f'\nlogged to wandb run {args.run_id}')


if __name__ == '__main__':
    main()
