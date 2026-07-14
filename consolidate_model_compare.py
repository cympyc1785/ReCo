"""기존 per-sample CSV들(self-built metrics_full, VBench edited, ViCLIP, Gemini)을
모델별로 join하여 단일 비교표 생성. 재실행/재추론 없음 — 디스크의 CSV만 읽음.

출력: all_results/model_comparison.csv  (+ stdout 표)
"""
import csv, glob, json, os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
AR = os.path.join(ROOT, 'all_results')

# 6개 모델: name -> val32 dir
MODELS = {
    'baseline': 'val32_baseline',
    'reco2025': 'val32_reco2025',
    'run4': 'val32_run4_2000',
    'run5': 'val32_run5_2000',
    'run7': 'val32_run7_2000',
    'run8': 'val32_run8_contrast_single_2000',
}

# self-built metrics_full.csv에서 평균낼 핵심 컬럼
SELF_COLS = ['psnr', 'ssim', 'lpips', 'bg_psnr', 'bg_lpips', 'fg_psnr',
             'diff_iou', 'diff_success', 'det_iou', 'det_success',
             'bgpure_psnr', 'bgpure_ssim', 'bgpure_lpips']
VB_COLS = ['subject_consistency', 'background_consistency', 'dynamic_degree',
           'imaging_quality', 'aesthetic_quality']


def mean_csv(path, cols, n=8):
    """metrics_full.csv 앞 n개 샘플의 컬럼 평균 (Gemini가 8개라 동일 모집단 비교 위해 기본 8)."""
    rows = list(csv.DictReader(open(path)))[:n]
    out = {}
    for c in cols:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[c]))
            except (KeyError, ValueError, TypeError):
                pass
        out[c] = round(float(np.mean(vals)), 4) if vals else None
    return out


# VBench edited summary (전체 8샘플 평균; 이미 집계됨)
vb = {r['model']: r for r in csv.DictReader(open(os.path.join(AR, 'vbench', 'vbench_edited_summary.csv')))}

rows = []
for name, d in MODELS.items():
    rec = {'model': name}
    # self-built (앞 8개로 통일)
    mf = os.path.join(AR, d, 'metrics_full.csv')
    if os.path.exists(mf):
        rec.update(mean_csv(mf, SELF_COLS, n=8))
    # vbench
    if name in vb:
        for c in VB_COLS:
            rec[c] = round(float(vb[name][c]), 4)
    # viclip
    vj = os.path.join(AR, 'viclip', f'{name}_viclip_summary.json')
    if os.path.exists(vj):
        rec['viclip'] = round(float(json.load(open(vj))['viclip_overall']), 4)
    # gemini (있으면)
    gj = os.path.join(AR, 'gemini', f'{name}_gemini_summary.json')
    if os.path.exists(gj):
        g = json.load(open(gj))
        for cat in ['edit_accuracy', 'video_quality', 'naturalness']:
            vals = [g[f'{cat}_{j}'] for j in (1, 2, 3) if g.get(f'{cat}_{j}') is not None]
            rec[f'gemini_{cat}'] = round(float(np.mean(vals)), 3) if vals else None
    rows.append(rec)

cols = (['model'] + SELF_COLS + VB_COLS + ['viclip',
        'gemini_edit_accuracy', 'gemini_video_quality', 'gemini_naturalness'])
out_csv = os.path.join(AR, 'model_comparison.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c) for c in cols})

# stdout 표
print('saved:', out_csv)
print()
w_name = max(len(r['model']) for r in rows)
for c in cols:
    if c == 'model':
        print(f'{"metric":<22} | ' + ' | '.join(f'{r["model"]:>10}' for r in rows))
        print('-' * (22 + 13 * len(rows)))
        continue
    line = f'{c:<22} | '
    line += ' | '.join(f'{(str(r.get(c)) if r.get(c) is not None else "-"):>10}' for r in rows)
    print(line)
