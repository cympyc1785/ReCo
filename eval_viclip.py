"""ViCLIP overall_consistency (영상-instruction 정합)를 우리 val32 영상에 적용.

VBench의 overall_consistency 로직을 영상별 prompt(instruction)로 직접 호출.
reco_vbench 환경 + tools/VBench import.

Usage:
  python eval_viclip.py --gen_dir all_results/val32_run5_2000 --name run5 --out_dir all_results/viclip
"""
import argparse, csv, json, os, sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, 'tools', 'VBench'))
import pkg_resources
if not hasattr(pkg_resources, 'packaging'):
    import packaging, packaging.version, packaging.specifiers, packaging.requirements
    pkg_resources.packaging = packaging

import numpy as np
import torch
from vbench.third_party.ViCLIP.viclip import ViCLIP
from vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer
from vbench.utils import clip_transform, read_frames_decord_by_fps, CACHE_DIR
from vbench.overall_consistency import get_vid_features, get_text_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen_dir', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--val_json', default='ReCo-Data/ReCo-Data/add/add_val_configs.json')
    ap.add_argument('--max_items', type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda')

    tokenizer = SimpleTokenizer(os.path.join(CACHE_DIR, 'ViCLIP/bpe_simple_vocab_16e6.txt.gz'))
    viclip = ViCLIP(tokenizer=tokenizer, pretrain=f'{CACHE_DIR}/ViCLIP/ViClip-InternVid-10M-FLT.pth').to(device).eval()
    tf = clip_transform(224)

    val = json.load(open(args.val_json))[:args.max_items]
    rows = []
    for i, it in enumerate(val):
        vp = os.path.join(args.gen_dir, f'{i:03d}_edited.mp4')
        query = it['instruction_final_refine'].strip()
        with torch.no_grad():
            imgs = tf(read_frames_decord_by_fps(vp, num_frames=8, sample='middle')).to(device)
            vfeat = get_vid_features(viclip, imgs.unsqueeze(0))
            tfeat = get_text_features(viclip, query, tokenizer)
            score = float((vfeat @ tfeat.T)[0][0].cpu())
        rows.append({'idx': i, 'phrase': query[:60], 'viclip_overall': round(score, 4)})
        print(f'[{i:02d}] {score:.4f}', flush=True)

    with open(os.path.join(args.out_dir, f'{args.name}_viclip.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['idx', 'phrase', 'viclip_overall']); w.writeheader(); w.writerows(rows)
    mean = round(float(np.mean([r['viclip_overall'] for r in rows])), 4)
    json.dump({'viclip_overall': mean}, open(os.path.join(args.out_dir, f'{args.name}_viclip_summary.json'), 'w'), indent=2)
    print(f'[{args.name}] viclip_overall mean: {mean}')


if __name__ == '__main__':
    main()
