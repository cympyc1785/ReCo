"""WorldTraj/dynamicverse/DAVIS → ReCo 학습 포맷 변환.

scene별:
  src  = inpaint_result_effecterase.mp4            (EffectErase로 객체 제거된 영상)
  tar  = video_input.mp4 (480x832로 리사이즈)       (원본 = 객체 있는 영상)
  text = "Add a " + category/category.json 의 reasoning 첫 항목 value
  mask = mask/*.png (instance mask, RGB 색상 인코딩) → 합집합 binary → video_masks mp4

정렬/패딩: src·tar·mask를 공통 T(min)로 자르고 마지막 프레임 반복으로 81프레임 패딩.
출력 레이아웃 (ReCo와 동일):
  davis_data/add/{add_data_configs.json, add_val_configs.json, src_videos/, tar_videos/}
  davis_data/video_masks/add/<scene>.mp4

val = 앞 8개 scene (train과 의도적으로 겹침 — overfitting 실험용, 학습 제외하지 않음)
"""
import glob
import json
import os

import cv2
import numpy as np
from PIL import Image
import decord

import sys
sys.path.insert(0, 'scripts')
from diffsynth import save_video

ROOT = 'WorldTraj/dynamicverse/DAVIS'
OUT = 'davis_data'
H, W, TARGET_F = 480, 832, 81


def read_resize(path):
    vr = decord.VideoReader(path)
    frames = vr.get_batch(list(range(len(vr)))).asnumpy()
    if frames.shape[1:3] != (H, W):
        frames = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in frames])
    return frames


def pad_to(frames, n):
    if len(frames) >= n:
        return frames[:n]
    pad = np.repeat(frames[-1:], n - len(frames), axis=0)
    return np.concatenate([frames, pad], axis=0)


def main():
    os.makedirs(f'{OUT}/add/src_videos', exist_ok=True)
    os.makedirs(f'{OUT}/add/tar_videos', exist_ok=True)
    os.makedirs(f'{OUT}/video_masks/add', exist_ok=True)

    entries = []
    scenes = sorted(os.listdir(ROOT))
    for s in scenes:
        p = os.path.join(ROOT, s)
        src_p = os.path.join(p, 'inpaint_result_effecterase.mp4')
        tar_p = os.path.join(p, 'video_input.mp4')
        cat_p = os.path.join(p, 'category/category.json')
        mask_pngs = sorted(glob.glob(os.path.join(p, 'mask', '*.png')))
        if not (os.path.exists(src_p) and os.path.exists(tar_p) and os.path.exists(cat_p) and mask_pngs):
            print(f'[skip] {s}: 파일 누락')
            continue

        cat = json.load(open(cat_p))
        reasoning = cat.get('reasoning') or {}
        if not reasoning:
            print(f'[skip] {s}: reasoning 비어있음')
            continue
        desc = list(reasoning.values())[0].strip()
        desc = desc[0].lower() + desc[1:] if desc else desc   # "Man in ..." → "man in ..."
        instruction = 'Add a ' + desc

        src = read_resize(src_p)
        tar = read_resize(tar_p)
        # instance mask (RGB 색상별 인스턴스) → 합집합 binary
        masks = []
        for mp in mask_pngs:
            m = np.array(Image.open(mp).convert('RGB'))
            binm = (m.max(-1) > 0).astype(np.uint8) * 255
            binm = cv2.resize(binm, (W, H), interpolation=cv2.INTER_NEAREST)
            masks.append(np.stack([binm] * 3, -1))
        masks = np.stack(masks)

        T = min(len(src), len(tar), len(masks))
        src, tar, masks = pad_to(src[:T], TARGET_F), pad_to(tar[:T], TARGET_F), pad_to(masks[:T], TARGET_F)

        save_video(list(src), f'{OUT}/add/src_videos/{s}.mp4', fps=16, quality=7)
        save_video(list(tar), f'{OUT}/add/tar_videos/{s}.mp4', fps=16, quality=7)
        save_video(list(masks), f'{OUT}/video_masks/add/{s}.mp4', fps=16, quality=7)

        entries.append({
            'src_video': f'add/src_videos/{s}.mp4',
            'tar_video': f'add/tar_videos/{s}.mp4',
            'instruction_final_refine': instruction,
        })
        print(f'[ok] {s}: T={T} | {instruction[:60]}', flush=True)

    json.dump(entries, open(f'{OUT}/add/add_data_configs.json', 'w'), indent=2, ensure_ascii=False)
    json.dump(entries[:8], open(f'{OUT}/add/add_val_configs.json', 'w'), indent=2, ensure_ascii=False)
    print(f'\n총 {len(entries)}개 scene 변환 완료. val(고정 8개, train과 겹침): {[os.path.basename(e["tar_video"]) for e in entries[:8]]}')


if __name__ == '__main__':
    main()
