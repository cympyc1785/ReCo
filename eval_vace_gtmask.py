"""원조 VACE 용법으로 평가: GT object mask 영역만 inpainting.

ReCo의 in-context concat(우측 절반 전부 생성) 대신, 단일 832폭 입력으로
src 영상 + GT mask(생성할 영역) + instruction을 줘서 mask 영역만 생성.
→ "mask가 주어졌을 때의 상한선" 측정용 (ReCo 목표 = mask 없이 이 수준 도달).

출력: {idx:03d}_edited.mp4 (480x832, eval_metrics_suite.py 호환) + {idx:03d}_grid.mp4 (검수용)

Usage:
    python eval_vace_gtmask.py --process_id 0 --num_procs 4 --max_items 32 \
        --out_dir all_results/val32_vace_base_gtmask [--no_lora | --lora_ckpt <ckpt>]
"""
import argparse
import json
import os

import numpy as np
import torch

import sys
sys.path.insert(0, 'scripts')
from diffsynth import ModelManager, WanVideoPipeline, save_video
from inference_reco_single import add_lora_to_model, seed_everything
from video_metrics import read_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_json", type=str, default="ReCo-Data/ReCo-Data/add/add_val_configs.json")
    parser.add_argument("--data_root", type=str, default="ReCo-Data/ReCo-Data")
    parser.add_argument("--base_wan_folder", type=str, default="checkpoints")
    parser.add_argument("--lora_ckpt", type=str, default=None)
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--process_id", type=int, default=0)
    parser.add_argument("--num_procs", type=int, default=1)
    parser.add_argument("--max_items", type=int, default=32)
    args = parser.parse_args()

    seed_everything(2025)
    os.makedirs(args.out_dir, exist_ok=True)
    val_items = json.load(open(args.val_json))[:args.max_items]
    my_indices = list(range(args.process_id, len(val_items), args.num_procs))

    # ---------- model ----------
    ckpt_list = [
        f"{args.base_wan_folder}/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors",
        f"{args.base_wan_folder}/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        f"{args.base_wan_folder}/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth",
    ]
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(ckpt_list, torch_dtype=torch.bfloat16)
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
    if not args.no_lora:
        assert args.lora_ckpt, "--lora_ckpt 또는 --no_lora 필요"
        for model in [pipe.vace, pipe.denoising_model()]:
            add_lora_to_model(model, lora_rank=128, lora_alpha=128,
                              lora_target_modules="q,k,v,o,ffn.0,ffn.2",
                              init_lora_weights="kaiming", pretrained_lora_path=args.lora_ckpt)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.eval()

    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    for idx in my_indices:
        it = val_items[idx]
        out_path = os.path.join(args.out_dir, f"{idx:03d}_edited.mp4")
        if os.path.exists(out_path):
            continue
        src = read_video(os.path.join(args.data_root, it['src_video'])).astype(np.float32)
        gt = read_video(os.path.join(args.data_root, it['tar_video'])).astype(np.float32)
        mask = read_video(os.path.join(args.data_root, 'video_masks/add', os.path.basename(it['tar_video'])))
        T = min(len(src), len(gt), len(mask), 81)

        src_t = torch.from_numpy(src[:T] / 127.5 - 1).permute(3, 0, 1, 2).unsqueeze(0)      # [1,3,T,H,W]
        gt_t = torch.from_numpy(gt[:T] / 127.5 - 1).permute(3, 0, 1, 2).unsqueeze(0)
        m_t = torch.from_numpy((mask[:T] > 127).astype(np.float32)).permute(3, 0, 1, 2).unsqueeze(0)

        vace_video = src_t * (1 - m_t)            # GT mask 영역만 비움 (원조 VACE inpainting 용법)
        vace_mask = m_t                            # 1 = 생성할 영역
        f, h, w = src_t.shape[2], src_t.shape[3], src_t.shape[4]

        with torch.no_grad():
            video = pipe(
                prompt=it['instruction_final_refine'].strip(),
                negative_prompt=negative_prompt,
                num_inference_steps=50,
                height=h, width=w, num_frames=81,
                seed=1, tiled=False,
                vace_video=vace_video.to(dtype=pipe.torch_dtype, device=pipe.device),
                vace_video_ref=torch.zeros_like(vace_video).to(dtype=pipe.torch_dtype, device=pipe.device),
                vace_mask=vace_mask.to(dtype=pipe.torch_dtype, device=pipe.device),
                tar_video=gt_t.to(dtype=pipe.torch_dtype, device=pipe.device),
                ref_img_pil=None, inference=True,
            )

        # 출력 그리드 [2h, 2w]: 상단=[vace_video|GT], 하단=[생성|mask] → 생성 = [h:, :w]
        grid = np.stack([np.array(fr) for fr in video])
        edited = grid[:, h:, :w, :]
        save_video(list(edited), out_path, fps=16, quality=5)
        save_video(list(grid), os.path.join(args.out_dir, f"{idx:03d}_grid.mp4"), fps=16, quality=6)
        with open(os.path.join(args.out_dir, f"{idx:03d}_prompt.txt"), 'w') as ftxt:
            ftxt.write(it['instruction_final_refine'].strip() + '\n')
        print(f"[proc {args.process_id}] {idx:03d} done", flush=True)


if __name__ == "__main__":
    main()
