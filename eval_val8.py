"""Quick standalone evaluation on the 8-video validation holdout.

ReCo-Data/add/add_val_configs.json의 holdout 8개에 대해 추론을 돌리고
PSNR/SSIM/LPIPS를 계산한다. GPU별 분산 실행을 위해 --process_id/--num_procs 지원.

Usage (4-GPU 분산):
    for i in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$i python eval_val8.py --process_id $i --num_procs 4 \
            --lora_ckpt <ckpt> &
    done; wait
"""
import os
import json
import argparse
import sys

import numpy as np
import torch
from skimage.metrics import structural_similarity as compute_ssim
import lpips as lpips_pkg

# scripts/ 폴더 재구성 이후 모듈 위치 반영
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from diffsynth import ModelManager, WanVideoPipeline, save_video
from inference_reco_single import add_lora_to_model, seed_everything
from reco_data_test_mix_data import ReCo_Dataset_train, collate_fn_with_diff_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_json", type=str, default="ReCo-Data/ReCo-Data/add/add_val_configs.json")
    parser.add_argument("--video_folder", type=str, default="./ReCo-Data/ReCo-Data")
    parser.add_argument("--base_wan_folder", type=str, default="checkpoints")
    parser.add_argument("--lora_ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="all_results/val8_eval")
    parser.add_argument("--process_id", type=int, default=0)
    parser.add_argument("--num_procs", type=int, default=1)
    args = parser.parse_args()

    seed_everything(2025)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- dataset ----------
    with open(args.val_json, "r", encoding="utf-8") as f:
        val_items = json.load(f)
    dataset = ReCo_Dataset_train(all_data_list=val_items, base_video_folder=args.video_folder,
                                 read_video_from_local=True, task_name='add', user_first_frame=False)
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
    for model in [pipe.vace, pipe.denoising_model()]:
        add_lora_to_model(model, lora_rank=128, lora_alpha=128,
                          lora_target_modules="q,k,v,o,ffn.0,ffn.2",
                          init_lora_weights="kaiming", pretrained_lora_path=args.lora_ckpt)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.eval()

    lpips_model = lpips_pkg.LPIPS(net='alex').to("cuda").eval()

    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    # ---------- eval loop ----------
    results = []
    for idx in my_indices:
        sample = dataset[idx]
        batch = collate_fn_with_diff_mask([sample])
        b, c, f, h, w = batch["tar_video_key"].shape

        with torch.no_grad():
            video = pipe(
                prompt=batch["prompt"][0],
                negative_prompt=negative_prompt,
                num_inference_steps=50,
                height=h, width=w, num_frames=81,
                seed=1, tiled=False,
                vace_video=batch["tar_video_key"][:1].to(dtype=pipe.torch_dtype, device=pipe.device),
                vace_video_ref=batch["ref_video"][:1].to(dtype=pipe.torch_dtype, device=pipe.device),
                vace_mask=batch["tar_video_key_mask"][:1].to(dtype=pipe.torch_dtype, device=pipe.device),
                tar_video=batch["tar_video"][:1].to(dtype=pipe.torch_dtype, device=pipe.device),
                ref_img_pil=None, inference=True,
            )

        # ---- metrics: 생성 edit 영역 vs GT ----
        # pipe 출력은 2x2 그리드 [T, 2h, 2w, C]: 상단=[입력|GT], 하단=[생성|마스크]
        gen = np.stack([np.array(frame) for frame in video])                    # [T,2h,2w,C] uint8
        gen_tar = gen[:, h:, w // 2:w, :]                                       # 하단 좌측 concat의 우측 절반 = 생성 edit
        gt = batch["tar_video"][0].permute(1, 2, 3, 0).float().numpy()          # [f,h,w,c], [-1,1]
        gt_tar = ((gt[:, :, w // 2:, :] + 1) * 127.5).clip(0, 255).astype(np.uint8)

        T = min(len(gen_tar), len(gt_tar))
        gen_tar, gt_tar = gen_tar[:T], gt_tar[:T]

        mse = np.mean((gen_tar.astype(np.float64) - gt_tar.astype(np.float64)) ** 2)
        psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-10))
        ssim = float(np.mean([compute_ssim(gt_tar[t], gen_tar[t], channel_axis=2) for t in range(T)]))

        gen_t = torch.from_numpy(gen_tar).permute(0, 3, 1, 2).float().to("cuda") / 127.5 - 1
        gt_t = torch.from_numpy(gt_tar).permute(0, 3, 1, 2).float().to("cuda") / 127.5 - 1
        with torch.no_grad():
            lpips_chunks = [lpips_model(gen_t[beg:beg + 8], gt_t[beg:beg + 8]).flatten()
                            for beg in range(0, T, 8)]
        lpips_val = torch.cat(lpips_chunks).mean().item()

        video_name = batch["video_name"][0]
        save_video(video, os.path.join(args.out_dir, f"{video_name}.mp4"), fps=16, quality=5)
        with open(os.path.join(args.out_dir, f"{video_name}.txt"), "w") as f_txt:
            f_txt.write(f'{batch["prompt"][0]}\n')

        results.append({"video": video_name, "prompt": batch["prompt"][0],
                        "psnr": float(psnr), "ssim": ssim, "lpips": float(lpips_val)})
        print(f"[proc {args.process_id}] {video_name}: PSNR {psnr:.3f}, SSIM {ssim:.4f}, LPIPS {lpips_val:.4f}", flush=True)

        del video, batch, gen_t, gt_t

    with open(os.path.join(args.out_dir, f"metrics_proc{args.process_id}.json"), "w") as f_out:
        json.dump(results, f_out, indent=2, ensure_ascii=False)
    print(f"[proc {args.process_id}] done: {len(results)} videos", flush=True)


if __name__ == "__main__":
    main()
