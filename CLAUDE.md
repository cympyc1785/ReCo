# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Instruction-based editing model인 ReCo를 이용하여 video에 dynamic object insertion을 하는 연구.

목표 : Novel View Synthesis를 통해 sparse view로부터 랜더된 비디오를 받아서 text instruction을 받아 mask input 없이도 적절한 위치에 dynamic object (human, animal)를 insertion할 수 있는 모델을 만든다.

가설 : 현재 파이프라인을 조금 수정하고 데이터를 추가하면 카메라가 dynamic하게 움직이는 rendered video에서도 dynamic object insertion을 잘 수행할 수 있을 것이다.

현재 문제점 
- Scale Ambiguity : rendered video의 background scene에 어울리지 않는 dynamic object의 크기로 생성됨.
- Object-Scene Inconsistency : dynamic object가 scene과 consistent하지 않음. Scene의 ground에 접지해있어야하는데 foot sliding 같은 현상이 일어남.
- Background Error Propagation : NVS에서 랜더된 비디오에 noise나 뻥 뚫린 공간이 있으면 editing model에서 수정해주지 못하고 유지됨.
- Base Model Performance : 비교적 작은 모델인 Wan VACE 1.3B 모델 기반이라 조금 퀄리티가 떨어짐.

개선방안
- add, remove data만 활용하여 학습시 개선되는지 확인
- 카메라가 움직이는 데이터로 학습
- 직접 카메라 condition을 줘서 학습
- keyframe bbox or point guidance
- artifact video 개선 학습 (uncertainty map)

## Overview

ReCo is a region-constraint in-context video editing model built on **Wan2.1-VACE-1.3B**, fine-tuned with LoRA via a **vendored, modified copy of DiffSynth-Studio** (in `DiffSynth-Studio/`). It supports four editing tasks: `add`, `remove`, `replace`, `style` (plus `_wf`-suffixed variants for propagation given an edited first frame).

There are no tests, linting, or CI in this repo.

## Environment Setup

```bash
conda create -n reco python=3.11 -y
conda activate reco
pip install -r requirements.txt   # note: installs DiffSynth-Studio with `pip install -e .`
```

The local `DiffSynth-Studio/` is **not** upstream — it contains ReCo-specific changes (e.g., `ModelManager_custom` in `diffsynth/models/model_manager_custom.py` zeroes VACE patch-embedding channels 16–32; custom VACE/condition-merge logic in `diffsynth/pipelines/wan_video.py`). Always edit the vendored copy, never pip-install upstream over it.

A **second conda env `reco_vbench`** is required for the VBench-based eval scripts (`eval_vbench.py`, `eval_viclip.py`, `eval_grit_overlay.py`) — they import from the vendored `tools/VBench/` and need its heavier deps (clip, pyiqa, decord, detectron2/GRiT). These scripts inject a `pkg_resources.packaging` shim for newer setuptools; GRiT also needs `CUDA_HOME=/usr/local/cuda-12.8`. The self-built metric scripts (`eval_metrics_suite.py`, `video_metrics.py`) run in the base `reco` env.

## Common Commands

NOTE: upstream commit "Reorganize project folders" moved train/inference/dataset scripts into `scripts/`. Always run them from the repo root so relative paths (`checkpoints/`, `./ReCo-Data`, `all_results/`) resolve.

```bash
# Inference (single task; edit the script to pick the task)
bash scripts/infer_server_single.sh
# or directly:
python scripts/inference_reco_single.py --task_name replace \
    --test_txt_file_name assets/replace_test.txt \
    --lora_ckpt checkpoints/ReCo/2026_01_16_v1_release.ckpt

# Training (4-GPU torchrun + DeepSpeed Stage 2, LoRA rank/alpha 128); run inside a screen session
bash scripts/train.sh

# Quick holdout evaluation (8 fixed validation videos → PSNR/SSIM/LPIPS), then wandb logging
python eval_val8.py --lora_ckpt <ckpt> --out_dir all_results/<dir>   # supports --process_id/--num_procs for multi-GPU
python log_val8_to_wandb.py --json_dir all_results/<dir> --run_id <wandb_run> --ckpt_step <step>

# Dataset sanity-check / visualization
python scripts/reco_data_test_single.py --json_path ./ReCo-Data/ReCo-Data/add/add_data_configs.json --video_folder ./ReCo-Data/ReCo-Data --debug

# Download data subsets (HF)
python download_reco_add.py                  # add task videos
python download_reco_video_masks_add.py      # GT object masks for add
bash tools/download_ReCo-Bench.sh

# Evaluation (two-stage, Gemini-2.5-flash-thinking via OpenAI-compatible API; needs OPENAI_API_KEY)
cd tools && bash eval_run_via_gemini.sh
```

## Required Local Assets (not in git)

- `checkpoints/Wan2.1-VACE-1.3B/` — base model weights (DiT/T5/VAE)
- `checkpoints/ReCo/2026_01_16_v1_release.ckpt` — released ReCo LoRA checkpoint (used as "baseline" in wandb)
- `ReCo-Data/ReCo-Data/{add,remove,replace,style}/` — each task has `{task}_data_configs.json` + `src_videos/` + `tar_videos/` (extracted mp4s, not tars)
- `ReCo-Data/ReCo-Data/video_masks/{task}/` — GT object mask videos (480×832), named after `tar_videos` files
- `ReCo-Data/ReCo-Data/add/add_val_configs.json` — fixed 8-video validation holdout (human/animal × static/moving); excluded from training automatically

## wandb

- Use the `WANDB_API_KEY` exported in `scripts/train.sh` (entity `VCAI_Vid`, project `ReCo`) — `~/.netrc` holds an old account, never rely on it.
- A deleted wandb run id cannot be reused: re-initializing it times out. Pick a new `--run_name` instead.
- run_name convention: append active extra losses (e.g. `train_run2_contrast_attnscore`).

## Hardcoded Paths That Must Be Edited

- `scripts/train.py` `LightningModelForTrain.train_dataloader`: `json_folder`/`video_folder`/`mask_video_folder` (currently `./ReCo-Data/ReCo-Data`); same paths in `run_fixed_validation`.
- `scripts/train.sh`: `DIT_PATH` (DiT/T5/VAE weight paths), `CUDA_VISIBLE_DEVICES`, `GPU_NUM`, `MASTER_ADDR`, `--run_name`.
- `scripts/inference_reco_single.py`: model paths via `--base_wan_folder` (use `checkpoints`) and output folder (`all_results/single_test/`).
- `tools/eval_step1_run_gemini_api.py`: API base URL (`custom_base_url`) and `OPENAI_API_KEY`.

## Architecture

**Training (`scripts/train.py`):** PyTorch Lightning `LightningModelForTrain` wraps `WanVideoPipeline`. `--train_architecture` selects what gets trained: `all_lora` (LoRA on both VACE and DiT — the configuration used by `scripts/train.sh`), `lora` (VACE only), `vace` (full VACE fine-tune), or `full`. Checkpoints save only trainable params (`lora_weights_wan-*.ckpt`) plus a full DeepSpeed resume folder (`wan_deepspeed_folder-*.ckpt`, use with `--resume_ckpt_folder`). Models are loaded through `ModelManager_custom` (vendored; instantiates `WanModel_w_attnscore`/`VaceWanModel_w_attnscore`).

**Losses (paper Eq.17: `L = L_ic + λ1·L_latent + λ2·L_attn`, λ=1e-3):** base flow-matching MSE is always on (`loss_base`). `use_contrast_loss` enables Eq.13 latent regularization (`mask_separation_loss` on pred-x̂₁ src/tar diff vs GT mask → `diff_loss`); `use_attnscore_loss` enables Eq.14-16 attention regularization (`loss_attnscore`). Both need GT masks (`diff_mask`); samples without masks skip them. The authors' raw reference implementation is `tools/train_reco_add_region_loss_raw.py` (note: it gates L_attn to 'vpdata'+'replace' samples only). `use_mse_loss` (edit-area-weighted MSE) is a code-only option not in the paper.

**Validation:** `run_fixed_validation` runs every `log_video_steps` (250) on the 8 holdout videos (2 per rank), computes PSNR/SSIM/LPIPS vs GT, logs `val/*` metrics and `val_videos/*` to wandb, and saves grid videos under `all_videos/{run}/step_{N}/`. Pipeline inference output is a **2×2 grid** (top: [input | GT], bottom: [generated | mask]) — the generated edit region is `[h:, w//2:w]`.

**Data pipeline (`scripts/reco_data_test_mix_data.py` — imported by `scripts/train.py`, not duplicated):**
- `ReCo_Dataset_train` — per-task dataset; reads `{task}_data_configs.json` (fields: `src_video`, `tar_video`, `instruction_final_refine`), supports local disk or S3 (boto3), shards by `rank::world_size`. Resolution 480×832, up to 81 frames.
- `WebMixDatasetWithLength` — IterableDataset that mixes the four tasks by cumulative probability.
- `ReCo_Dataset_train` also accepts `mask_video_folder` to load GT object masks into `diff_mask` (zeros when absent).
- `scripts/reco_data_test_single.py` has a separate standalone `ReCo_Dataset` used only for visualization/debugging.

**Inference (`scripts/inference_reco_single.py`):** Parses `assets/{task}_test.txt` files where each line is:
```
video_filename.mp4: instruction text | optional/reference_image.png
```
The reference image is used for add/replace; for `_wf` (propagation) tasks it is the edited first frame. The pipeline is called with `vace_video` (conditioned video), `vace_mask` (edit region), and optional reference image; outputs go to `all_results/single_test/{task_name}/`.

**ReCo-Bench evaluation (`tools/` is the canonical copy; `ReCo-Bench/` holds the downloaded benchmark data plus near-identical script copies):**
- `eval_step1_run_gemini_api.py` — Gemini scores each edited video per dimension (edit accuracy, video quality, naturalness), outputs per-video JSON to `all_results/gemini_results`.
- `eval_step2_get_final_scores.py` — aggregates JSONs into final per-task and overall scores. Run only after all four tasks are evaluated.

**val32 model-comparison pipeline (root-level `eval_*.py`; the `metric/vbench-eval` branch).** Compares checkpoints on the fixed `add` holdout (first 32 of `add_val_configs.json`). All metric scripts assume a generation dir containing `{idx:03d}_edited.mp4` named in val-config order, and write a **per-sample CSV** (low/high scoring samples can be opened as video — a project convention) plus an aggregate. Run from repo root.
- Step 0 — generate: `eval_val8.py --lora_ckpt <ckpt> --out_dir all_results/val32_<name> --max_items 32` (supports `--no_lora` for the pure Wan2.1-VACE-1.3B base, `--process_id/--num_procs` for multi-GPU). Produces the `{idx:03d}_edited.mp4` files every downstream script consumes.
- `video_metrics.py` — registry (`get_metric(name)`) of self-built metrics: `psnr/ssim/lpips`, `masked_psnr/ssim/lpips` (`region='foreground'|'background'` via GT mask), `fvd` (I3D from `submodule/frechet_video_distance-pytorch`), and three insertion-bbox metrics — `diff_bbox` (RGB diff, calibrated percentile 75/floor 8/min_area 800), `detection_bbox` (GroundingDINO), `hybrid_bbox` (diff∩det). `read_video`/`extract_add_phrase` helpers.
- `eval_metrics_suite.py` — runs the full `video_metrics` suite → `metrics_full.csv` + aggregate wandb logging + best/worst-by-`diff_iou` summary.
- `vbench_metrics.py` — a *separate, self-built* lightweight (non-VLM) reimplementation of VBench dimensions (DINOv2/CLIP consistency, RAFT dynamic-degree, MUSIQ) — its own registry, not to be confused with the real VBench in the next item.
- `eval_vbench.py` / `eval_viclip.py` / `eval_grit_overlay.py` — wrap the **real** vendored `tools/VBench/` (run in the `reco_vbench` env): VBench `custom_input` dimensions, ViCLIP text-video alignment, and GRiT detection overlays respectively.
- `eval_gemini_val32.py` — applies the ReCo-Bench 9-dimension Gemini rubric to val32 edited videos (reuses `tools/eval_step1_run_gemini_api.py`). Note: API key + base URL are hardcoded in the script; sequential with `--req_interval` for the free-tier rate limit.
- `consolidate_model_compare.py` — joins all the above per-model CSV/JSON outputs under `all_results/` into `all_results/model_comparison.csv`. Pure disk read, no re-inference; the `MODELS` dict maps model name → val32 dir and must be edited to add runs.
- `eval_vace_gtmask.py` — oracle eval: original VACE usage (GT-mask inpainting) to measure the upper bound when a mask is available. `visualize_bbox_metrics.py` renders bbox/diff overlays for sanity-checking the bbox metrics.

`EXPERIMENTS.log` is the running narrative of metric analysis and training runs (val32 numbers, bbox-metric findings, per-run loss configs) — read it for context on past experiments alongside `CHANGELOG.md`.

## Don't

- 실험 결과를 임의로 요약 금지 (wandb 원본 수치 그대로).
- 기존 (published) config 수정 금지. 새 실험은 새 config 파일
- `git push`, `git merge`, `git pull`, 브랜치 생성/삭제 금지.
- GPU 4~7 사용 금지

## Changelog

이 프로젝트는 [Keep a Changelog](https://keepachangelog.com/) 규약을 따른다.

코드를 변경할 때마다 `CHANGELOG.md`의 `[Unreleased]` 섹션에 항목을 추가할 것.

실험할 때마다 실험한 내용 단위로 notion api인 /home/korea_kh63/.config/notion/notion-token를 이용해서 정리해줘.

## 작업 흐름

코드 변경 작업이 끝나면 **반드시** 다음을 수행한다:
1. 변경 내용을 `CHANGELOG.md`의 `[Unreleased]`에 기록
2. 사용자에게 어느 카테고리에 추가했는지 알려줄 것

이 단계를 건너뛰지 말 것. 사소한 변경이라도 사용자에게 영향이 있으면 기록한다.