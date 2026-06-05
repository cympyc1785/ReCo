# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## Common Commands

```bash
# Inference (single task; edit the script to pick the task)
bash infer_server_single.sh
# or directly:
python inference_reco_single.py --task_name replace \
    --test_txt_file_name assets/replace_test.txt \
    --lora_ckpt all_ckpts/2026_01_16_v1_release.ckpt

# Training (8-GPU torchrun + DeepSpeed Stage 2, LoRA rank/alpha 128)
bash train.sh

# Dataset sanity-check / visualization
python reco_data_test_single.py --json_path ./ReCo-Data/replace/replace_data_configs.json --video_folder ./ReCo-Data --debug
python reco_data_test_mix_data.py --json_folder ./ReCo-Data --video_folder ./ReCo-Data --debug

# Download data / benchmark
bash tools/download_ReCo-Data.sh
bash tools/download_ReCo-Bench.sh

# Evaluation (two-stage, Gemini-2.5-flash-thinking via OpenAI-compatible API; needs OPENAI_API_KEY)
cd tools && bash eval_run_via_gemini.sh
```

## Required Local Assets (not in git)

- `Wan-AI/Wan2.1-VACE-1.3B/` — base model weights
- `all_ckpts/*.ckpt` — ReCo LoRA checkpoints
- `ReCo-Data/{add,remove,replace,style}/` — each task has `{task}_data_configs.json` + `src_videos/` + `tar_videos/`

## Hardcoded Paths That Must Be Edited

- `train.py` `LightningModelForTrain.train_dataloader` (~line 206): `json_folder` and `video_folder` point at the original authors' paths (local mount + S3 bucket). Update before training.
- `train.sh`: `DIT_PATH` (DiT/T5/VAE weight paths), `CUDA_VISIBLE_DEVICES`, `GPU_NUM`, `$MASTER_ADDR`.
- `inference_reco_single.py`: Wan-AI model paths and output folder (`all_results/single_test/`).
- `tools/eval_step1_run_gemini_api.py`: API base URL (`custom_base_url`) and `OPENAI_API_KEY`.

## Architecture

**Training (`train.py`):** PyTorch Lightning `LightningModelForTrain` wraps `WanVideoPipeline`. `--train_architecture` selects what gets trained: `all_lora` (LoRA on both VACE and DiT — the configuration used by `train.sh`), `lora` (VACE only), `vace` (full VACE fine-tune), or `full`. Checkpoints save only trainable params. Models are loaded through `ModelManager_custom` (vendored).

**Data pipeline (`reco_data_test_mix_data.py` — imported by `train.py`, not duplicated):**
- `ReCo_Dataset_train` — per-task dataset; reads `{task}_data_configs.json` (fields: `src_video`, `tar_video`, `instruction_final_refine`), supports local disk or S3 (boto3), shards by `rank::world_size`. Resolution 480×832, up to 81 frames.
- `WebMixDatasetWithLength` — IterableDataset that mixes the four tasks by cumulative probability.
- `reco_data_test_single.py` has a separate standalone `ReCo_Dataset` used only for visualization/debugging.

**Inference (`inference_reco_single.py`):** Parses `assets/{task}_test.txt` files where each line is:
```
video_filename.mp4: instruction text | optional/reference_image.png
```
The reference image is used for add/replace; for `_wf` (propagation) tasks it is the edited first frame. The pipeline is called with `vace_video` (conditioned video), `vace_mask` (edit region), and optional reference image; outputs go to `all_results/single_test/{task_name}/`.

**Evaluation (`tools/` is the canonical copy; `ReCo-Bench/` holds the downloaded benchmark data plus near-identical script copies):**
- `eval_step1_run_gemini_api.py` — Gemini scores each edited video per dimension (edit accuracy, video quality, naturalness), outputs per-video JSON to `all_results/gemini_results`.
- `eval_step2_get_final_scores.py` — aggregates JSONs into final per-task and overall scores. Run only after all four tasks are evaluated.

## Don't

- 실험 결과를 임의로 요약 금지 (wandb 원본 수치 그대로).
- 기존 (published) config 수정 금지. 새 실험은 새 config 파일
- `git push`, `git merge`, `git pull`, 브랜치 생성/삭제 금지.
- GPU 4~7 사용 금지

## Changelog

이 프로젝트는 [Keep a Changelog](https://keepachangelog.com/) 규약을 따른다.

코드를 변경할 때마다 `CHANGELOG.md`의 `[Unreleased]` 섹션에 항목을 추가할 것.

## 작업 흐름

코드 변경 작업이 끝나면 **반드시** 다음을 수행한다:
1. 변경 내용을 `CHANGELOG.md`의 `[Unreleased]`에 기록
2. 사용자에게 어느 카테고리에 추가했는지 알려줄 것

이 단계를 건너뛰지 말 것. 사소한 변경이라도 사용자에게 영향이 있으면 기록한다.