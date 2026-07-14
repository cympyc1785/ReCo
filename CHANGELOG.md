# Changelog

이 프로젝트의 모든 주요 변경 사항을 이 파일에 기록한다.
형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.1.0/)를 따른다.

## [Unreleased]

### Added

**평가 인프라**
- `video_metrics.py`: registry 방식 모듈형 비디오 metric 라이브러리 (`get_metric(name)`으로 선택 로딩).
  - 화질: `psnr`/`ssim`/`lpips` (GPU 경로 — PSNR torch, SSIM torchmetrics, CPU와 diff≤1e-5 검증).
  - masked: `masked_psnr`/`ssim`/`lpips` (`region='foreground'|'background'`, GT mask로 배경 보존·객체 영역 분리).
  - 삽입 위치/스케일 bbox: `diff_bbox`(RGB diff 기반, GT mask grid search 캘리브레이션 percentile 75/floor 8/min_area 800), `detection_bbox`(GroundingDINO), `hybrid_bbox`(diff∩det). val32 비교에서 세 방식 IoU 수렴 확인.
  - 분포: `fvd`(I3D, `submodule/frechet_video_distance-pytorch` fork를 package alias로 로드).
  - 헬퍼: `read_video`, `extract_add_phrase`(명사구 내 쉼표 보존).
- `eval_metrics_suite.py`: 표준 평가 스크립트 — 전체 metric suite + **per-sample CSV**(`metrics_full.csv`, 영상 경로 포함) + 집계 wandb 로깅 + best/worst 샘플 요약.
- `eval_vace_gtmask.py`: 원조 VACE 용법 평가 — 단일 832폭 입력으로 GT object mask 영역만 inpainting ("mask 주어졌을 때의 상한선" oracle 측정).
- `visualize_bbox_metrics.py`: bbox/diff 검증 시각화 — `{idx}_bbox.mp4`([GT \| edited]에 GT파랑/diff초록/det빨강 사각형), `{idx}_diff.mp4`([edited \| RGB diff 히트맵]).
- `submodule/frechet_video_distance-pytorch`: FVD 계산용 fork clone (I3D 가중치 포함).
- 의존성: `lpips` 패키지 설치 (AlexNet 가중치 포함).

**데이터 다운로드/변환**
- `download_reco_video_masks_add.py`: HF `HiDream-ai/ReCo-Data`에서 `video_masks/add/`만 다운로드.
- `prepare_davis_reco.py`: WorldTraj/dynamicverse/DAVIS → ReCo 포맷 변환 (src=`inpaint_result_effecterase.mp4`, tar=`video_input.mp4` 832×480, text="Add a "+reasoning 첫 value 소문자화, instance mask 합집합 binary, 81프레임 패딩). 출력 `davis_data/`, val=앞 8개(overfit 실험용 train 겹침).

**학습 — validation & loss**
- `scripts/train.py`: 고정 holdout validation (`run_fixed_validation`) — 250 step마다 추론, metric을 wandb `val/*`로, 영상을 `val_videos/{source,edited,gt}` 3분류 key로 로깅. 학습 중 metric은 앞 32개, 영상 visualize는 앞 8개. `val/bg_*`·`fg_*`(masked) + `val/diff_*`(diff_bbox) + `val/bg_pure_*`(GT mask ∪ 생성객체 bbox 제외) 포함.
- `scripts/train.py`: 논문 loss 구현 — Eq.13 latent regularization(`mask_separation_loss`, `use_contrast_loss`, λ₁=1e-3), Eq.14–16 attention regularization(`use_attnscore_loss`, λ₂=1e-3), `use_attn_global_only`(L_edit 제외 L_global만). GT mask 없는 샘플은 자동 스킵. `diff_loss`/`loss_attnscore` 개별 로깅.
- `scripts/train.py`: `--max_steps` 인자, `RECO_DATA_ROOT`(데이터셋 루트 전환)·`RECO_EXCLUDE_VAL`(val 학습제외 토글) env.
- `reco_data_test_mix_data.py`: `ReCo_Dataset_train`에 `mask_video_folder` 인자 — GT object mask를 `diff_mask`로 로딩.
- `eval_val8.py`: holdout standalone 평가 스크립트 (GPU 분산), `--no_lora`(순수 base 모델), `--max_items`, edited 영역 분리 저장.
- 학습 config: `scripts/train_davis_overfit.sh`(run6, release ckpt 시작), `scripts/train_run7_contrast_attnglobal.sh`(run7).
- `ReCo-Data/ReCo-Data/add/add_val_configs.json`: validation holdout config (사람/동물 × 정적/이동 다양화, 최종 128개 — 앞 8개 visualize 고정).

**문서**
- `CLAUDE.md`: Claude Code용 저장소 가이드. val32 model-comparison 평가 파이프라인(root-level `eval_*.py`, `video_metrics.py`/`vbench_metrics.py` registry, `consolidate_model_compare.py`), 두 번째 conda env `reco_vbench`(VBench 계열 스크립트용), `EXPERIMENTS.log` 안내 추가.
- `CHANGELOG.md`, `EXPERIMENTS.log`: 변경/실험 기록.

### Changed
- 로컬 환경 적응: `train.py`/`train.sh`의 데이터·가중치 경로를 원저자 경로(mount+S3)에서 로컬 `checkpoints/`·`./ReCo-Data/ReCo-Data`로 변경, add task 단일 학습, GPU 4장(0–3), `WANDB_MODE=online`(키는 train.sh export).
- upstream "Reorganize project folders" 병합 대응: train/inference/dataset 스크립트가 `scripts/`로 이동됨에 따라 경로·import·문서 갱신.
- run_name 규약: 활성 loss를 이름에 표기 (`train_run{N}_{loss}`). 삭제된 wandb run id 재사용 시 init 타임아웃 → 새 이름 사용.
- validation 영상 저장: 2×2 grid → source/edited/gt 분리, `step_{N}/` 하위 폴더, `wandb.Video` 무시 fps 인자 제거.

### Fixed
- validation metric: 파이프라인 출력이 2×2 그리드(상단 [입력\|GT], 하단 [생성\|마스크])인 점 반영해 생성 edit 영역 슬라이싱 수정 (단일 출력 가정 broadcast 에러 해결).
- `tools/eval_step1_run_gemini_api.py`: `OpenAIVLMEngine`의 OpenAI 클라이언트에 `timeout=120.0, max_retries=0` 지정 — 응답 없는 연결이 무한 대기(do_poll 상태로 hang)하던 문제 해결, 자체 retry 루프가 정상 동작하도록 변경.

### 평가 — Gemini VLM (ReCo-Bench 9차원)
- `eval_gemini_val32.py`: val32 edited 영상을 Gemini(`gemini-2.5-flash`)로 9차원 채점(edit_accuracy·video_quality·naturalness 각 3) → 모델별 csv/json. `tools/eval_step1_run_gemini_api.py`의 엔진·프롬프트·프레임추출·파서 재사용. 무료 티어 분당 한도 대응으로 순차 + `--req_interval`(기본 13s).
- baseline 모델 8샘플 평가 완료. **무료 티어 일일 한도(`generate_content_free_tier_requests`, 20 req/day, gemini-2.5-flash)** 에 도달 — 나머지 5개 모델 평가는 일일 quota 리셋 또는 유료 billing 필요.
