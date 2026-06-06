# Changelog

이 프로젝트의 모든 주요 변경 사항을 이 파일에 기록한다.
형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.1.0/)를 따른다.

## [Unreleased]

### Fixed
- `train.py`/`eval_val8.py`: validation metric 계산 시 파이프라인 출력이 2×2 그리드(3328×960: 상단=[입력|GT], 하단=[생성|마스크])인 점을 반영해 생성 edit 영역 추출 슬라이싱 수정 (기존 코드는 단일 1664×480 출력을 가정해 broadcast 에러 발생).

### Changed
- `ReCo-Data/ReCo-Data/add/add_val_configs.json`: validation holdout을 8개 → **128개로 확장**. 맨 앞 8개는 기존 visualize set 그대로 유지, 추가 120개는 subject(사람/동물/탈것/사물) × action(이동/정적) 버킷별 균등 샘플링 (human 40, animal 40, vehicle 20, object 20; seed 777, src 영상 중복 배제). 전체 128개가 학습에서 자동 제외됨 (115,652 → 115,524).
- `scripts/train.py`: 학습 중 `run_fixed_validation`은 val config의 **맨 앞 32개로 metric 계산, 그중 앞 8개만 영상 visualize** (나머지 96개는 `eval_val8.py` offline 평가용).
- `scripts/train.py`: validation 영상 wandb 로깅을 2×2 그리드 → **`val_videos/source`·`val_videos/edited`·`val_videos/gt` 3개 분류 key**로 변경 (key에 데이터 이름 미포함, 샘플 인덱스 순 리스트로 로깅 → wandb slider 탐색). 로컬 저장도 `step_{N}/{idx}_{source|edited|gt}.mp4` + `{idx}_prompt.txt` 형식으로 변경. `wandb.Video`의 무시되는 `fps` 인자 제거 (파일 경로 입력 시 경고 발생).
- 데이터 경로 변경: `ReCo-Data/add` → `ReCo-Data/ReCo-Data/add`로 이동됨에 따라 `train.py`(train_dataloader, run_fixed_validation), `eval_val8.py`, `download_reco_add.py`의 경로를 `./ReCo-Data/ReCo-Data` 기준으로 수정.
- upstream "Reorganize project folders" 병합 후속 수정: `eval_val8.py`에 `scripts/` sys.path 추가 (`inference_reco_single`/`reco_data_test_mix_data`가 `scripts/`로 이동), `CLAUDE.md`의 명령/경로/아키텍처 문서를 `scripts/` 구조와 현재 워크플로(holdout 평가, wandb 규칙, loss 구성)에 맞게 갱신.
- `train.py`: `run_val_func`의 검증 영상 저장 경로에 `step_{N}` 하위 폴더 추가 (`all_videos/{run_name}/step_1500/gs_...mp4` 형태로 step별 분리 저장).
- `train.py`: `train_dataloader`의 데이터 경로를 원저자 경로(로컬 mount + S3)에서 로컬 `./ReCo-Data`로 변경, `task_list`를 `['add']`만 사용하도록 변경 (`sample_prob_list=[1.0]`, `read_video_from_local=True`).
- `train.sh`: `DIT_PATH`를 로컬 `checkpoints/Wan2.1-VACE-1.3B/` 가중치 경로로 변경, `GPU_NUM=8 → 4`, `CUDA_VISIBLE_DEVICES=3 → 0,1,2,3` (GPU 4–7은 다른 작업 사용 중), 미설정 시 `MASTER_ADDR=localhost:29500` 기본값 추가.
- `train.sh`: `WANDB_MODE`를 `offline → online`으로 변경, 빈 `WANDB_API_KEY` export 제거 (~/.netrc 로그인 정보 사용).
- `train.sh`: step 1500 체크포인트에서 재개하도록 `--resume_ckpt_folder` 추가.

### Added
- `reco_data_test_mix_data.py`: `ReCo_Dataset_train`에 `mask_video_folder` 인자 추가 — GT object mask 영상(`video_masks/{task}/{tar_name}.mp4`, 480×832)을 `diff_mask`로 로딩 가능 (기본값 None, 현재 학습에서는 미사용).
- `train.py`: 논문 loss 완전 재현 — ① Eq.13 latent regularization (`mask_separation_loss` 연결, pred x̂₁의 src/tar latent diff vs GT mask, λ₁=1e-3, `use_contrast_loss=True`) ② Eq.14–16 attention regularization (block별 attnscore(L_edit+L_global) 평균, λ₂=1e-3, `use_attnscore_loss=True`). GT mask 없는 샘플은 두 regularization 모두 자동 스킵. `diff_loss`/`loss_attnscore`가 wandb에 개별 기록됨.
- `train.sh`: `--run_name train_run2_contrast_attnscore` — 활성화된 loss 이름을 run_name에 표기, 처음부터 학습(resume 제거). (이전 이름 `train_base_run1_contrast_attnscore`는 wandb run 삭제 이력 때문에 재사용 시 init 타임아웃이 발생해 변경.)
- `train.py`: 고정 holdout validation 추가 (`run_fixed_validation`) — `ReCo-Data/add/add_val_configs.json`의 8개 영상으로 250 step마다 추론, PSNR/SSIM/LPIPS를 wandb(`val/*`)로 전송하고 생성 영상을 `wandb.Video`(`val_videos/*`) 및 로컬 `step_{N}/`에 저장. holdout은 학습 데이터에서 자동 제외(115,652 → 115,644). 기존 `run_val_func`(학습 batch 샘플 저장) 호출은 이 함수로 교체.
- `ReCo-Data/ReCo-Data/add/add_val_configs.json`: validation holdout config 신규 생성 (기존 `add_data_configs.json`은 수정하지 않음). 수차례 교체를 거쳐 최종 구성: 사람 정적 2(peeling/kneeling) + 동물 정적 2(sitting 개/고양이) + 사람 이동 2(walking/treadmill) + 동물 이동 2(산책 셰퍼드/걷는 말). 검토용 영상 사본: `all_results/val_set_review/`.
- 의존성: `lpips` 패키지 설치 (AlexNet 가중치 포함).
- `eval_val8.py`: holdout 8개 영상에 대한 standalone 평가 스크립트 (GPU별 분산 지원, PSNR/SSIM/LPIPS → `all_results/val8_eval/`).
- `download_reco_video_masks_add.py`: HF `HiDream-ai/ReCo-Data`에서 `video_masks/add/`만 `ReCo-Data/ReCo-Data/`로 받는 다운로드 스크립트.
- `CLAUDE.md`: Claude Code용 저장소 가이드 문서 추가.
- `CHANGELOG.md`: 본 파일 추가.
