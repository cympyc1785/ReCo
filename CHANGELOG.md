# Changelog

이 프로젝트의 모든 주요 변경 사항을 이 파일에 기록한다.
형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.1.0/)를 따른다.

## [Unreleased]

### Fixed
- `train.py`/`eval_val8.py`: validation metric 계산 시 파이프라인 출력이 2×2 그리드(3328×960: 상단=[입력|GT], 하단=[생성|마스크])인 점을 반영해 생성 edit 영역 추출 슬라이싱 수정 (기존 코드는 단일 1664×480 출력을 가정해 broadcast 에러 발생).

### Changed
- `scripts/train.py`: 학습 중 validation에 신규 metric 통합 — `val/bg_*`·`fg_*`(GT mask 기준 masked psnr/ssim/lpips), `val/diff_iou`·`diff_area_ratio`·`diff_success`·`diff_outside_mask`(diff_bbox), **`val/bg_pure_psnr`·`bg_pure_lpips`**(GT mask ∪ 생성객체 diff-bbox 사각형 제외한 순수 배경 — 위치 어긋남 벌점과 진짜 배경 훼손을 분리). val dataset에 mask 로딩 연결, 중복 LPIPS 로더 제거.
- `scripts/train.py`: `--max_steps` 인자 추가 (Lightning Trainer 전달, -1=무제한). run5 ablation을 위해 `use_contrast_loss`/`use_attnscore_loss`를 다시 False로 (run4와 loss만 다른 쌍둥이 실험).
- `scripts/train.sh`: `--run_name train_run5_base --max_steps 2000`.
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
- `scripts/train.py`: `use_attn_global_only` 플래그 — attention regularization에서 **L_global만 적용** (edit-area constraint L_edit 제외; block별 [L_edit, L_global] 쌍에서 홀수 인덱스만 사용). run7 구성: contrast(λ1=1e-3) + attn-global(λ2=1e-3).
- `scripts/train_run7_contrast_attnglobal.sh`: run7 학습 config (ReCo-Data, from scratch, max_steps 2000 — run4/run5와 동일 조건 ablation).
- `prepare_davis_reco.py`: WorldTraj/dynamicverse/DAVIS → ReCo 포맷 변환 (src=`inpaint_result_effecterase.mp4`, tar=`video_input.mp4` 832×480 리사이즈, text="Add a "+reasoning 첫 value, **instance mask 합집합 binary** → video_masks mp4, 공통 T 절단 후 81프레임 패딩). 출력: `davis_data/`, val=앞 8개(train과 의도적 겹침 — overfit 실험).
- `scripts/train.py`: `RECO_DATA_ROOT` env로 데이터셋 루트 전환, `RECO_EXCLUDE_VAL=0`으로 val 학습 제외 비활성화 가능.
- `scripts/train_davis_overfit.sh`: run6(DAVIS overfit) 전용 학습 config — release ckpt에서 시작, max_steps 2000.
- `eval_vace_gtmask.py`: 원조 VACE 용법 평가 스크립트 — in-context concat 대신 단일 832폭 입력으로 src + **GT object mask**(생성 영역) + instruction을 줘서 mask 영역만 inpainting. "mask가 주어졌을 때의 상한선" 측정용 (ReCo 목표 = mask 없이 이 수준). grid 영상 검수본 포함 저장.
- `eval_val8.py`: `--no_lora` 옵션 추가 — LoRA 없이 순수 Wan2.1-VACE-1.3B base로 추론 (base 모델 성능 기준점 측정용).
- `video_metrics.py`: 모듈형 비디오 metric 라이브러리 (registry 방식, `get_metric(name)`으로 선택 로딩) — `psnr`/`ssim`/`lpips` + **masked 변형**(`region='foreground'|'background'`, GT mask로 배경 보존·객체 영역 분리 측정) + **FVD**(I3D). identity/masked-분리/FVD sanity 테스트 통과. GPU 경로 추가(PSNR torch, SSIM torchmetrics — CPU와 일치 검증 diff≤1e-5).
- `visualize_bbox_metrics.py`: bbox/diff metric 검증용 시각화 — `{idx}_bbox.mp4`(edited 위 GT파랑/diff초록/det빨강 사각형), `{idx}_diff.mp4`([edited | RGB diff+threshold 틴트] concat). 출력: `all_results/metric_viz/{model}/`.
- `eval_metrics_suite.py`: 표준 평가 스크립트 — 생성 폴더에 전체 metric suite 적용, **per-sample CSV 저장**(`metrics_full.csv`, 영상 경로 포함 — 낮은/높은 샘플 직접 확인용), 집계 wandb 로깅, best/worst 샘플 요약 출력. 앞으로 모든 ckpt 평가의 표준 도구.
- `video_metrics.py`: 삽입 위치/스케일 bbox metric 3종 추가 — `diff_bbox`(RGB diff 기반, GT mask grid search로 캘리브레이션: percentile 75/floor 8/min_area 800, IoU 0.34→0.455), `detection_bbox`(GroundingDINO), `hybrid_bbox`(diff 영역과 겹치는 검출만 채택). val32 16×2 비교에서 세 방식 IoU 수렴(±0.002) 확인, calibrated diff를 주 지표로 채택. `extract_add_phrase` 헬퍼(명사구 내 쉼표 보존) 포함.
- `submodule/frechet_video_distance-pytorch`: FVD 계산용 fork(cympyc1785) clone — I3D 가중치 포함, `video_metrics.py`가 package alias로 로드.
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
