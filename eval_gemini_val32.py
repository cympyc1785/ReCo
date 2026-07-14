"""val32 edited 영상을 Gemini(ReCo-Bench 9차원)로 평가 → 모델별 json + csv.

tools/eval_step1_run_gemini_api.py의 엔진/프롬프트/프레임추출/파서를 재사용.
각 샘플: src | edited 를 hstack(224)해서 Gemini에 instruction과 함께 보내
edit_accuracy(3) + video_quality(3) + naturalness(3) = 9개 점수(1~10) 채점.

Usage:
  python eval_gemini_val32.py --gen_dir all_results/val32_run5_2000 --name run5 \
      --out_dir all_results/gemini
"""
import argparse, csv, json, os, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools'))
import importlib
_g = importlib.import_module('eval_step1_run_gemini_api')
OpenAIVLMEngine = _g.OpenAIVLMEngine
sys_prompt = _g.benchmark_score_eval_sys_prompt
get_videos_from_path = _g.get_videos_from_path
load_output_as_json = _g.load_output_as_json

KEY = 'sk-YOUR_API_TOKENS'
BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/openai/'
MODEL = 'gemini-2.5-flash'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen_dir', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--data_root', default='ReCo-Data/ReCo-Data')
    ap.add_argument('--val_json', default='ReCo-Data/ReCo-Data/add/add_val_configs.json')
    ap.add_argument('--max_items', type=int, default=32)
    ap.add_argument('--fps', type=int, default=2)
    ap.add_argument('--req_interval', type=float, default=13.0, help='요청 간 간격(초), 무료 티어 분당5 대응')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    os.environ['OPENAI_API_KEY'] = KEY
    engine = OpenAIVLMEngine(model_name=MODEL, max_tokens=8192, custom_base_url=BASE_URL)

    val = json.load(open(args.val_json))[:args.max_items]
    stride = max(1, 81 // (args.fps * 5))
    data_list, meta = [], []
    frame_root = os.path.join(args.out_dir, f'{args.name}_frames')
    for i, it in enumerate(val):
        src = os.path.join(args.data_root, it['src_video'])
        edited = os.path.join(args.gen_dir, f'{i:03d}_edited.mp4')
        folder = get_videos_from_path(video_src_path=src, video_tar_path=edited,
                                      resolution_h=224, out_folder=os.path.join(frame_root, f'{i:03d}'))
        imgs = sorted(os.path.join(folder, f) for f in os.listdir(folder))[::stride]
        data_list.append({'image_path': imgs,
                           'question': sys_prompt + f"instruction: {it['instruction_final_refine'].strip()}"})
        meta.append({'idx': i, 'phrase': it['instruction_final_refine'].strip()})

    # 무료 티어 분당 5요청 한도 → 순차 + 간격
    import time
    from tqdm import tqdm
    responses = []
    for k, item in enumerate(tqdm(data_list, desc=f'gemini {args.name}')):
        responses.append(engine.process_single_item(item))
        if k < len(data_list) - 1:
            time.sleep(args.req_interval)

    rows = []
    for m, resp in zip(meta, responses):
        parsed = load_output_as_json(resp)
        m['response'] = parsed
        # 9개 점수 평탄화
        flat = {}
        if isinstance(parsed, dict):
            for cat in ['edit_accuracy', 'video_quality', 'naturalness']:
                sc = parsed.get(cat, {}).get('scores') if isinstance(parsed.get(cat), dict) else None
                if isinstance(sc, list):
                    for j, v in enumerate(sc):
                        flat[f'{cat}_{j+1}'] = v
        rows.append({**{'idx': m['idx'], 'phrase': m['phrase']}, **flat})

    json.dump(meta, open(os.path.join(args.out_dir, f'{args.name}_gemini.json'), 'w'), ensure_ascii=False, indent=2)
    if rows:
        cols = ['idx', 'phrase'] + [f'{c}_{j}' for c in ['edit_accuracy','video_quality','naturalness'] for j in (1,2,3)]
        with open(os.path.join(args.out_dir, f'{args.name}_gemini.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for r in rows: w.writerow({k: r.get(k) for k in cols})
        # 집계
        import numpy as np
        agg = {}
        for c in cols[2:]:
            vals = [r[c] for r in rows if isinstance(r.get(c), (int, float))]
            agg[c] = round(float(np.mean(vals)), 3) if vals else None
        json.dump(agg, open(os.path.join(args.out_dir, f'{args.name}_gemini_summary.json'), 'w'), indent=2)
        print(f'[{args.name}] 집계:', agg)
    print('saved:', os.path.join(args.out_dir, f'{args.name}_gemini.csv'))


if __name__ == '__main__':
    main()
