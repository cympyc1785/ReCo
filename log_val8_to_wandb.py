"""eval_val8 결과(metrics_proc*.json)를 집계해 기존 wandb run에 logging한다."""
import glob
import json
import argparse

import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--json_dir", type=str, default="all_results/val8_eval")
parser.add_argument("--project", type=str, default="ReCo")
parser.add_argument("--entity", type=str, default="VCAI_Vid")
parser.add_argument("--run_id", type=str, default="train_base_run1")
parser.add_argument("--ckpt_step", type=int, required=True)
args = parser.parse_args()

results = []
for p in sorted(glob.glob(f"{args.json_dir}/metrics_proc*.json")):
    results.extend(json.load(open(p)))

assert len(results) > 0, "no metric jsons found"
psnr = sum(r["psnr"] for r in results) / len(results)
ssim = sum(r["ssim"] for r in results) / len(results)
lpips_v = sum(r["lpips"] for r in results) / len(results)

print(f"videos: {len(results)}")
for r in results:
    print(f"  {r['video'][:60]}: PSNR {r['psnr']:.3f}, SSIM {r['ssim']:.4f}, LPIPS {r['lpips']:.4f}")
print(f"mean: PSNR {psnr:.3f}, SSIM {ssim:.4f}, LPIPS {lpips_v:.4f}")

run = wandb.init(entity=args.entity, project=args.project, id=args.run_id, name=args.run_id, resume="allow")
run.log({
    "val/psnr": psnr, "val/ssim": ssim, "val/lpips": lpips_v,
    "trainer/global_step": args.ckpt_step,
})
run.finish()
print(f"logged to wandb run {args.run_id} (ckpt step {args.ckpt_step})")
