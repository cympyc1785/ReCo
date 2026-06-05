"""Download only the `add/` subfolder of the HiDream-ai/ReCo-Data dataset
from Hugging Face into `./ReCo-Data/ReCo-Data/add/...` (preserving the dataset's
internal directory layout under `add/`).

Source: https://huggingface.co/datasets/HiDream-ai/ReCo-Data/tree/main/add
Target: /NHNHOME/WORKSPACE/0226010013_A/cympyc1785/video_generation/ReCo/ReCo-Data/ReCo-Data
"""
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "HiDream-ai/ReCo-Data"
TARGET = Path("/NHNHOME/WORKSPACE/0226010013_A/cympyc1785/video_generation/ReCo/ReCo-Data/ReCo-Data")


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(TARGET),
        allow_patterns=["add/**"],
        max_workers=8,
    )
    print(f"[done] downloaded to: {path}")


if __name__ == "__main__":
    main()
