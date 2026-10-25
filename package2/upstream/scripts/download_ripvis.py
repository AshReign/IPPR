"""Download the parts of RipVIS v1.8.4 that this project needs.

Usage (from the repo root, with the virtual environment active):
    python -m scripts.download_ripvis base      # annotations, sampled images, val videos, docs
    python -m scripts.download_ripvis devvideos # the 17 dev videos (after `split` has run)

Files go to `ripvis_root` from configs/paths.yaml. Nothing here is ever committed:
the RipVIS licence forbids redistribution.
"""
import argparse
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Irikos/RipVIS"
MANIFESTS = Path(__file__).resolve().parents[1] / "data" / "manifests"

BASE_PATTERNS = [
    "train/coco_annotations/*",
    "train/yolo_annotations.zip",
    "train/sampled_images.zip",
    "val/*",                      # val annotations, images, YOLO labels and all 36 val videos
    "*.pdf",
    "Readme.md",
    "compute_*.py",
]


def ripvis_root(arg):
    if arg:
        return Path(arg)
    from configs import paths
    return Path(paths()["ripvis_root"])


def unzip_all(root: Path):
    """Extract every .zip next to itself and print its top-level entries for a layout check."""
    for z in sorted(root.rglob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            tops = sorted({n.split("/")[0] for n in zf.namelist() if n.strip("/")})
            print(f"{z.relative_to(root)} -> top-level entries: {tops[:10]}")
            zf.extractall(z.parent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["base", "devvideos"])
    ap.add_argument("--root", help="override ripvis_root from configs/paths.yaml")
    args = ap.parse_args()
    root = ripvis_root(args.root)
    root.mkdir(parents=True, exist_ok=True)

    if args.stage == "base":
        snapshot_download(REPO_ID, repo_type="dataset", local_dir=root, allow_patterns=BASE_PATTERNS)
        unzip_all(root)
    else:
        dev_file = MANIFESTS / "dev_videos.txt"
        if not dev_file.exists():
            raise SystemExit("Run `python -m data.build_manifests split` first.")
        ids = dev_file.read_text().split()
        snapshot_download(REPO_ID, repo_type="dataset", local_dir=root,
                          allow_patterns=[f"train/videos/{v}.mp4" for v in ids])
        missing = [v for v in ids if not (root / "train" / "videos" / f"{v}.mp4").exists()]
        print("Missing dev videos:", missing or "none")


if __name__ == "__main__":
    main()
