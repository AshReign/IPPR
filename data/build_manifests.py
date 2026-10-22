"""Package 1: build the RipVIS manifests used by every other package.

Run the stages in order from the repo root:
    python -m data.build_manifests inventory   # parse COCO files, check counts against A1 Table 3.1
    python -m data.build_manifests split       # choose the 17 dev videos (frozen once written)
    python -m scripts.download_ripvis devvideos
    python -m data.build_manifests meta        # read fps / frame count / size from dev + val videos
    python -m data.build_manifests finalize    # YOLO image lists, single-rip list, frames index
    python -m data.build_manifests yolo-local  # machine-specific YOLO lists + yaml (not committed)

Committed outputs (data/manifests/): videos.csv, frames_all.csv, dev_videos.txt,
val_videos.txt, yolo_train_images.txt, yolo_dev_images.txt, single_rip_val.txt,
frames_index.csv. Local outputs (outputs/yolo/, gitignored): train.txt, dev.txt, ripvis.yaml.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "data" / "manifests"
SEED = 42

# RipVIS Readme, Known Limitation 2: annotated frame numbers do not match the real video frames.
MISMATCH = {f"RipVIS-{n}" for n in (
    "001 003 004 006 007 011 012 014 016 017 018 020 "
    "021 022 024 033 034 035 036 037 039 040").split()}

# Expected values from A1 Table 3.1 / 3.2 (computed from the same COCO files).
EXPECTED = {
    ("train", "videos"): 112, ("train", "rip_videos"): 90, ("train", "norip_videos"): 22,
    ("train", "annotated_frames"): 14616, ("train", "masks"): 15685, ("train", "multi_rip_videos"): 43,
    ("val", "videos"): 36, ("val", "rip_videos"): 30, ("val", "norip_videos"): 6,
    ("val", "annotated_frames"): 4349, ("val", "masks"): 4556, ("val", "multi_rip_videos"): 15,
}
DEV_MULTI, DEV_SINGLE, DEV_NORIP = 7, 7, 3

NAME = re.compile(r"(RipVIS-(?:NR-)?\d{3})_(\d{5})")


def ripvis_root(arg):
    if arg:
        return Path(arg)
    from configs import paths
    return Path(paths()["ripvis_root"])


# ---------------------------------------------------------------- inventory
def frames_table(coco_path: Path, split: str) -> pd.DataFrame:
    """One row per annotated image: video, frame number, number of ground-truth masks."""
    coco = json.loads(coco_path.read_text())
    masks_per_image = pd.Series([a["image_id"] for a in coco["annotations"]], dtype="int64").value_counts()
    rows = []
    for im in coco["images"]:
        m = NAME.search(im["file_name"])
        if m is None:
            raise ValueError(f"Unexpected file name: {im['file_name']}")
        base = Path(im["file_name"]).name
        rows.append(dict(source_split=split, video_id=m.group(1), frame_idx=int(m.group(2)),
                         image_id=im["id"], image_rel=f"{split}/images/{base}",
                         n_gt=int(masks_per_image.get(im["id"], 0))))
    return pd.DataFrame(rows).sort_values(["video_id", "frame_idx"]).reset_index(drop=True)


def check_counts(frames: pd.DataFrame, videos: pd.DataFrame) -> None:
    problems = []
    for split in ("train", "val"):
        v, f = videos[videos.source_split == split], frames[frames.source_split == split]
        got = {
            "videos": len(v), "rip_videos": int(v.has_rip.sum()), "norip_videos": int((~v.has_rip).sum()),
            "annotated_frames": len(f), "masks": int(f.n_gt.sum()),
            "multi_rip_videos": int((v.max_rips_per_frame >= 2).sum()),
        }
        for k, val in got.items():
            exp = EXPECTED[(split, k)]
            flag = "OK " if val == exp else "BAD"
            print(f"  [{flag}] {split:5s} {k:17s} got {val:6d}  expected {exp:6d}")
            if val != exp:
                problems.append(f"{split} {k}")
    if problems:
        raise SystemExit(f"Counts differ from A1 Table 3.1: {problems}. Find out why before continuing.")


def stage_inventory(root: Path):
    frames = pd.concat([
        frames_table(root / "train" / "coco_annotations" / "train.json", "train"),
        frames_table(root / "val" / "coco_annotations" / "val.json", "val"),
    ], ignore_index=True)
    videos = (frames.groupby(["source_split", "video_id"])
              .agg(n_annotated=("frame_idx", "size"), max_rips_per_frame=("n_gt", "max"),
                   first_frame=("frame_idx", "min"), last_frame=("frame_idx", "max"))
              .reset_index())
    videos["has_rip"] = ~videos.video_id.str.contains("NR")
    videos["in_mismatch_list"] = videos.video_id.isin(MISMATCH)
    videos["split"] = videos.source_split          # refined by `split`
    print("Checking counts against A1 Table 3.1:")
    check_counts(frames, videos)

    missing = [p for p in frames.image_rel if not (root / p).exists()]
    print(f"Image files found: {len(frames) - len(missing)}/{len(frames)}"
          + (f"  (first missing: {missing[0]})" if missing else ""))

    MANIFESTS.mkdir(parents=True, exist_ok=True)
    frames.to_csv(MANIFESTS / "frames_all.csv", index=False)
    videos.to_csv(MANIFESTS / "videos.csv", index=False)
    print(f"Wrote frames_all.csv ({len(frames)} rows) and videos.csv ({len(videos)} rows)")


# ---------------------------------------------------------------- split
def stage_split(force: bool):
    dev_file = MANIFESTS / "dev_videos.txt"
    if dev_file.exists() and not force:
        raise SystemExit("dev_videos.txt already exists and is frozen. Use --force only if the team agrees.")
    videos = pd.read_csv(MANIFESTS / "videos.csv")
    rng = np.random.default_rng(SEED)

    pool = videos[(videos.source_split == "train") & ~videos.in_mismatch_list]
    rip = pool[pool.has_rip]

    def pick(ids, k):
        ids = sorted(ids)                          # sort first so the draw is reproducible
        if len(ids) < k:
            raise SystemExit(f"Only {len(ids)} candidates, need {k}")
        return list(rng.choice(ids, size=k, replace=False))

    dev = (pick(rip[rip.max_rips_per_frame >= 2].video_id, DEV_MULTI)
           + pick(rip[rip.max_rips_per_frame == 1].video_id, DEV_SINGLE)
           + pick(pool[~pool.has_rip].video_id, DEV_NORIP))

    videos["split"] = videos.source_split
    videos.loc[videos.video_id.isin(dev), "split"] = "dev"
    excluded_val = (videos.source_split == "val") & videos.in_mismatch_list
    videos.loc[excluded_val, "split"] = "val_excluded"
    val = sorted(videos.loc[videos.split == "val", "video_id"])

    assert len(dev) == 17 and len(set(dev)) == 17
    assert not set(dev) & MISMATCH
    print(f"Dev: {len(dev)} videos | final val: {len(val)} videos "
          f"({int(videos[videos.split == 'val'].has_rip.sum())} rip) | "
          f"val excluded: {sorted(videos.loc[excluded_val, 'video_id'])}")

    dev_file.write_text("\n".join(sorted(dev)) + "\n")
    (MANIFESTS / "val_videos.txt").write_text("\n".join(val) + "\n")
    videos.to_csv(MANIFESTS / "videos.csv", index=False)
    print(videos[videos.split == "dev"][["video_id", "has_rip", "max_rips_per_frame", "n_annotated"]]
          .to_string(index=False))


# ---------------------------------------------------------------- meta
def video_path(root: Path, row) -> Path:
    return root / row.source_split / "videos" / f"{row.video_id}.mp4"


def stage_meta(root: Path):
    import cv2
    videos = pd.read_csv(MANIFESTS / "videos.csv")
    frames = pd.read_csv(MANIFESTS / "frames_all.csv")
    need = videos[videos.split.isin(["dev", "val"])]
    missing = [r.video_id for r in need.itertuples() if not video_path(root, r).exists()]
    if missing:
        raise SystemExit(f"Videos not downloaded yet: {missing}")

    meta_cols = ["fps", "n_frames", "width", "height", "sample_step", "sampling_rate", "last_frame_ok"]
    rows = []
    for r in need.itertuples():
        cap = cv2.VideoCapture(str(video_path(root, r)))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        idx = np.sort(frames.loc[frames.video_id == r.video_id, "frame_idx"].to_numpy())
        step = float(np.median(np.diff(idx))) if len(idx) > 1 else np.nan
        rows.append(dict(video_id=r.video_id, fps=fps, n_frames=n, width=w, height=h, sample_step=step,
                         sampling_rate=fps / step if step else np.nan, last_frame_ok=bool(idx.max() < n)))
    videos = videos.drop(columns=[c for c in meta_cols if c in videos]).merge(
        pd.DataFrame(rows), on="video_id", how="left")

    bad = videos[videos.split.isin(["dev", "val"]) & (videos.last_frame_ok == False)]  # noqa: E712
    print(videos[videos.split.isin(["dev", "val"])]
          [["video_id", "split", "fps", "n_frames", "width", "height", "sampling_rate", "last_frame_ok"]]
          .to_string(index=False))
    if len(bad):
        print(f"\nWARNING: annotated frame beyond video end in {list(bad.video_id)}; "
              "these may have the frame-mismatch problem too. Raise it with the team.")
    videos.to_csv(MANIFESTS / "videos.csv", index=False)


# ---------------------------------------------------------------- finalize
def stage_finalize():
    videos = pd.read_csv(MANIFESTS / "videos.csv")
    frames = pd.read_csv(MANIFESTS / "frames_all.csv").merge(videos[["video_id", "split", "fps"]], on="video_id")

    train_imgs = frames[frames.split == "train"].image_rel
    dev_imgs = frames[frames.split == "dev"].image_rel
    leak = set(frames[frames.split == "train"].video_id) & set(frames[frames.split == "dev"].video_id)
    assert not leak, f"Video in both train and dev: {leak}"
    (MANIFESTS / "yolo_train_images.txt").write_text("\n".join(train_imgs) + "\n")
    (MANIFESTS / "yolo_dev_images.txt").write_text("\n".join(dev_imgs) + "\n")

    single = videos[(videos.split == "val") & videos.has_rip & (videos.max_rips_per_frame == 1)]
    (MANIFESTS / "single_rip_val.txt").write_text("\n".join(sorted(single.video_id)) + "\n")

    idx = frames[frames.split.isin(["dev", "val"])].copy()
    if idx.fps.isna().any():
        raise SystemExit("fps missing; run the `meta` stage first.")
    idx["t_sec"] = idx.frame_idx / idx.fps
    idx[["video_id", "split", "frame_idx", "t_sec", "n_gt", "image_id", "image_rel"]].to_csv(
        MANIFESTS / "frames_index.csv", index=False)

    print(f"YOLO images: train {len(train_imgs)}, dev {len(dev_imgs)}")
    print(f"Single-rip val videos: {len(single)}")
    print(f"frames_index.csv: {len(idx)} annotated dev/val frames")


# ---------------------------------------------------------------- yolo-local
def stage_yolo_local(root: Path):
    """Absolute-path image lists + dataset yaml for THIS machine (outputs/ is gitignored)."""
    out = REPO / "outputs" / "yolo"
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, src in (("train", "yolo_train_images.txt"), ("dev", "yolo_dev_images.txt")):
        rel = (MANIFESTS / src).read_text().split()
        paths = [(root / p).resolve() for p in rel]
        (out / f"{name}.txt").write_text("\n".join(p.as_posix() for p in paths) + "\n")
        # Ultralytics finds labels by swapping the last 'images' folder for 'labels'
        labels = [p.parent.parent / "labels" / (p.stem + ".txt") for p in paths]
        report[name] = (sum(p.exists() for p in paths), sum(lb.exists() for lb in labels), len(paths))
    (out / "ripvis.yaml").write_text(
        f"train: {(out / 'train.txt').as_posix()}\nval: {(out / 'dev.txt').as_posix()}\nnames:\n  0: rip_current\n")
    for name, (imgs, labs, n) in report.items():
        print(f"{name}: images found {imgs}/{n}, label files found {labs}/{n} "
              "(no-rip images may have no label file; that is fine)")
    print(f"Wrote {out / 'ripvis.yaml'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["inventory", "split", "meta", "finalize", "yolo-local"])
    ap.add_argument("--root", help="override ripvis_root from configs/paths.yaml")
    ap.add_argument("--force", action="store_true", help="re-draw the dev split (team decision only)")
    a = ap.parse_args()
    if a.stage == "inventory":
        stage_inventory(ripvis_root(a.root))
    elif a.stage == "split":
        stage_split(a.force)
    elif a.stage == "meta":
        stage_meta(ripvis_root(a.root))
    elif a.stage == "finalize":
        stage_finalize()
    else:
        stage_yolo_local(ripvis_root(a.root))


if __name__ == "__main__":
    main()
