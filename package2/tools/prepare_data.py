"""Create an isolated P2 dataset view from immutable P1 manifests and local RipVIS.

No source file or upstream manifest is changed. Run with Python on Windows or WSL.
The default YAML paths target WSL, where the existing CUDA environment resides.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from PIL import Image


NAME = re.compile(r"^(RipVIS-(?:NR-)?\d+)_(\d+)\.(?:jpg|jpeg|png)$", re.I)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def emit(payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def write_owned(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return "existing_hardlink"
        if destination.stat().st_size == source.stat().st_size and sha256(destination) == sha256(source):
            return "existing_identical_copy"
        raise RuntimeError(f"Refusing to replace differing existing file: {destination}")
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        with source.open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
        if sha256(source) != sha256(destination):
            raise RuntimeError(f"Copy integrity failure: {destination}")
        return "copy"


def platform_path(path, target):
    text = Path(path).resolve().as_posix()
    if target == "linux" and re.match(r"^[A-Za-z]:/", text):
        return "/mnt/" + text[0].lower() + text[2:]
    if target == "windows" and text.startswith("/mnt/"):
        parts = text.split("/")
        return parts[2].upper() + ":/" + "/".join(parts[3:])
    return text


def safe_relative_image(rel):
    value = PurePosixPath(rel)
    if value.is_absolute() or ".." in value.parts or len(value.parts) != 3:
        raise ValueError(f"Unsafe image path in frozen manifest: {rel}")
    if value.parts[0] not in ("train", "val") or value.parts[1] != "images" or not NAME.match(value.name):
        raise ValueError(f"Unexpected image path: {rel}")
    return value


def audit_label(text, source):
    count = 0
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 7 or len(fields) % 2 != 1:
            raise ValueError(f"Not a polygon row: {source}:{number}, {len(fields)} tokens")
        if fields[0] != "0":
            raise ValueError(f"Unexpected class: {source}:{number}: {fields[0]}")
        coords = [float(value) for value in fields[1:]]
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in coords):
            raise ValueError(f"Nonfinite or unnormalized polygon: {source}:{number}")
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--yaml-platform", choices=("linux", "windows"), default="linux")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    started = time.perf_counter()
    root = args.project_root.resolve()
    p2 = root / "package2"
    upstream = p2 / "upstream"
    manifests = upstream / "data/manifests"
    sources = root / "data/ripvis"
    destination = p2 / "dataset"
    configs = p2 / "configs"
    baseline = json.loads((p2 / "upstream_manifest.json").read_text(encoding="utf-8"))
    frozen_hashes = {path.name: sha256(path) for path in sorted(manifests.iterdir()) if path.is_file()}
    for relative, expected in baseline["files"].items():
        if sha256(upstream / relative) != expected:
            raise RuntimeError(f"Upstream changed from acquired main: {relative}")

    videos = read_csv(manifests / "videos.csv")
    video_info = {row["video_id"]: row for row in videos}
    frames = read_csv(manifests / "frames_all.csv")
    frame_index = read_csv(manifests / "frames_index.csv")
    train = (manifests / "yolo_train_images.txt").read_text().split()
    dev = (manifests / "yolo_dev_images.txt").read_text().split()
    dev_videos = (manifests / "dev_videos.txt").read_text().split()
    val_videos = (manifests / "val_videos.txt").read_text().split()
    val = [row["image_rel"] for row in frames if video_info[row["video_id"]]["split"] == "val"]
    frame_by_path = {row["image_rel"]: row for row in frames}
    if len(frame_by_path) != len(frames):
        raise ValueError("Duplicated image paths in P1 frames_all.csv")
    for split, rows in (("train", train), ("dev", dev), ("val", val)):
        assert len(rows) == len(set(rows)), f"Duplicate image in {split}"
        for rel in rows:
            safe_relative_image(rel)
            assert video_info[frame_by_path[rel]["video_id"]]["split"] == split, rel
    train_ids = {frame_by_path[rel]["video_id"] for rel in train}
    dev_ids = {frame_by_path[rel]["video_id"] for rel in dev}
    val_ids = {frame_by_path[rel]["video_id"] for rel in val}
    assert len(train_ids) == 95 and len(dev_ids) == 17 and len(val_ids) == 30
    assert dev_ids == set(dev_videos) and val_ids == set(val_videos)
    assert not (train_ids & dev_ids or train_ids & val_ids or dev_ids & val_ids)
    assert len(train) == 12195 and len(dev) == 2421
    excluded = sorted(row["video_id"] for row in videos if row["split"] == "val_excluded")
    assert excluded == [f"RipVIS-{number:03d}" for number in (1, 7, 12, 14, 24, 39)]
    assert {row["image_rel"] for row in frame_index} == set(dev + val)
    for row in frame_index:
        info = video_info[row["video_id"]]
        frame = int(row["frame_idx"])
        assert 0 <= frame < int(float(info["n_frames"])), row
        assert abs(float(row["t_sec"]) - frame / float(info["fps"])) < 1e-9, row

    label_counts = {}
    label_archive_details = {}
    link_counts = Counter()
    for split in ("train", "val"):
        archive = sources / split / "yolo_annotations.zip"
        selected_count = 0
        polygon_count = 0
        with zipfile.ZipFile(archive) as handle:
            for entry in handle.infolist():
                rel = PurePosixPath(entry.filename)
                if rel.parts[:2] != ("yolo_annotations", "labels") or entry.is_dir():
                    continue
                if len(rel.parts) != 3 or rel.suffix != ".txt" or not NAME.match(rel.with_suffix(".jpg").name):
                    raise ValueError(f"Unexpected ZIP label: {entry.filename}")
                data = handle.read(entry)
                text = data.decode("utf-8-sig")
                count = audit_label(text, entry.filename)
                assert count > 0, entry.filename
                target = destination / split / "labels" / rel.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if target.read_bytes() != data:
                        raise RuntimeError(f"Refusing to replace differing label: {target}")
                else:
                    with target.open("xb") as out:
                        out.write(data)
                label_counts[f"{split}/images/{rel.stem}.jpg"] = count
                selected_count += 1
                polygon_count += count
        label_archive_details[split] = {"sha256": sha256(archive), "base_labels": selected_count,
                                        "polygons": polygon_count, "all_class_zero": True,
                                        "all_coordinates_finite_normalized": True,
                                        "all_rows_polygon_not_bbox": True}
        coco_name = "train.json" if split == "train" else "val.json"
        link_counts[link_or_copy(sources / split / "coco_annotations" / coco_name,
                                destination / split / "coco_annotations" / coco_name)] += 1
    counts = {}
    for split, rows in (("train", train), ("dev", dev), ("val", val)):
        for rel in rows:
            assert label_counts.get(rel, 0) == int(frame_by_path[rel]["n_gt"]), rel
        present = sum(rel in label_counts for rel in rows)
        counts[split] = {"images": len(rows), "labeled_images": present,
                         "empty_images": len(rows) - present,
                         "polygons": sum(label_counts.get(rel, 0) for rel in rows),
                         "videos": len({frame_by_path[rel]["video_id"] for rel in rows})}
    assert counts["dev"] == {"images": 2421, "labeled_images": 2149, "empty_images": 272,
                            "polygons": counts["dev"]["polygons"], "videos": 17}
    # Link all base images, including excluded val, without changing their role in manifests.
    # Only train/dev lists are referenced by a training YAML.
    for row in frames:
        rel = safe_relative_image(row["image_rel"])
        source = sources / rel.parts[0] / "extracted/sampled_images/images" / rel.name
        if not source.is_file():
            raise FileNotFoundError(source)
        link_counts[link_or_copy(source, destination / str(rel))] += 1
    for video in videos:
        if video["split"] in ("dev", "val"):
            rel = Path(video["source_split"]) / "videos" / (video["video_id"] + ".mp4")
            link_counts[link_or_copy(sources / rel, destination / rel)] += 1

    pilot = sorted(random.Random(42).sample(train, 320))
    lists = {"train": train, "dev": dev, "pilot320": pilot}
    for name, rows in lists.items():
        write_owned(configs / (name + ".txt"), "\n".join(platform_path(destination / rel, args.yaml_platform) for rel in rows) + "\n")
        # Stable dataset-relative copies document exact membership without machine paths.
        write_owned(configs / (name + "_relative.txt"), "\n".join(rows) + "\n")
    for name, train_list in (("training", "train"), ("pilot", "pilot320")):
        write_owned(configs / (name + ".yaml"),
                    f"train: {platform_path(configs / (train_list + '.txt'), args.yaml_platform)}\n"
                    f"val: {platform_path(configs / 'dev.txt', args.yaml_platform)}\n"
                    "names:\n  0: rip_current\n")
    emit({"stage": "dataset_view_and_configs_ready", "counts": counts, "pilot_images": 320,
          "yaml_platform": args.yaml_platform, "full_decode_audit": "starting"})

    coco_dimensions = {}
    for split, name in (("train", "train.json"), ("val", "val.json")):
        coco = json.loads((sources / split / "coco_annotations" / name).read_text())
        assert len(coco["categories"]) == 1, coco["categories"]
        for image in coco["images"]:
            coco_dimensions[f"{split}/images/{Path(image['file_name']).name}"] = (int(image["width"]), int(image["height"]))
    def check_image(row):
        rel = row["image_rel"]
        with Image.open(destination / rel) as image:
            if image.size != coco_dimensions[rel]:
                raise ValueError(f"COCO image dimension mismatch: {rel}")
            image.load()  # Decode every pixel; do not count header-only checks as full decode.
            return rel, image.size
    decoded = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for _, dimensions in executor.map(check_image, frames):
            decoded += 1
            if decoded % 1000 == 0:
                emit({"stage": "decode_audit", "completed": decoded, "total": len(frames)})
    assert frozen_hashes == {path.name: sha256(path) for path in sorted(manifests.iterdir()) if path.is_file()}
    selected_video_records = []
    for video in videos:
        if video["split"] not in ("dev", "val"):
            continue
        path = sources / video["source_split"] / "videos" / (video["video_id"] + ".mp4")
        selected_video_records.append({"video_id": video["video_id"], "split": video["split"],
                                       "source_path": str(path), "bytes": path.stat().st_size,
                                       "p1_fps": float(video["fps"]), "p1_frame_count": int(float(video["n_frames"])),
                                       "p1_width": int(float(video["width"])), "p1_height": int(float(video["height"]))})
    report = {
        "status": "image_and_label_preflight_passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "p1_commit": baseline["commit"], "p1_acquisition": baseline["acquired_via"],
        "dataset_revision": json.loads((root / "data/ripvis_manifest.json").read_text())["sha"],
        "frozen_manifest_sha256": frozen_hashes, "upstream_unchanged": True,
        "source_files_preserved": True, "dataset_view": str(destination),
        "link_operations": dict(link_counts), "split_counts": counts,
        "splits": {"train": sorted(train_ids), "dev": sorted(dev_ids), "val": sorted(val_ids), "val_excluded": excluded},
        "train_dev_val_video_disjoint": True, "additional_training_images_used": False,
        "label_archives": label_archive_details,
        "all_image_decode": {"passed": decoded, "total": len(frames), "all_dimensions_match_coco": True,
                             "includes_excluded_val_for_integrity_only": True},
        "pilot": {"size": 320, "seed": 42, "algorithm": "sorted(random.Random(42).sample(exact_P1_train_order, 320))",
                  "labeled_images": sum(rel in label_counts for rel in pilot),
                  "empty_images": sum(rel not in label_counts for rel in pilot),
                  "relative_manifest_sha256": sha256(configs / "pilot320_relative.txt")},
        "configs": {name: {"path": str(configs / name), "sha256": sha256(configs / name)}
                    for name in ("training.yaml", "pilot.yaml", "train.txt", "dev.txt", "pilot320.txt")},
        "yaml_platform": args.yaml_platform,
        "annotated_index_checks": {"exact_dev_plus_val_membership": True, "frame_indices_in_p1_range": True,
                                   "timestamps_equal_frame_index_divided_by_p1_fps": True,
                                   "warning": "P1 frame-index/FPS arithmetic is not independent decode/alignment proof."},
        "selected_videos": selected_video_records,
        "pending_checks": ["Original-video timestamp/full-decode audit", "Annotated-image alignment to decoded frames",
                           "Fresh SHA256 validation of selected original videos"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_owned(p2 / "reports/data_preflight.json", json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    emit({"stage": "complete", "report": str(p2 / "reports/data_preflight.json"),
          "decoded": decoded, "elapsed_seconds": report["elapsed_seconds"]})


if __name__ == "__main__":
    main()
