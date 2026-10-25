"""Read-only CPU diagnosis of a temporary checkpoint's raw dev confidences."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest", default="upstream/data/manifests/frames_index.csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = YOLO(str(args.checkpoint)).model.cpu().eval()
    nc = int(model.model[-1].nc)
    with args.manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "dev" and int(row["n_gt"]) > 0]
    selected, seen = [], set()
    for row in rows:
        if row["video_id"] not in seen:
            selected.append(row)
            seen.add(row["video_id"])
        if len(selected) == 5:
            break
    results = []
    for row in selected:
        image = cv2.imread(str(args.dataset_root / row["image_rel"]))
        transformed = LetterBox(new_shape=(640, 640), auto=False)(image=image)
        tensor = torch.from_numpy(np.ascontiguousarray(transformed[:, :, ::-1].transpose(2, 0, 1))).float()[None] / 255
        with torch.inference_mode():
            output = model(tensor)
        prediction = output[0] if isinstance(output, tuple) else output
        scores = prediction[0, 4:4+nc].amax(0)
        results.append({"image_rel": row["image_rel"], "n_gt_polygons": int(row["n_gt"]),
            "raw_head_shape": list(prediction.shape), "max_raw_class_confidence": scores.max().item(),
            "raw_candidates_ge_0_1": int((scores >= np.float32(.1)).sum().item()),
            "raw_candidates_ge_0_01": int((scores >= np.float32(.01)).sum().item()),
            "raw_candidates_ge_0_001": int((scores >= np.float32(.001)).sum().item()),
            "top5_confidences": scores.topk(5).values.tolist()})
    report = {"checkpoint": str(args.checkpoint), "device": "cpu", "class_count": nc,
              "names": model.names, "preprocessing": "imgsz640 square letterbox, RGB float32 /255",
              "samples": results, "interpretation": "Diagnostic only. No confidence threshold, weight, split or acceptance gate changed."}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
