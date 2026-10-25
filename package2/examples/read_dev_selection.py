"""Read local P2 selection inputs without implementing P4/P5 output rules."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from collect_dev_predictions import SelectionCandidateReader

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("sqlite", type=Path)
parser.add_argument("--threshold", type=float, default=.5)
args = parser.parse_args()
reader = SelectionCandidateReader(args.sqlite)
try:
    print("kind:", reader.metadata["kind"], "complete:", reader.complete)
    print("checkpoint:", reader.metadata["checkpoint"]["sha256"])
    for index, record in enumerate(reader.iter_frames(args.threshold)):
        frame = record["frame"]
        print(frame["video_id"], frame["frame_idx"], "PTS:", frame["timestamp"],
              "P1:", frame["p1_t_sec"], "retained:", len(record["candidates"]))
        # candidate['rle'] is already a native original-size COCO RLE;
        # candidate['conf'] is float32 and candidate_index preserves NMS order.
        # Empty candidate lists remain explicit rows. No area or metric rules.
        if index >= 4:
            break
finally:
    reader.close()
