"""Final scoring (Package 5, step 8): predictions in, results table and bootstrap out.

Usage (from the repo root, with the virtual environment active):
    python -m scripts.final_scoring --split val --pred T0=<t0.json> --pred T1=<t1.json> --pred T2=<t2.json>
    (optional) --fps T0=30.1 --fps T1=29.8 --fps T2=25.0     # from Package 6

Run it on val ONCE, after all three configs are frozen. Use --split dev for dry runs.

Writes to --out (default outputs/final_<split>/):
    results.csv, results.md   one row per condition: P, R, F2, false-alarm rate,
                              fragmentation per minute, median delay (n detected / n tracks), FPS
    bootstrap.csv             paired bootstrap of F2: T1-T0, T2-T0, T2-T1
    per_video.csv             TP/FP/FN/F2 per video and condition (Package 6 picks figures from it)
    per_frame_<cond>.csv      per-frame counts
    run_info.json             git commit, prediction file hashes, seed
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from eval.bootstrap import paired_bootstrap
from eval.evaluator import MANIFESTS, evaluate, load_gt, scores, summarise, temporal_frames, video_counts
from eval.temporal_metrics import summarise as temporal_summary, track_stats

REPO = Path(__file__).resolve().parents[1]
SEED = 42


def single_rip_videos(split: str):
    """Videos used for fragmentation and delay. Val: the frozen P1 list. Dev (dry runs only):
    rip videos that never have more than one mask in a frame."""
    if split == "val":
        return (MANIFESTS / "single_rip_val.txt").read_text().split()
    v = pd.read_csv(MANIFESTS / "videos.csv")
    return sorted(v[(v.split == "dev") & v.has_rip & (v.max_rips_per_frame == 1)].video_id)


def key_values(pairs):
    out = {}
    for item in pairs or []:
        k, _, v = item.partition("=")
        out[k] = v
    return out


def markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "val"], required=True)
    ap.add_argument("--pred", action="append", required=True, help="CONDITION=path/to/predictions.json")
    ap.add_argument("--fps", action="append", help="CONDITION=frames per second (Package 6)")
    ap.add_argument("--out", help="output folder (default outputs/final_<split>)")
    ap.add_argument("--resamples", type=int, default=1000)
    args = ap.parse_args()

    preds, fps = key_values(args.pred), key_values(args.fps)
    out = Path(args.out or REPO / "outputs" / f"final_{args.split}")
    out.mkdir(parents=True, exist_ok=True)
    gt = load_gt(args.split)
    single = single_rip_videos(args.split)

    rows, per_video, counts = [], [], {}
    for cond, path in preds.items():
        per_frame = evaluate(json.loads(Path(path).read_text()), gt)
        per_frame.to_csv(out / f"per_frame_{cond}.csv", index=False)
        s = summarise(per_frame)
        t = temporal_summary([track_stats(v, temporal_frames(per_frame, v)) for v in single])
        rows.append({"condition": cond, "precision": s["precision"], "recall": s["recall"],
                     "f2": s["f2"], "false_alarm_rate": s["false_alarm_rate"],
                     "frag_per_min": t["frag_per_min"], "median_delay_s": t["median_delay_s"],
                     "delay_n": f"{t['n_detected']}/{t['n_tracks']}",
                     "fps": float(fps[cond]) if cond in fps else None,
                     "tp": s["tp"], "fp": s["fp"], "fn": s["fn"]})
        counts[cond] = video_counts(per_frame)
        v = counts[cond].assign(condition=cond)
        per_video.append(v.assign(f2=scores(v.tp, v.fp, v.fn)[3]))

    results = pd.DataFrame(rows)
    results.to_csv(out / "results.csv", index=False)
    (out / "results.md").write_text(markdown(results.round(3)))
    pd.concat(per_video).to_csv(out / "per_video.csv", index=False)
    pairs = [p for p in [("T1", "T0"), ("T2", "T0"), ("T2", "T1")] if set(p) <= set(preds)]
    boot = paired_bootstrap(counts, pairs, args.resamples, SEED) if pairs else pd.DataFrame()
    boot.to_csv(out / "bootstrap.csv", index=False)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    info = {"split": args.split, "commit": commit, "seed": SEED, "resamples": args.resamples,
            "single_rip_videos": single,
            "predictions": {c: {"path": str(p), "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
                            for c, p in preds.items()}}
    (out / "run_info.json").write_text(json.dumps(info, indent=2))

    print(markdown(results.round(3)))
    if len(boot):
        print(markdown(boot[["comparison", "delta_f2", "ci_low", "ci_high", "share_positive"]].round(3)))
    print("written to", out)


if __name__ == "__main__":
    main()
