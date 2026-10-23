"""Package 4, steps 6-7: apply the selection rule to the dev search CSV and freeze.

Writes one YAML per condition (t0.yaml, t1.yaml, t2.yaml) stamped with the git
commit, the SHA-256 of the search CSV and the time of freezing. Refuses to
overwrite existing frozen configs ("never re-tune after this point") unless
--force is given, so an accidental re-run cannot silently change the result.

Example:
    python scripts/select_configs.py --search-csv runs/dev_search/dev_search.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.tune_dev import git_commit  # noqa: E402
from temporal.selection import select  # noqa: E402

METRIC_KEYS = ["precision", "recall", "f2", "false_alarm_rate", "tp", "fp", "fn"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(search_csv: Path, out_dir: Path, force: bool = False) -> dict[str, dict]:
    chosen = select(pd.read_csv(search_csv))
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [out_dir / f"{c.lower()}.yaml" for c in chosen
                if (out_dir / f"{c.lower()}.yaml").exists()]
    if existing and not force:
        raise FileExistsError(f"frozen configs already exist: {existing}; "
                              "re-tuning after freezing is not allowed (use --force "
                              "only to repair a broken freeze, and log why)")

    stamp = {"git_commit": git_commit(), "source_csv": str(search_csv),
             "source_csv_sha256": sha256(search_csv),
             "frozen_at": dt.datetime.now().isoformat(timespec="seconds")}
    for cond, row in chosen.items():
        doc = {"condition": cond, "config_id": row["config_id"],
               "params": json.loads(row["params"]),
               "dev_metrics": {k: float(row[k]) for k in METRIC_KEYS},
               "dev_t0_recall_r0": row["r0"], "selection_flag": row["flag"],
               **stamp}
        with open(out_dir / f"{cond.lower()}.yaml", "w") as fh:
            yaml.safe_dump(doc, fh, sort_keys=False)
    return chosen


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search-csv", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "configs/frozen"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    chosen = freeze(Path(args.search_csv), Path(args.out_dir), args.force)
    for cond, row in chosen.items():
        print(f"{cond}: {row['config_id']}  F2={row['f2']:.4f}  R={row['recall']:.4f}"
              + (f"  [{row['flag']}]" if row["flag"] else ""))


if __name__ == "__main__":
    main()
