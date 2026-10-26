"""P3 ↔ P2 pilot-cache integration. Run from the repo root.

This is an integration smoke test, NOT a P5 scoring/export pipeline.
Uses every decoded frame and original PTS-based timestamps.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from temporal.tca import TCA


def load_reader_class(p2_delivery):
    reader_root = Path(p2_delivery).expanduser().resolve() / 'reader'
    if not (reader_root / 'p2' / 'cache_io.py').is_file():
        raise FileNotFoundError(f'Missing P2 reader: {reader_root / "p2/cache_io.py"}')
    sys.path.insert(0, str(reader_root))
    from p2.cache_io import VideoCacheReader
    return VideoCacheReader


def normalize_params(params):
    """Accept P4 names and existing P3 names without changing TCA core."""
    params = dict(params or {})
    aliases = {'iou_thr': 'iou_threshold', 'sigma_px': 'sigma'}
    for old, new in aliases.items():
        if old in params:
            if new in params:
                raise ValueError(f'Both {old} and {new} supplied')
            params[new] = params.pop(old)
    allowed = {'iou_threshold', 'timeout', 'tau', 'present_dur',
               'absence_dur', 'sigma', 'low', 'high', 'margin'}
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f'Unknown TCA parameters: {sorted(unknown)}')
    return params


def run_pilot(video_cache, params=None, conf_threshold=0.1, max_frames=None):
    """Return a diagnostic summary; no GT annotations or final COCO output.

    video_cache is an open P2 VideoCacheReader. State resets on every call.
    """
    if not 0 <= conf_threshold <= 1:
        raise ValueError('conf_threshold must be between 0 and 1')
    tca = TCA(**normalize_params(params))
    total = len(video_cache)
    n = total if max_frames is None else min(total, max_frames)
    counts = []
    candidate_counts = []
    started = time.perf_counter()
    first_timestamp = last_timestamp = None
    for i in range(n):
        frame = video_cache.read(i)
        if frame['frame_idx'] != i:
            raise ValueError(f'Unexpected frame index {frame["frame_idx"]} != {i}')
        timestamp = float(frame['timestamp'])
        if first_timestamp is None:
            first_timestamp = timestamp
        last_timestamp = timestamp
        scores = frame['conf']
        masks = frame['soft_mask']
        if len(scores) != len(masks):
            raise ValueError(f'Confidence/mask count mismatch on frame {i}')
        selected = masks[scores >= conf_threshold]
        # Cast each selected fp16 soft mask to float32 inside existing TCA.update.
        outputs = tca.update(selected, timestamp)
        counts.append(len(outputs))
        candidate_counts.append(len(selected))
    elapsed = time.perf_counter() - started
    return {
        'video_id': video_cache.metadata['source_identity']['video_id'],
        'frames_processed': n,
        'frames_available': total,
        'first_timestamp': first_timestamp,
        'last_timestamp': last_timestamp,
        'selected_candidates': sum(candidate_counts),
        'frames_with_candidates': sum(x > 0 for x in candidate_counts),
        'frames_with_tca_output': sum(x > 0 for x in counts),
        'output_instances_total_over_frames': sum(counts),
        'elapsed_seconds_including_sqlite_read': round(elapsed, 4),
        'ms_per_frame_including_sqlite_read': round(1000 * elapsed / n, 3) if n else None,
        'note': 'Pilot integration only; not model FPS or P5 metrics',
    }


def main():
    ap = argparse.ArgumentParser(description='P3/P2 SQLite pilot integration smoke test')
    ap.add_argument('--p2-dir', required=True, help='Extracted P2 delivery folder')
    ap.add_argument('--video', default='RipVIS-071', help='Video ID, without .sqlite')
    ap.add_argument('--max-frames', type=int, default=30, help='0 means ALL frames')
    ap.add_argument('--conf', type=float, default=0.1, help='Temporary prefilter, NOT final T0')
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--present-dur', type=float, default=0.2)
    ap.add_argument('--absence-dur', type=float, default=0.5)
    ap.add_argument('--timeout', type=float, default=2.0)
    ap.add_argument('--high', type=float, default=0.4)
    ap.add_argument('--low', type=float, default=0.2)
    ap.add_argument('--iou-thr', type=float, default=0.3)
    ap.add_argument('--sigma-px', type=float, default=2.0)
    args = ap.parse_args()
    folder = Path(args.p2_dir).expanduser().resolve()
    db_path = folder / f'{args.video}.sqlite'
    if not db_path.is_file():
        ap.error(f'SQLite cache not found: {db_path}')
    Reader = load_reader_class(folder)
    reader = Reader(db_path)
    try:
        if not reader.complete:
            raise ValueError('P2 cache is not marked complete')
        params = dict(tau=args.tau, present_dur=args.present_dur,
                      absence_dur=args.absence_dur, timeout=args.timeout,
                      high=args.high, low=args.low, iou_thr=args.iou_thr,
                      sigma_px=args.sigma_px)
        result = run_pilot(reader, params, args.conf,
                           None if args.max_frames == 0 else args.max_frames)
        print(json.dumps(result, indent=2))
    finally:
        reader.close()


if __name__ == '__main__':
    main()
