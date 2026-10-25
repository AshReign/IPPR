"""Compare baseline and optimized TCA on identical synthetic frames."""
import argparse
import time
import numpy as np
from temporal.tca import TCA as Optimized
from temporal.tca_before import TCA as Baseline


def make_frames(count, height, width, fps):
    for i in range(count):
        a = np.zeros((height, width), dtype=np.float32)
        x = 20 + (i // 10) % max(1, width - 60)
        y = height // 3
        a[y:y+24, x:x+24] = 0.95
        yield i / fps, [a]


def run(cls, frames):
    tracker = cls()
    outputs = []
    start = time.perf_counter()
    for timestamp, masks in frames:
        result = tracker.update(masks, timestamp)
        outputs.append([(track_id, mask.copy()) for track_id, mask in result])
    return time.perf_counter() - start, outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=1800)
    parser.add_argument('--height', type=int, default=360)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--fps', type=float, default=30)
    args = parser.parse_args()
    if args.frames < 1 or args.fps <= 0 or args.height < 50 or args.width < 80:
        parser.error('Invalid frame count, fps, or dimensions')
    data = list(make_frames(args.frames, args.height, args.width, args.fps))
    baseline_time, baseline = run(Baseline, data)
    optimized_time, optimized = run(Optimized, data)
    equal = len(baseline) == len(optimized) and all(
        len(a) == len(b) and all(
            id_a == id_b and np.array_equal(mask_a, mask_b)
            for (id_a, mask_a), (id_b, mask_b) in zip(a, b)
        ) for a, b in zip(baseline, optimized)
    )
    print(f'Frames: {args.frames} ({args.width}x{args.height}, {args.fps:g} FPS)')
    print(f'Before: {baseline_time:.3f} s, {baseline_time * 1000 / args.frames:.3f} ms/frame')
    print(f'After:  {optimized_time:.3f} s, {optimized_time * 1000 / args.frames:.3f} ms/frame')
    print(f'Speedup: {baseline_time / optimized_time:.2f}x')
    print(f'Outputs identical: {equal}')
    if not equal:
        raise SystemExit('ERROR: output changed after optimization')


if __name__ == '__main__':
    main()
