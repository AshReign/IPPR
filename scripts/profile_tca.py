"""Synthetic long-video profiler; replace frame generator with real cache later."""
import argparse
import cProfile
import io
import pstats
import time
import numpy as np
from temporal.tca import TCA


def frames(count, height, width, fps):
    for i in range(count):
        mask = np.zeros((height, width), np.float32)
        x = 20 + (i // 10) % max(1, width - 60)
        y = height // 3
        mask[y:y+24, x:x+24] = 0.95
        yield i / fps, [mask]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=1800)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--fps", type=float, default=30)
    args = parser.parse_args()
    tca = TCA()
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    for t, detections in frames(args.frames, args.height, args.width, args.fps):
        tca.update(detections, t)
    profiler.disable()
    elapsed = time.perf_counter() - start
    print(f"Frames: {args.frames}")
    print(f"Total: {elapsed:.3f} s")
    print(f"Average: {elapsed * 1000 / args.frames:.3f} ms/frame")
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(15)
    print(stream.getvalue())


if __name__ == "__main__":
    main()
