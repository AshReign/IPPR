"""Inspect one complete P2 frame without model/P5/torch imports or filtering."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--frame-idx', type=int, default=0)
    args = parser.parse_args()
    codec = Path(__file__).resolve().parent / 'p2/cache_io.py'
    if hashlib.sha256(codec.read_bytes()).hexdigest() != '77b14aa00fb5ebaee22448fb0a9bf040dfc90db2ebc20f889904019cdb7d06ed':
        raise ValueError('Reader codec differs from the frozen schema')
    spec = importlib.util.spec_from_file_location('p2_raw_reader', codec)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reader = module.VideoCacheReader(args.cache)
    try:
        if not reader.complete:
            raise ValueError('Cache is still incomplete')
        frame = reader.read(args.frame_idx)
        print(json.dumps({'video_id': reader.metadata['source_identity']['video_id'],
                          'frames_in_video': len(reader), 'frame_idx': frame['frame_idx'],
                          'timestamp': frame['timestamp'], 'pts': frame['pts'],
                          'raw_candidates': len(frame['conf']), 'boxes': frame['boxes'].tolist(),
                          'conf': frame['conf'].tolist(), 'soft_mask_shape': frame['soft_mask'].shape,
                          'soft_mask_dtype': str(frame['soft_mask'].dtype), 'original_hw': reader.metadata['original_shape_hw'],
                          'transform': reader.metadata['transform'], 'no_output_filter_applied': True}, indent=2))
    finally:
        reader.close()


if __name__ == '__main__':
    main()
