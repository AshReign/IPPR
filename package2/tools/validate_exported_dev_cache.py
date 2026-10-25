"""Compare actual exported dev-video frames with same-frame native inference.

This complements the annotated-JPEG codec validation. It reads existing full
video SQLite files, decodes their original source videos, and checks 50 seeded
annotated frame indices. Final-val pixels are never read by this tool.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import ExitStack
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from p2.cache_io import VideoCacheReader
from p2.soft_masks import PREDICT_DEFAULTS, SoftMaskPredictor, restore_work_masks
import p2.soft_masks as soft_masks
from export_cache import canonical, json_write, runtime_after_prediction, sha256, validate_production_runtime

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binary_iou(left, right):
    require(left.shape == right.shape, 'Mask dimensions differ')
    union = int(np.count_nonzero(left | right))
    return float(np.count_nonzero(left & right) / union) if union else 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='0')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--diagnostic-pilot', action='store_true')
    args = parser.parse_args()
    args.cache_dir = (ROOT / args.cache_dir).resolve()
    args.checkpoint = (ROOT / args.checkpoint).resolve()
    args.output = (ROOT / args.output).resolve()
    args.cache_dir.relative_to(ROOT)
    args.output.relative_to(ROOT)
    require(not args.output.exists(), 'Use a new validation output directory')
    require(args.frames >= 50, 'At least 50 annotated frames are required')
    manifests = ROOT / 'upstream/data/manifests'
    dev_ids = sorted((manifests / 'dev_videos.txt').read_text().split())
    baseline = json.loads((ROOT / 'upstream_manifest.json').read_text())
    for name in ['frames_index.csv', 'dev_videos.txt']:
        require(sha256(manifests / name) == baseline['files']['data/manifests/' + name], 'Frozen manifest changed')
    ids = [video for video in dev_ids if (args.cache_dir / f'{video}.sqlite').is_file()]
    require(len(ids) == 17 or args.diagnostic_pilot, 'Full frozen dev scope must be cached first')
    require(bool(ids), 'No dev video caches found')
    with (manifests / 'frames_index.csv').open(newline='', encoding='utf-8-sig') as file:
        rows = [row for row in csv.DictReader(file) if row['split'] == 'dev' and row['video_id'] in ids]
    require(len(rows) >= args.frames, 'Insufficient annotated frames in cached dev videos')
    samples = random.Random(args.seed).sample(rows, args.frames)
    grouped = defaultdict(dict)
    for row in samples:
        grouped[row['video_id']][int(row['frame_idx'])] = row
    checkpoint_hash = sha256(args.checkpoint)
    module_hash = sha256(soft_masks.__file__)
    args.output.mkdir(parents=True)
    (args.output / 'sample.json').write_text(json.dumps(samples, indent=2), encoding='utf-8')
    import av
    import torch
    if args.device == 'cpu':
        require(1 <= args.cpu_threads <= 4, 'CPU diagnostic thread budget must be 1..4')
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(1)
        import cv2
        cv2.setNumThreads(1)
    from ultralytics import YOLO
    started = time.perf_counter()
    comparisons = []
    cache_files = []
    actual_runtime = None
    total_frames = 0
    empty_frames = 0
    run_identity = None
    try:
        with ExitStack() as stack:
            readers = {}
            source_paths = {}
            for video in ids:
                path = args.cache_dir / f'{video}.sqlite'
                reader = VideoCacheReader(path)
                stack.callback(reader.close)
                readers[video] = reader
                require(reader.complete, f'Incomplete cache: {video}')
                identity = reader.metadata['run_identity']
                if run_identity is None:
                    run_identity = identity
                require(identity == run_identity, 'Mixed model/configuration caches')
                require(identity['checkpoint_sha256'] == checkpoint_hash, 'Checkpoint differs from exported cache')
                require(identity['prediction_config']['soft_mask_module_sha256'] == module_hash, 'Mask code changed since export')
                require(identity['prediction_config']['inference'] == PREDICT_DEFAULTS, 'Inference configuration differs')
                require(reader.metadata['source_identity']['video_id'] == video, 'Source video identity mismatch')
                require(reader.metadata['source_identity']['split'] == 'dev', 'Only dev caches may be validated here')
                require(len(reader) == reader.metadata['source_identity']['expected_decoded_frames'], 'Cache frame count mismatch')
                source_paths[video] = ROOT.parent / reader.metadata['source_identity']['relative_path']
                require(sha256(source_paths[video]) == reader.metadata['source_identity']['sha256'],
                        f'Original video bytes changed: {video}')
                if not args.diagnostic_pilot:
                    require(identity['kind'] == 'final' and identity['gates']['frozen'], 'Need frozen final dev caches')
                cache_files.append({'video_id': video, 'path': str(path), 'sha256': sha256(path), 'frames': len(reader),
                                    'source_sha256': reader.metadata['source_identity']['sha256']})
            model = YOLO(str(args.checkpoint))
            runtime_device = torch.device('cpu') if args.device == 'cpu' else args.device
            for video in sorted(grouped):
                reader = readers[video]
                wanted = grouped[video]
                source = source_paths[video]
                seen = set()
                with av.open(str(source)) as container:
                    require(len(container.streams.video) == 1, 'Expected one video stream')
                    stream = container.streams.video[0]
                    stream.codec_context.thread_count = 1
                    first_pts = None
                    for index, decoded in enumerate(container.decode(stream)):
                        if first_pts is None:
                            first_pts = decoded.pts
                        if index not in wanted:
                            continue
                        cached = reader.read(index)
                        require(cached['pts'] == decoded.pts, 'Cached frame PTS does not match the same decoded frame')
                        timestamp = float((decoded.pts - first_pts) * Fraction(decoded.time_base))
                        require(abs(cached['timestamp'] - timestamp) <= 1e-9, 'Cache timeline differs')
                        image = decoded.to_ndarray(format='bgr24')
                        require(list(image.shape[:2]) == reader.metadata['original_shape_hw'], 'Frame dimensions differ')
                        result = model.predict(image, predictor=SoftMaskPredictor, device=runtime_device, **PREDICT_DEFAULTS)[0]
                        if args.device == 'cpu':
                            require(torch.get_num_threads() == args.cpu_threads and torch.get_num_interop_threads() == 1,
                                    'CPU inference thread budget changed')
                        runtime = runtime_after_prediction(result)
                        if actual_runtime is None:
                            actual_runtime = runtime
                        require(runtime == actual_runtime, 'Validation backend changed during the run')
                        if not args.diagnostic_pilot:
                            validate_production_runtime(runtime, run_identity['gates']['validation_runtime'])
                        raw = result.p2_cache_candidates
                        boxes = raw.boxes.xyxy.detach().float().cpu().numpy()
                        confidence = raw.boxes.conf.detach().float().cpu().numpy()
                        require(np.array_equal(boxes, cached['boxes']) and np.array_equal(confidence, cached['conf']),
                                'Native candidate identity/order differs from the actual cache')
                        require(raw.p2_transform == reader.metadata['transform'], 'Transform differs from exported cache')
                        restored_tensor, reconstruction = restore_work_masks(
                            cached['soft_mask'], image.shape[:2], cached['boxes'],
                            transform=reader.metadata['transform'], return_diagnostics=True)
                        restored = restored_tensor.numpy()
                        native = result.soft_masks.detach().float().cpu().numpy()
                        standard = (result.masks.data.detach().cpu().numpy().astype(bool)
                                    if result.masks is not None else np.empty((0, *image.shape[:2]), dtype=bool))
                        require(np.array_equal(native > .5, standard),
                                'Native soft masks differ from independent library binary output')
                        original_index = result.candidate_index.detach().cpu().numpy().tolist()
                        native_map = {candidate: offset for offset, candidate in enumerate(original_index)}
                        for candidate in range(len(confidence)):
                            reference = standard[native_map[candidate]] if candidate in native_map else np.zeros(image.shape[:2], bool)
                            predicted = restored[candidate] > .5
                            score = binary_iou(predicted, reference)
                            item = {'video_id': video, 'frame_idx': index, 'candidate_index': candidate,
                                    'timestamp': timestamp, 'p1_t_sec': float(wanted[index]['t_sec']),
                                    'conf': float(confidence[candidate]), 'native_area': int(reference.sum()),
                                    'restored_area': int(predicted.sum()), 'iou': score, 'pass': score >= .98,
                                    'reconstruction_status': reconstruction[candidate]['status'],
                                    'reconstruction_reason': reconstruction[candidate].get('reason', '')}
                            comparisons.append(item)
                            if not item['pass']:
                                np.savez_compressed(args.output / f'fail_{video}_{index}_{candidate}.npz',
                                                    reference=reference, restored=predicted, box=boxes[candidate])
                        total_frames += 1
                        empty_frames += len(confidence) == 0
                        seen.add(index)
                        json_write(args.output / 'progress.json', {'status': 'running', 'annotated_dev_frames': total_frames,
                                   'target_frames': args.frames, 'instance_count': len(comparisons),
                                   'failed_instances': sum(not item['pass'] for item in comparisons)})
                        del result, raw, native, standard, restored, restored_tensor, image
                        if len(seen) == len(wanted):
                            break
                require(seen == set(wanted), f'Could not decode every sampled frame from {video}')
            for item in cache_files:
                require(sha256(item['path']) == item['sha256'], 'Cache changed during validation')
        require(sha256(args.checkpoint) == checkpoint_hash, 'Checkpoint changed during validation')
        failures = sum(not item['pass'] for item in comparisons)
        native_nonempty = sum(item['native_area'] > 0 for item in comparisons)
        passed = total_frames >= 50 and native_nonempty > 0 and failures == 0
        complete_scope = len(ids) == 17 and not args.diagnostic_pilot
        summary = {'schema': 'p2-exported-dev-cache-validation-v1', 'status': 'PASS' if passed else 'FAIL',
                   'production_validation_eligible': passed and complete_scope and actual_runtime['device_type'] == 'cuda',
                   'complete_dev_scope': complete_scope, 'annotated_dev_frames': total_frames,
                   'empty_prediction_frames': empty_frames, 'instance_count': len(comparisons),
                   'native_nonempty_instance_count': native_nonempty,
                   'reconstruction_fallback_instances': sum(item['reconstruction_status'] == 'fallback' for item in comparisons),
                   'failed_instances': failures, 'all_instances_pass': passed,
                   'min_instance_iou': min((item['iou'] for item in comparisons), default=None),
                   'checkpoint_sha256': checkpoint_hash, 'soft_mask_module_sha256': module_hash,
                   'prediction_config_sha256': run_identity['prediction_config_sha256'],
                   'cache_run_identity_sha256': hashlib.sha256(canonical(run_identity).encode()).hexdigest(),
                   'validation_runtime': actual_runtime, 'cache_files': cache_files,
                   'seed': args.seed, 'sample_method': 'uniform random without replacement from P1 annotated dev rows',
                   'sample_sha256': sha256(args.output / 'sample.json'), 'validator_sha256': sha256(__file__),
                   'reference': 'same decoded source-video pixels, identical checkpoint and 640 inference; independent library original binary output, also checked equal to uncompressed native probabilities >0.5',
                   'elapsed_seconds': time.perf_counter() - started}
        if comparisons:
            with (args.output / 'instances.csv').open('w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=list(comparisons[0]))
                writer.writeheader()
                writer.writerows(comparisons)
        json_write(args.output / 'summary.json', summary)
        json_write(args.output / 'progress.json', {key: summary[key] for key in
                   ['status', 'annotated_dev_frames', 'instance_count', 'failed_instances', 'production_validation_eligible']})
        print(json.dumps(summary, indent=2))
        if not passed:
            raise SystemExit(2)
    except Exception as error:
        json_write(args.output / 'error.json', {'status': 'ERROR', 'error': repr(error),
                   'production_validation_eligible': False, 'annotated_dev_frames': total_frames})
        json_write(args.output / 'progress.json', {'status': 'ERROR', 'error': repr(error),
                   'production_validation_eligible': False, 'annotated_dev_frames': total_frames})
        raise


if __name__ == '__main__':
    main()
