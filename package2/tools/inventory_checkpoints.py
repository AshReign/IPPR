"""Inventory every saved EMA checkpoint after a completed P2 formal run."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run = (ROOT / args.run).resolve()
    output = (ROOT / args.output).resolve()
    run.relative_to(ROOT)
    output.relative_to(ROOT)
    if output.exists():
        raise FileExistsError(output)
    state_path = run / 'p2_measurements.json'
    state = json.loads(state_path.read_text())
    if state['stage'] != 'formal' or state['status'] != 'completed':
        raise ValueError('Wait for a successfully completed formal run')
    epochs = [entry['epoch'] for entry in state['epochs']]
    if epochs != list(range(1, len(epochs) + 1)) or not 1 <= len(epochs) <= 30:
        raise ValueError('Formal run epoch history is inconsistent')
    expected_names = {f'epoch{epoch - 1}.pt' for epoch in epochs}
    if {p.name for p in (run / 'weights').glob('epoch*.pt')} != expected_names:
        raise ValueError('Missing or unexpected per-epoch checkpoints')
    recorded_hashes = {Path(entry['path']).name: entry['sha256'] for entry in state['checkpoint_files']}
    import torch
    entries = []
    for epoch in epochs:
        path = run / 'weights' / f'epoch{epoch - 1}.pt'
        digest = sha(path)
        if digest != recorded_hashes.get(path.name):
            raise ValueError(f'Checkpoint changed after training: {path}')
        # Only trusted locally generated checkpoints under this run directory.
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint.get('epoch') != epoch - 1 or checkpoint.get('ema') is None:
            raise ValueError(f'Wrong epoch or absent EMA weights: {path}')
        if not all(torch.isfinite(value).all().item() for value in checkpoint['ema'].state_dict().values()
                   if value.is_floating_point()):
            raise ValueError(f'Nonfinite saved EMA parameters/buffers: {path}')
        entries.append({'path': str(path.relative_to(ROOT)), 'sha256': digest,
                        'bytes': path.stat().st_size, 'completed_epoch': epoch,
                        'stored_epoch_zero_based': checkpoint['epoch'], 'weights': 'EMA'})
        del checkpoint
    payload = {'schema': 'p2-checkpoint-inventory-v1', 'status': 'completed_formal_run',
               'p1_commit': state['p1_commit'], 'actual_epochs': len(epochs),
               'measurement_sha256': sha(state_path), 'run': str(run.relative_to(ROOT)),
               'dev_manifest_sha256': sha(ROOT / 'upstream/data/manifests/yolo_dev_images.txt'),
               'local_dev_image_list_sha256': state['image_list_sha256']['dev.txt'],
               'checkpoints': entries, 'native_best_is_not_final_selection': True}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'output': str(output), 'saved_epochs': len(entries),
                      'pending_P5_combinations': len(entries) * 17}))


if __name__ == '__main__':
    main()
