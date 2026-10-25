"""Select from a complete versioned P5 dev score matrix; does not run P5."""
from __future__ import annotations
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from p2.selection import select

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', required=True)
    parser.add_argument('--checkpoints', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--protocol', default='configs/selection_protocol.json',
                        help='Versioned selection protocol; use configs/selection_p4_protocol.json for confirmed P4 output')
    args = parser.parse_args()
    protocol = (ROOT / args.protocol).resolve()
    protocol.relative_to(ROOT)
    scores_path = (ROOT / args.scores).resolve()
    inventory_path = (ROOT / args.checkpoints).resolve()
    output = (ROOT / args.output).resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise FileExistsError(output)
    scores = json.loads(scores_path.read_text(), parse_float=Decimal)
    inventory = json.loads(inventory_path.read_text())
    for entry in inventory.get('checkpoints', []):
        path = (ROOT / entry['path']).resolve()
        path.relative_to(ROOT)
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'Checkpoint no longer matches inventory: {path}')
    result = select(scores, inventory, hashlib.sha256(protocol.read_bytes()).hexdigest())
    result.update(score_matrix_sha256=hashlib.sha256(scores_path.read_bytes()).hexdigest(),
                  checkpoint_inventory_sha256=hashlib.sha256(inventory_path.read_bytes()).hexdigest())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False, default=str)
        handle.write('\n')
    print(json.dumps(result, allow_nan=False, default=str))


if __name__ == '__main__':
    main()
