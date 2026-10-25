"""Record the actual reusable environment without installing or altering packages."""
from __future__ import annotations
import datetime as dt
import hashlib
import importlib.metadata as md
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    import psutil
    import torch
    import ultralytics
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    from ultralytics.utils import ops
    packages = {d.metadata['Name']: d.version for d in md.distributions() if d.metadata['Name']}
    gpu = torch.cuda.get_device_properties(0)
    report = {
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python_executable": sys.executable, "python": sys.version,
        "platform": platform.platform(), "runtime": "Existing local WSL2 on the user's Windows RTX 4060 Ti machine",
        "torch": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "ultralytics": ultralytics.__version__,
        "gpu": {"name": gpu.name, "vram_bytes": gpu.total_memory, "compute_capability": f"{gpu.major}.{gpu.minor}"},
        "wsl_memory": psutil.virtual_memory()._asdict(),
        "workspace_disk": psutil.disk_usage(str(ROOT))._asdict(),
        "packages": packages,
        "p1_source_commit": json.loads((ROOT / 'upstream_manifest.json').read_text())['commit'],
        "pretrained_sha256": hashlib.sha256((ROOT.parent / 'weights/yolo11n-seg.pt').read_bytes()).hexdigest(),
        "note": "P1 requirements are preserved. Existing GPU environment is reused with this separate exact P2 lock; source/API and data compatibility are checked rather than claiming P1's newer package versions are installed."
    }
    (REPORTS / 'environment.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], text=True, capture_output=True, check=True)
    (ROOT / 'requirements-p2-existing.lock.txt').write_text(freeze.stdout, encoding='utf-8')
    check = subprocess.run([sys.executable, '-m', 'pip', 'check'], text=True, capture_output=True)
    (REPORTS / 'pip_check.txt').write_text(check.stdout + check.stderr, encoding='utf-8')
    src = REPORTS / 'locked_source'
    src.mkdir(exist_ok=True)
    for name, obj in [('trainer', BaseTrainer), ('segmentation_predictor', SegmentationPredictor), ('ops', ops)]:
        p = Path(inspect.getfile(obj))
        (src / (name + '.py')).write_bytes(p.read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k not in ['packages']}, indent=2))
    print('pip_check:',check.returncode,check.stdout.strip())

if __name__ == '__main__':
    main()
