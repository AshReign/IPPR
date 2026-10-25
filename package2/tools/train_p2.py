"""Native Ultralytics training with measured, auditable P2 pilot/formal runs."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]

def dump(path, payload):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['pilot', 'formal'], required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    import torch
    from ultralytics import YOLO
    run = ROOT / 'runs' / a.name
    if run.exists():
        raise SystemExit(f'Refusing to overwrite existing run: {run}')
    run.mkdir(parents=True)
    dataset = ROOT / 'configs' / ('pilot.yaml' if a.stage == 'pilot' else 'training.yaml')
    data_report = json.loads((ROOT / 'reports/data_preflight.json').read_text())
    if data_report.get('status') != 'image_and_label_preflight_passed':
        raise RuntimeError('Data preflight has not passed')
    if data_report['split_counts']['dev']['images'] != 2421 or not data_report['train_dev_val_video_disjoint']:
        raise RuntimeError('Frozen data protocol mismatch')
    if a.stage == 'pilot' and not 200 <= data_report['pilot']['size'] <= 500:
        raise RuntimeError('Pilot subset outside required 200-500 range')
    weight = ROOT.parent / 'weights/yolo11n-seg.pt'
    options = dict(data=str(dataset), imgsz=640, batch=8, epochs=2 if a.stage == 'pilot' else 30,
                   patience=5, amp=True, seed=42, deterministic=True, save_period=1,
                   workers=a.workers, device=0, cache=False, project=str(ROOT/'runs'),
                   name=a.name, exist_ok=True, plots=True, val=True, pretrained=True,
                   resume=False, verbose=False)
    state = dict(stage=a.stage, status='starting', pid=os.getpid(),
                 started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                 initial_weight=str(weight), initial_weight_sha256=sha(weight),
                 dataset_yaml_sha256=sha(dataset), options=options,
                 data_preflight_sha256=sha(ROOT/'reports/data_preflight.json'),
                 image_list_sha256={name:sha(ROOT/'configs'/name) for name in
                                    ['train.txt','dev.txt','pilot320.txt']},
                 p1_commit=json.loads((ROOT/'upstream_manifest.json').read_text())['commit'],
                 script_sha256=sha(__file__), epochs=[], validations=[], optimizer_steps=0,
                 amp_scale_reductions=0, stage_note='Pilot is temporary and not a frozen final model' if a.stage=='pilot' else 'Final model selection still requires P5 dev F2 evaluator')
    dump(run/'p2_measurements.json',state)
    batch_index = 0
    train_started = None
    val_started = None
    wall_start = time.perf_counter()
    handles = []
    def persist():
        state['elapsed_s'] = time.perf_counter()-wall_start
        dump(run/'p2_measurements.json',state)
    def before_train(trainer):
        if not trainer.amp:
            raise RuntimeError('Native AMP check disabled AMP: refusing a silent protocol change')
        if trainer.batch_size != 8 or trainer.args.imgsz != 640:
            raise RuntimeError('Native trainer changed required batch or image size')
        state['actual_amp']=bool(trainer.amp)
        state['actual_train_images']=len(trainer.train_loader.dataset)
        state['actual_dev_images']=len(trainer.test_loader.dataset)
        expected_train = data_report['pilot']['size'] if a.stage == 'pilot' else data_report['split_counts']['train']['images']
        if state['actual_train_images'] != expected_train or state['actual_dev_images'] != data_report['split_counts']['dev']['images']:
            raise RuntimeError('Actual loader counts differ from the verified frozen image lists')
        state['actual_optimizer']=type(trainer.optimizer).__name__
        state['actual_optimizer_groups']=[{key:value for key,value in group.items()
                                          if key in ('lr','initial_lr','betas','momentum','weight_decay','eps')}
                                         for group in trainer.optimizer.param_groups]
        state['native_nbs']=trainer.args.nbs
        state['initial_accumulate']=trainer.accumulate
        state['actual_lr0']=trainer.args.lr0
        state['actual_momentum']=trainer.args.momentum
        state['native_checkpoint_weights']='EMA; epoch0.pt is the end of completed epoch 1'
        state['actual_validation_batch_size']=trainer.test_loader.batch_size
        state['actual_train_workers']=trainer.train_loader.num_workers
        def post_step(*args, **kwargs):
            state['optimizer_steps'] += 1
        handles.append(trainer.optimizer.register_step_post_hook(post_step))
        persist()
    def epoch_start(trainer):
        nonlocal train_started, batch_index
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        train_started=time.perf_counter()
        batch_index=0
        state.update(status='training', epoch=trainer.epoch+1, batch=0,
                     batches=len(trainer.train_loader))
        persist()
    def batch_start(trainer):
        state['amp_scale_before_batch']=float(trainer.scaler.get_scale())
    def batch_end(trainer):
        nonlocal batch_index
        batch_index+=1
        scale=float(trainer.scaler.get_scale())
        if scale < state['amp_scale_before_batch']:
            state['amp_scale_reductions']+=1
        losses=trainer.loss_items.detach().float().cpu().tolist()
        if not all(math.isfinite(float(v)) for v in losses):
            raise RuntimeError(f'Nonfinite training loss at epoch {trainer.epoch+1}, batch {batch_index}: {losses}')
        state.update(batch=batch_index, last_losses=losses, amp_scale=scale)
        if batch_index==1 or batch_index%5==0 or batch_index==len(trainer.train_loader):
            persist()
    def epoch_end(trainer):
        torch.cuda.synchronize()
        checked_models=[('model',trainer.model)]
        if trainer.ema is not None:
            checked_models.append(('ema',trainer.ema.ema))
        bad=[f'{label}.{name}' for label,module in checked_models for name,value in module.state_dict().items()
             if value.is_floating_point() and not torch.isfinite(value.detach()).all()]
        if bad: raise RuntimeError(f'Nonfinite model/EMA parameters or buffers: {bad[:10]}')
        state['epochs'].append(dict(epoch=trainer.epoch+1, train_seconds=time.perf_counter()-train_started,
                                    train_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                    train_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                    optimizer_steps_cumulative=state['optimizer_steps'],
                                    accumulate=trainer.accumulate,
                                    amp_scale=float(trainer.scaler.get_scale())))
        persist()
    def val_start(validator):
        nonlocal val_started
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        val_started=time.perf_counter()
        state['status']='validating'
        persist()
    def val_end(validator):
        torch.cuda.synchronize()
        state['validations'].append(dict(after_epoch=state.get('epoch'),
                                        validation_seconds=time.perf_counter()-val_started,
                                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                        note='Native validation metrics are diagnostics, not P5 F2 selection'))
        persist()
    model=YOLO(str(weight))
    for event,callback in [('on_train_start',before_train),('on_train_epoch_start',epoch_start),
                           ('on_train_batch_start',batch_start),('on_train_batch_end',batch_end),
                           ('on_train_epoch_end',epoch_end),('on_val_start',val_start),('on_val_end',val_end)]:
        model.add_callback(event,callback)
    try:
        model.train(**options)
        if state['optimizer_steps']==0: raise RuntimeError('No real optimizer updates occurred')
        state.update(status='completed', completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                     checkpoint_files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p))
                                       for p in sorted((run/'weights').glob('*.pt'))])
        persist()
    except BaseException as e:
        state.update(status='failed',error=repr(e),traceback=traceback.format_exc())
        persist()
        raise
    finally:
        for h in handles: h.remove()
    print(json.dumps({k:state[k] for k in ['status','stage','elapsed_s','optimizer_steps','amp_scale_reductions','epochs','validations']},indent=2))

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
