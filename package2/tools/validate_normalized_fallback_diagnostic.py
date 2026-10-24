"""CPU-only diagnostic of a support-normalized saturation fallback.

Reads every original GPU cache sample and replays the exact fixed 50 images on
CPU. GPU/CPU candidate differences are recorded, so this is not a GPU gate.
Production decoder and delivered v2 bundle remain unchanged.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils import ops

from p2.cache_io import VideoCacheReader
from p2.soft_masks import PREDICT_DEFAULTS, SoftMaskPredictor, restore_work_masks


def normalized_fallback(work, box, shape):
    work = torch.as_tensor(work).float()
    box = torch.as_tensor(box).float()
    oh, ow = shape
    wh, ww = work.shape
    scaled = box*torch.tensor([ww/ow,wh/oh]*2)
    support = ops.crop_mask(torch.ones((1,wh,ww)),scaled[None])
    support_up = F.interpolate(support[None], shape, mode="bilinear", align_corners=False)[0]
    values_up = F.interpolate(work[None,None], shape, mode="bilinear", align_corners=False)[0]
    normalized = torch.where(support_up>0, values_up/support_up.clamp_min(1e-12), 0)
    return ops.crop_mask(normalized,box[None])[0]


def binary_iou(a,b):
    union = int((a|b).sum())
    return int((a&b).sum())/union if union else 1.


def synthetic_checks():
    shape = (80,128)
    box = torch.tensor([11.3,9.4,112.8,71.2])
    scaled = box*torch.tensor([32/128,20/80]*2)
    support = ops.crop_mask(torch.ones((1,20,32)),scaled[None])[0]
    original_support = ops.crop_mask(torch.ones((1,*shape)),box[None])[0].bool()
    records=[]
    for value in (0.,.25,.5,1.):
        restored=normalized_fallback(support*value,box,shape)
        assert bool(torch.isfinite(restored).all())
        assert not bool((restored[~original_support]!=0).any())
        if value<=.5:
            assert not bool((restored>.5).any())
        else:
            assert torch.equal(restored>.5,original_support)
        records.append({"value":value,"finite":True,"outside_zero":True,"expected_binary":True})
    tiny_box=torch.tensor([1.,1.,1.1,1.1])
    empty=normalized_fallback(torch.zeros((20,32)),tiny_box,shape)
    assert not bool((empty!=0).any())
    records.append({"case":"no_work_support","all_zero":True})
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation",type=Path,required=True)
    parser.add_argument("--dataset-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    synthetic=synthetic_checks()
    baseline=json.loads((args.validation/"summary.json").read_text())
    sampling=json.loads((args.validation/"sampled_frames.json").read_text())
    selected=[sampling["frames"][i] for i in sampling["cache_indices"]]
    model=YOLO(baseline["checkpoint"])
    rows,frames,mismatches=[],[],[]
    started=time.perf_counter()
    for frame in selected:
        stem=f"{frame['video_id']}_{int(frame['frame_idx']):06d}"
        reader=VideoCacheReader(args.validation/"roundtrip_samples"/(stem+".sqlite"))
        try:
            cached,meta=reader.read(0),reader.metadata
        finally:
            reader.close()
        image=cv2.imread(str(args.dataset_root/frame["image_rel"]))
        result=model.predict(image,predictor=SoftMaskPredictor,device=torch.device("cpu"),**PREDICT_DEFAULTS)[0]
        raw=result.p2_cache_candidates
        frame_record={"video_id":frame["video_id"],"frame_idx":int(frame["frame_idx"]),
            "gpu_candidates":len(cached["conf"]),"cpu_candidates":len(raw.boxes)}
        if len(raw.boxes)!=len(cached["conf"]):
            frame_record["candidate_count_match"]=False
            mismatches.append(frame_record)
            frames.append(frame_record)
            continue
        box_error=float(np.max(np.abs(raw.boxes.xyxy.numpy()-cached["boxes"]))) if len(raw.boxes) else 0.
        conf_error=float(np.max(np.abs(raw.boxes.conf.numpy()-cached["conf"]))) if len(raw.boxes) else 0.
        frame_record.update(candidate_count_match=True,max_gpu_cpu_box_abs_error=box_error,
            max_gpu_cpu_conf_abs_error=conf_error,
            exact_boxes=bool(np.array_equal(raw.boxes.xyxy.numpy(),cached["boxes"])),
            exact_conf=bool(np.array_equal(raw.boxes.conf.numpy(),cached["conf"])))
        # This is a diagnostic correspondence check, not a substitute for a
        # same-device validation. Large changes are kept as explicit mismatches.
        if box_error>.1 or conf_error>1e-4:
            mismatches.append(frame_record)
        current,details=restore_work_masks(cached["soft_mask"],image.shape[:2],cached["boxes"],
            transform=meta,return_diagnostics=True)
        native_map={int(candidate):i for i,candidate in enumerate(result.candidate_index)}
        for ci,detail in enumerate(details):
            affected=detail["status"]=="fallback" and detail.get("reason")=="saturated_probability_has_unbounded_logit_interval"
            proposed=normalized_fallback(cached["soft_mask"][ci],cached["boxes"][ci],image.shape[:2]) if affected else current[ci]
            native=result.soft_masks[native_map[ci]]>.5 if ci in native_map else torch.zeros(image.shape[:2],dtype=torch.bool)
            row={"video_id":frame["video_id"],"frame_idx":int(frame["frame_idx"]),"candidate_index":ci,
                "saturation_fallback":affected,"saturated_observations":detail.get("saturated_observations",0),
                "native_empty":ci not in native_map,"current_iou_against_cpu_native":binary_iou(current[ci]>.5,native),
                "proposed_iou_against_cpu_native":binary_iou(proposed>.5,native)}
            gpu_failure=next((f for f in baseline["failures"] if f["video_id"]==frame["video_id"] and int(f["frame_idx"])==int(frame["frame_idx"]) and f["candidate_index"]==ci),None)
            if gpu_failure:
                with np.load(args.validation/"failures"/f"{stem}_{gpu_failure['instance_idx']}.npz",allow_pickle=False) as data:
                    gpu_native=torch.from_numpy(data["soft_probabilities"])>.5
                row.update(saved_gpu_vs_cpu_native_iou=binary_iou(gpu_native,native),
                    proposed_iou_against_saved_gpu_native=binary_iou(proposed>.5,gpu_native))
            rows.append(row)
        frames.append(frame_record)
        print(json.dumps({"frames":len(frames),"total":len(selected),"candidates":len(rows),
            "affected":sum(r["saturation_fallback"] for r in rows),
            "below_0_98":sum(r["proposed_iou_against_cpu_native"]<.98 for r in rows)}),flush=True)
        del result,current,image
    report={"diagnostic_only":True,"production_unchanged":True,"validation_device":"cpu",
        "reference_note":"stored GPU working fields vs CPU replay native references; exact saved GPU reference also checked where available; full GPU gate remains required",
        "checkpoint_sha256":baseline["checkpoint_sha256"],"frames":len(frames),"candidates":len(rows),
        "saturation_fallback_cases":sum(r["saturation_fallback"] for r in rows),
        "proposed_failures_below_0_98":sum(r["proposed_iou_against_cpu_native"]<.98 for r in rows),
        "proposed_min_iou":min((r["proposed_iou_against_cpu_native"] for r in rows),default=None),
        "saturation_min_iou":min((r["proposed_iou_against_cpu_native"] for r in rows if r["saturation_fallback"]),default=None),
        "candidate_correspondence_mismatches":mismatches,"elapsed_seconds":time.perf_counter()-started,
        "synthetic_checks":synthetic,"frame_records":frames,"rows":rows}
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in {"rows","frame_records"}},indent=2))


if __name__=="__main__":
    main()
