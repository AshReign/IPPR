"""Fill exact working-mask areas; reuse immutable native/B libraries, never train.

Every nonempty work field must match the old float16 payload SHA and each instance
SHA before area acceptance. Existing real cache samples and zero-candidate frames
need no forward. Fresh forwards use the unchanged P2 cache predictor and defaults.
"""
import argparse
import os
from pathlib import Path
import sys

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--output-root', type=Path, required=True)
ap.add_argument('--diagnostic-root', type=Path, required=True)
ap.add_argument('--p2-root', type=Path, default=Path(__file__).resolve().parents[1])
ap.add_argument('--epochs', nargs='*', type=int)
ap.add_argument('--limit', type=int, help='Maximum ordinal per checkpoint, for prefix verification')
args = ap.parse_args()
ROOT, OLD, P2 = args.output_root.resolve(), args.diagnostic_root.resolve(), args.p2_root.resolve()
for key, sub in {'YOLO_CONFIG_DIR':'ultralytics','MPLCONFIGDIR':'matplotlib','TORCH_HOME':'torch',
                 'HF_HOME':'huggingface','TMPDIR':'tmp','XDG_CACHE_HOME':'cache','CUDA_CACHE_PATH':'cuda_cache'}.items():
    dest=ROOT/'runtime'/sub; dest.mkdir(parents=True,exist_ok=True); os.environ[key]=str(dest)
os.environ.update(PYTHONDONTWRITEBYTECODE='1', YOLO_AUTOINSTALL='false', WANDB_DISABLED='true',
                  WANDB_MODE='disabled',COMET_MODE='DISABLED',HF_HUB_DISABLE_TELEMETRY='1',
                  OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
sys.dont_write_bytecode=True
sys.path.insert(0,str(P2))
os.chdir(ROOT)
import gc, hashlib, importlib.util, json, resource, shutil, sqlite3, time

def guard(event, params):
    if event=='socket.connect': raise RuntimeError('Network disabled in area collection')
    if event=='open':
        path,mode,flags=params
        writing=(mode and any(c in mode for c in 'wax+')) or (flags and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if writing and isinstance(path,(str,bytes,os.PathLike)):
            resolved=Path(path).resolve()
            if not resolved.is_relative_to(ROOT) and str(resolved)!='/dev/null':
                raise RuntimeError(f'Write outside new selection directory: {resolved}')
sys.addaudithook(guard)
import numpy as np, cv2, torch
from ultralytics import YOLO
from ultralytics.utils import ops,nms
from p2 import soft_masks
from p2.cache_io import VideoCacheReader,VideoCacheWriter
cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
torch.set_num_threads(2); torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=True
torch.backends.cudnn.deterministic=False
torch.backends.cudnn.benchmark=False
torch.use_deterministic_algorithms(False)
torch.set_float32_matmul_precision('highest')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def canon(obj): return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)
def check(ok,msg):
    if not ok: raise RuntimeError(msg)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m; spec.loader.exec_module(m); return m
COLLECTION=P2/'outputs/selection_candidates_formal_v3_01'
reader=load('p4_native_reader',COLLECTION/'code_snapshot/tools/collect_dev_predictions.py')
INDEX=json.loads((COLLECTION/'index.json').read_text())
MANIFEST=json.loads((OLD/'DELIVERY_SHA256.json').read_text())['files']
def old_hash(rel):
    return MANIFEST.get(rel,MANIFEST.get(rel.replace('/','\\')))['sha256']

def run_one(entry):
    epoch=entry['epoch_zero_based']; started=time.perf_counter()
    native_path=P2/entry['path']; bpath=OLD/'B'/f'epoch{epoch:03d}.sqlite'
    check(sha(bpath)==old_hash(f'B/epoch{epoch:03d}.sqlite'),'B input SHA changed')
    original=reader.SelectionCandidateReader(native_path)
    check(original.complete and len(original)==2421,'Original collection incomplete')
    bm=sqlite3.connect(f'file:{bpath}?mode=ro',uri=True)
    metadata=dict(bm.execute('SELECT key,value FROM metadata')); bs=json.loads(metadata['spec'])
    check(metadata['complete']=='true' and bs['native_db_sha256']==sha(native_path),'B/native source mismatch')
    ck=Path(original.metadata['checkpoint']['path'])
    check(sha(ck)==entry['checkpoint_sha256']==bs['checkpoint_sha256'],'Checkpoint mismatch')
    check(original.metadata['config']['inference']==soft_masks.PREDICT_DEFAULTS,'Inference changed')
    check(original.metadata['config']['nms']==soft_masks.NMS_FIXED,'NMS changed')
    spec={'schema':'p2-working-area-sidecar-v1','checkpoint_sha256':sha(ck),'native_db_sha256':sha(native_path),
          'encoded_db_sha256':sha(bpath),'script_sha256':sha(__file__),'mask_rule':'float16_work > 0.5',
          'area_keep':'1000 * work_binary_area >= work_height * work_width','work_source':'unchanged P2 v3 float16',
          'split':'dev','completed_epoch':epoch+1,'expected_frames':2421,'expected_candidates':bs['expected_candidates']}
    dest=ROOT/'areas'/f'epoch{epoch:03d}.sqlite'
    db=sqlite3.connect(dest)
    db.executescript('''PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL;
       CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
       CREATE TABLE IF NOT EXISTS frames(ordinal INTEGER PRIMARY KEY,image_id INTEGER,work_h INTEGER,work_w INTEGER,
          n_candidates INTEGER,source TEXT,work_payload_sha256 TEXT,identity_verified INTEGER);
       CREATE TABLE IF NOT EXISTS areas(ordinal INTEGER,candidate_index INTEGER,work_binary_area INTEGER,
          work_sha256 TEXT,PRIMARY KEY(ordinal,candidate_index));''')
    prior=dict(db.execute('SELECT key,value FROM metadata'))
    if prior: check(json.loads(prior['spec'])==spec,'Resume provenance changed; use fresh output')
    else:
        with db: db.executemany('INSERT INTO metadata VALUES (?,?)',[('spec',canon(spec)),('complete','false')])
    n,lo,hi=db.execute('SELECT count(*),min(ordinal),max(ordinal) FROM frames').fetchone()
    check(n==0 or (lo==0 and hi==n-1),'Area prefix not contiguous')
    check(db.execute('SELECT count(*) FROM areas WHERE ordinal>=?',(n,)).fetchone()[0]==0,'Orphan area row')
    if prior.get('complete')=='true':
        check(n==2421 and db.execute('SELECT count(*) FROM areas').fetchone()[0]==bs['expected_candidates'],'False complete')
        print(canon({'epoch':epoch+1,'status':'ALREADY_COMPLETE'}),flush=True); db.close();bm.close();original.close();return
    model=None; new_forward=0; reused=0; empty=0; retained_videos=set()
    sample_root=ROOT/'areas'/'work_samples';sample_root.mkdir(exist_ok=True)
    for existing in sample_root.glob(f'epoch{epoch:03d}_*.json'):
        retained_videos.add(json.loads(existing.read_text())['video_id'])
    limit=min(2421,args.limit if args.limit is not None else 2421)
    for ordinal in range(n,limit):
        raw=original.read(ordinal); frame=raw['frame']; candidates=raw['candidates']
        pair=json.loads(bm.execute('SELECT pairing_json FROM frames WHERE ordinal=?',(ordinal,)).fetchone()[0])
        known=bm.execute('SELECT candidate_index,work_sha256 FROM masks WHERE ordinal=? ORDER BY candidate_index',(ordinal,)).fetchall()
        check(pair['image_id']==frame['image_id'] and pair['source_image_sha256']==frame['image_sha256'],'Frame identity differs')
        check(pair['candidate_count']==len(candidates)==len(known),'Candidate count differs')
        check(pair['identity_checked'] and pair['original_native_rle_exact_equal'] and pair['box_conf_byte_equal'],'Previous identity gate absent')
        check([x[0] for x in known]==list(range(len(candidates))),'Stored order differs')
        wh,ww=map(int,pair['work_shape_hw']); sample=None
        possible=[OLD/'B'/'work_samples'/f'epoch{epoch:03d}_{ordinal:04d}.sqlite']
        if epoch==4: possible.append(P2/'reports/formal_epoch5_gpu_v3_validation/roundtrip_samples'/f"{frame['video_id']}_{frame['frame_idx']:06d}.sqlite")
        for candidate_path in possible:
            if candidate_path.is_file(): sample=candidate_path;break
        if not candidates:
            work=np.empty((0,wh,ww),dtype='<f2'); source='empty_from_verified_candidate_library'; empty+=1
        elif sample is not None:
            cr=VideoCacheReader(sample); saved=cr.read(0)
            check(cr.complete and len(cr)==1 and cr.metadata['checkpoint_sha256']==spec['checkpoint_sha256'],'Cache specimen differs')
            cr.close(); work=saved['soft_mask'];boxes=saved['boxes']; conf=saved['conf']
            source='existing_float16_cache_sample';reused+=1
        else:
            if model is None: model=YOLO(str(ck))
            content=Path(frame['image_path']).read_bytes()
            check(hashlib.sha256(content).hexdigest()==frame['image_sha256'],'JPEG SHA changed')
            image=cv2.imdecode(np.frombuffer(content,np.uint8),cv2.IMREAD_COLOR)
            check(image is not None and image.shape[:2]==(frame['height'],frame['width']),'Image geometry differs')
            result=model.predict(image,predictor=soft_masks.CacheSoftMaskPredictor,device='0',**soft_masks.PREDICT_DEFAULTS)[0]
            check(not model.predictor.model.fp16,'Half inference is forbidden')
            runtime=reader.runtime_identity(result.boxes.data.device,str(result.boxes.data.dtype).removeprefix('torch.'))
            reader.require_equal_runtime(runtime,original.metadata['cuda_equivalence']['runtime'])
            boxes=result.boxes.xyxy.detach().cpu().numpy(); conf=result.boxes.conf.detach().cpu().numpy()
            cls=result.boxes.cls.detach().cpu().numpy();work=result.work_soft_masks.detach().cpu().numpy()
            check(len(work)==len(candidates) and np.all(cls==0),'Fresh candidate identity/count differs')
            check(np.array_equal(result.candidate_index.cpu().numpy(),np.arange(len(candidates))),'Fresh candidate order differs')
            for key in ('format_version','original_shape_hw','work_shape_hw','work_encoder','work_has_padding','prototype_crop_tlbr'):
                check(result.p2_transform[key]==raw['transform'][key],f'Transform differs: {key}')
            source='fresh_forward_exact_previous_float16_hash';new_forward+=1
            del result,image,content
        check(work.dtype==np.dtype('<f2') and work.shape==(len(candidates),wh,ww),'Working dtype/shape differs')
        check(hashlib.sha256(work.tobytes()).hexdigest()==pair['work_payload_sha256'],'Frame work SHA differs: stop, never positional coercion')
        rows=[]
        for ci,c in enumerate(candidates):
            check(c['candidate_index']==ci and c['class_id']==0,'Candidate index/class differs')
            check(c['box'].astype('<f4').tobytes()==boxes[ci].astype('<f4').tobytes(),'Box bytes differ')
            check(np.asarray(c['conf'],dtype='<f4').tobytes()==np.asarray(conf[ci],dtype='<f4').tobytes(),'Confidence bytes differ')
            worksha=hashlib.sha256(work[ci].tobytes()).hexdigest()
            check(worksha==known[ci][1],'Instance work SHA differs')
            rows.append((ordinal,ci,int(np.count_nonzero(work[ci]>0.5)),worksha))
        # A bounded real float16 roundtrip specimen per video/epoch enables direct
        # adapter/reuse comparisons without saving all raw soft fields.
        if candidates and frame['video_id'] not in retained_videos:
            outsample=sample_root/f'epoch{epoch:03d}_{ordinal:04d}.sqlite'
            wm={**raw['transform'],'checkpoint_sha256':spec['checkpoint_sha256'],'source_video':frame['video_id'],
                'source_frame_idx':frame['frame_idx'],'purpose':'P4 adapter validation; not full video cache'}
            writer=VideoCacheWriter(outsample,wm,resume=False);writer.append(0,0.,None,boxes,conf,work);writer.finish(1);writer.close()
            cr=VideoCacheReader(outsample); loaded=cr.read(0); cr.close()
            check(loaded['soft_mask'].tobytes()==work.tobytes(),'New sample roundtrip changed')
            dump(outsample.with_suffix('.json'),{'epoch_zero_based':epoch,'ordinal':ordinal,'video_id':frame['video_id'],
                 'image_id':frame['image_id'],'source':source,'work_payload_sha256':pair['work_payload_sha256']})
            retained_videos.add(frame['video_id'])
        with db:
            db.execute('INSERT INTO frames VALUES (?,?,?,?,?,?,?,?)',(ordinal,frame['image_id'],wh,ww,len(candidates),source,pair['work_payload_sha256'],1))
            db.executemany('INSERT INTO areas VALUES (?,?,?,?)',rows)
        if (ordinal+1)%100==0 or ordinal==limit-1:
            state={'epoch':epoch+1,'frames':ordinal+1,'expected_frames':2421,'new_forwards_this_invocation':new_forward,
                   'sample_reuse_this_invocation':reused,'empty_reuse_this_invocation':empty,'elapsed_seconds':time.perf_counter()-started,
                   'rss_peak_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'status':'RUNNING'}
            dump(ROOT/'areas'/'progress.json',state);print(canon(state),flush=True)
        del work,rows
    complete=limit==2421
    if complete:
        check(db.execute('SELECT count(*) FROM areas').fetchone()[0]==bs['expected_candidates'],'Final candidate count differs')
        check(db.execute('PRAGMA integrity_check').fetchone()[0]=='ok','Area DB integrity failure')
        with db: db.execute("UPDATE metadata SET value='true' WHERE key='complete'")
    totals=dict(db.execute('SELECT source,count(*) FROM frames GROUP BY source'))
    db.close();bm.close();original.close();del model;gc.collect();torch.cuda.empty_cache()
    report={'completed_epoch':epoch+1,'complete':complete,'sources':totals,'elapsed_seconds':time.perf_counter()-started,
            'sha256':sha(dest),'script_sha256':sha(__file__),'rss_peak_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    dump(ROOT/'areas'/f'epoch{epoch:03d}_report.json',report);print(canon(report),flush=True)

def main():
    check(torch.cuda.is_available(),'GPU unavailable')
    check(shutil.disk_usage(ROOT).free>10*1024**3,'Less than 10GiB free')
    for name,path in [('soft_masks',soft_masks.__file__),('installed_ops',ops.__file__),('installed_nms',nms.__file__)]:
        check(sha(path)==INDEX['config']['source_sha256'][name],f'Locked source changed: {name}')
    dump(ROOT/'reports'/'area_environment.json',{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
         'numpy_path':np.__file__,'cv2':cv2.__version__,'script_sha256':sha(__file__),'inference':soft_masks.PREDICT_DEFAULTS,
         'nms':soft_masks.NMS_FIXED,'source_module':str(soft_masks.__file__),'arguments':sys.argv})
    for entry in INDEX['checkpoints']:
        if args.epochs is None or entry['epoch_zero_based'] in args.epochs: run_one(entry)
    dump(ROOT/'areas'/'run_status.json',{'status':'REQUESTED_SCOPE_COMPLETE','prefix_only':args.limit is not None,
        'epochs':args.epochs,'script_sha256':sha(__file__)})
if __name__=='__main__': main()
