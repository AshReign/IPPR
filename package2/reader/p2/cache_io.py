"""One SQLite file per video; bounded transactions and lossless fp16 compression.

The stored float16 masks are true probabilities at working resolution. zlib does
not quantize them further. No binary-mask or final-output filtering occurs here.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sqlite3
import zlib
import numpy as np

SCHEMA = 'p2-video-cache-sqlite-v1'

def canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)

def array_bytes(array, dtype):
    return np.ascontiguousarray(array, dtype=dtype).tobytes()

class VideoCacheWriter:
    def __init__(self, path, metadata, resume=True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata, cache_schema=SCHEMA)
        self.signature = hashlib.sha256(canonical(self.metadata).encode()).hexdigest()
        exists = self.path.exists()
        if exists and not resume:
            raise FileExistsError(self.path)
        self.db = sqlite3.connect(self.path)
        self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        self.db.execute('''CREATE TABLE IF NOT EXISTS frames (
            frame_idx INTEGER PRIMARY KEY, timestamp REAL NOT NULL, pts INTEGER,
            instance_count INTEGER NOT NULL, boxes BLOB NOT NULL, confidences BLOB NOT NULL,
            soft_masks BLOB NOT NULL, raw_sha256 TEXT NOT NULL)''')
        saved = dict(self.db.execute('SELECT key,value FROM metadata'))
        if saved:
            if saved.get('signature') != self.signature:
                self.db.close()
                raise ValueError('Existing cache has a different source/model/config/transform; refusing to append')
        else:
            self.db.executemany('INSERT INTO metadata VALUES (?,?)', [('signature',self.signature),
                ('spec',canonical(self.metadata)),('complete','false')])
            self.db.commit()
        n, first, last = self.db.execute('SELECT COUNT(*),MIN(frame_idx),MAX(frame_idx) FROM frames').fetchone()
        if n and (first != 0 or last != n-1):
            self.db.close()
            raise ValueError('Resume cache does not contain a contiguous zero-based prefix')
        self.next_frame = n
        self.complete = saved.get('complete') == 'true'
        row=self.db.execute('SELECT timestamp FROM frames ORDER BY frame_idx DESC LIMIT 1').fetchone()
        self.last_timestamp = row[0] if row else None
        self.pending = 0

    def append(self, frame_idx, timestamp, pts, boxes, confidences, soft_masks):
        if self.complete: raise ValueError('Cannot append to a completed cache')
        if frame_idx != self.next_frame: raise ValueError('Frame order must be contiguous from zero')
        if not np.isfinite(timestamp) or (self.last_timestamp is not None and timestamp <= self.last_timestamp):
            raise ValueError('Timestamps must be finite and strictly increasing')
        boxes=np.asarray(boxes,dtype='<f4'); scores=np.asarray(confidences,dtype='<f4')
        masks=np.asarray(soft_masks,dtype='<f2')
        n=len(scores); shape=tuple(self.metadata['work_shape_hw'])
        if boxes.shape!=(n,4) or scores.shape!=(n,) or masks.shape!=(n,*shape):
            raise ValueError(f'Instance/shape mismatch: {boxes.shape},{scores.shape},{masks.shape}')
        if not (np.isfinite(boxes).all() and np.isfinite(scores).all() and np.isfinite(masks).all()):
            raise ValueError('Nonfinite cache values')
        if np.any(scores<np.float32(.1)) or np.any(scores>1) or np.any(masks<0) or np.any(masks>1):
            raise ValueError('Cache candidate confidence/probability outside protocol range')
        br=array_bytes(boxes,'<f4'); cr=array_bytes(scores,'<f4'); mr=array_bytes(masks,'<f2')
        digest=hashlib.sha256(br+cr+mr).hexdigest()
        self.db.execute('INSERT INTO frames VALUES (?,?,?,?,?,?,?,?)',
                        (frame_idx,float(timestamp),None if pts is None else int(pts),n,br,cr,zlib.compress(mr,1),digest))
        self.next_frame+=1; self.pending+=1; self.last_timestamp=float(timestamp)
        if self.pending>=16: self.flush()

    def flush(self):
        self.db.commit(); self.pending=0

    def finish(self, expected_frames):
        if self.next_frame!=expected_frames:
            raise ValueError(f'Expected {expected_frames} decoded frames, cached {self.next_frame}')
        self.flush()
        check=self.db.execute('PRAGMA integrity_check').fetchone()[0]
        if check!='ok': raise ValueError(f'SQLite integrity failure: {check}')
        self.db.execute("INSERT OR REPLACE INTO metadata VALUES ('complete','true')")
        self.db.commit(); self.complete=True

    def close(self):
        self.flush(); self.db.close()

class VideoCacheReader:
    def __init__(self,path):
        self.path=Path(path).resolve()
        self.db=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True)
        entries=dict(self.db.execute('SELECT key,value FROM metadata'))
        self.metadata=json.loads(entries['spec']); self.complete=entries.get('complete')=='true'
        if self.metadata['cache_schema']!=SCHEMA: raise ValueError('Unsupported cache schema')
        if entries['signature']!=hashlib.sha256(canonical(self.metadata).encode()).hexdigest():
            raise ValueError('Metadata signature mismatch')

    def __len__(self):
        return self.db.execute('SELECT COUNT(*) FROM frames').fetchone()[0]

    def read(self,frame_idx):
        row=self.db.execute('SELECT * FROM frames WHERE frame_idx=?',(int(frame_idx),)).fetchone()
        if row is None: raise KeyError(frame_idx)
        idx,t,pts,n,br,cr,compressed,digest=row
        mr=zlib.decompress(compressed)
        if hashlib.sha256(br+cr+mr).hexdigest()!=digest: raise ValueError(f'Corrupted frame {idx}')
        return {'frame_idx':idx,'timestamp':t,'pts':pts,
                'boxes':np.frombuffer(br,dtype='<f4').reshape(n,4).copy(),
                'conf':np.frombuffer(cr,dtype='<f4').copy(),
                'soft_mask':np.frombuffer(mr,dtype='<f2').reshape(n,*self.metadata['work_shape_hw']).copy()}

    def close(self): self.db.close()
