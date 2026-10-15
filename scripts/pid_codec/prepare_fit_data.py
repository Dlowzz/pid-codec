"""Keep the indexed OpenImages split and add explicitly in-training benchmarks.

Every image appears exactly once in the training index; the existing sampler
uniformly shuffles the union. No recursive OpenImages scan is performed.
"""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image


def prepare(source, destination, benchmarks):
    source=Path(source);destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    db=sqlite3.connect(destination/'images.sqlite')
    db.execute('ATTACH DATABASE ? AS previous',(str(source/'images.sqlite'),))
    db.execute('CREATE TABLE images(id INTEGER PRIMARY KEY,path TEXT UNIQUE NOT NULL,source TEXT NOT NULL,split_hash TEXT NOT NULL)')
    db.execute("INSERT INTO images SELECT * FROM previous.images WHERE source='openimages'")
    oi=np.array([r[0] for r in db.execute('SELECT id FROM images ORDER BY id')],dtype=np.int64)
    train=np.load(source/'train_ids.npy');val=np.load(source/'val_ids.npy')
    train=train[np.isin(train,oi)];val=val[np.isin(val,oi)]
    fit=[];files=[];by_source={}
    for name,directory,expected in benchmarks:
        paths=sorted(Path(directory).glob('*.png'))
        if len(paths)!=expected:raise ValueError(f'{name}: expected {expected} PNGs, found {len(paths)}')
        by_source[name]=[]
        for path in paths:
            path=path.resolve()
            with Image.open(path) as im:
                im.load();size=list(im.size)
                if name=='tecnick100' and (im.size!=(1200,1200) or im.mode!='RGB'):
                    raise ValueError(f'Not the classic Tecnick RGB image: {path}')
            digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
            cur=db.execute('INSERT INTO images(path,source,split_hash) VALUES(?,?,?)',
                           (str(path),name,hashlib.sha256(str(path).encode()).hexdigest()))
            fit.append(cur.lastrowid);by_source[name].append(cur.lastrowid)
            files.append(dict(id=cur.lastrowid,path=str(path),source=name,size=size,sha256=digest))
    fit=np.array(fit,dtype=np.int64)
    # Round-robin sources for diagnostic validation; all images still enter training.
    ordered=[ids[i] for i in range(max(map(len,by_source.values()))) for ids in by_source.values() if i<len(ids)]
    samples=np.concatenate([train,fit])
    assert len(np.unique(samples))==len(samples)
    assert not set(val)&set(samples)
    for name,ids in [('train',samples),('val',val),('fit',np.array(ordered,dtype=np.int64))]:
        np.save(destination/f'{name}_ids.npy',ids)
    db.commit()
    h=hashlib.sha256()
    for row in db.execute('SELECT * FROM images ORDER BY id'):
        h.update(json.dumps(row,separators=(',',':')).encode()+b'\n')
    for ids in [samples,val,np.array(ordered,dtype=np.int64)]:h.update(ids.tobytes())
    counts=dict(db.execute('SELECT source,COUNT(*) FROM images GROUP BY source'))
    db.close()
    result=dict(format=1,fingerprint=h.hexdigest(),sources=counts,train_count=len(samples),
        unique_train_count=len(train)+len(fit),openimages_train_count=len(train),val_count=len(val),fit_count=len(fit),
        sampling='Uniform shuffled union; each image appears once; no benchmark oversampling.',
        fit_sampling_fraction=len(fit)/len(samples),
        split='OpenImages train/val membership preserved exactly from source; benchmarks intentionally enter training.',
        fit_note='Fitting diagnostics only. Kodak/Tecnick/CLIC validation are NOT held-out test results for this run.',
        source_manifest=str(source),source_fingerprint=json.loads((source/'manifest.json').read_text())['fingerprint'],
        removed_sources=['lsdir','mlic_prepared'],benchmark_files=files,
        deduplication='Exact path dedup; no claim of cross-source content deduplication')
    (destination/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='benchmark_files'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kodak',type=Path,required=True);p.add_argument('--tecnick',type=Path,required=True)
    p.add_argument('--clic',type=Path,required=True)
    a=p.parse_args()
    prepare(a.source,a.output,[('kodak24',a.kodak,24),('tecnick100',a.tecnick,100),('clic2020_pro_valid',a.clic,41)])
