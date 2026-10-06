"""One-time index merge; never recursively scan datasets at training startup."""
import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def prepare(args):
    dest=ROOT/'data/pid_codec';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'manifest.json').exists():
        print((dest/'manifest.json').read_text());return
    original=Path(args.sqlite_index).resolve() if args.sqlite_index else None
    lists=[Path(p).resolve() for p in args.image_list]
    if not original and not lists:
        raise ValueError('Supply --image-list files and/or --sqlite-index; no dataset is included')
    temp=dest/'images.building.sqlite'
    if temp.exists(): temp.unlink()
    db=sqlite3.connect(temp)
    db.execute('CREATE TABLE images(id INTEGER PRIMARY KEY,path TEXT UNIQUE NOT NULL,source TEXT NOT NULL,split_hash TEXT NOT NULL)')
    counts={}
    def insert(path,source):
        p=str(Path(path).resolve())
        db.execute('INSERT OR IGNORE INTO images(path,source,split_hash) VALUES(?,?,?)',
                   (p,source,hashlib.sha256(('pid-stage1-split-3407:'+p).encode()).hexdigest()))
    if original:
        src=sqlite3.connect(f'file:{original}?mode=ro',uri=True)
        # Optional existing index schema: images(id, root, path, source).
        for base,path,source in src.execute('SELECT root,path,source FROM images ORDER BY id'):
            insert(Path(base)/path,source)
        src.close()
    for i,listing in enumerate(lists):
        with listing.open() as f:
            for line in f:
                name=line.strip()
                if name:
                    p=Path(name)
                    insert(p if p.is_absolute() else listing.parent/p,f'image_list_{i}')
    db.commit()
    ids=[r[0] for r in db.execute('SELECT id FROM images ORDER BY split_hash,id')]
    if len(ids)<=2000:
        raise ValueError('Need more than 2000 images for the fixed validation split')
    val=np.array(ids[:2000],dtype=np.int64)
    train=np.array(sorted(ids[2000:]),dtype=np.int64)
    np.save(dest/'train_ids.npy',train);np.save(dest/'val_ids.npy',val)
    h=hashlib.sha256()
    for row in db.execute('SELECT id,path,source,split_hash FROM images ORDER BY id'):
        h.update(json.dumps(row,separators=(',',':')).encode()+b'\n')
    h.update(train.tobytes());h.update(val.tobytes())
    counts=dict(db.execute('SELECT source,COUNT(*) FROM images GROUP BY source').fetchall())
    checks=[]
    from PIL import Image
    for source in counts:
        # Check actual files from each source without rewalking 881k paths.
        for path, in db.execute('SELECT path FROM images WHERE source=? ORDER BY id LIMIT 4',(source,)):
            with Image.open(path) as im: checks.append({'path':path,'size':list(im.size)})
    db.close();os.replace(temp,dest/'images.sqlite')
    manifest=dict(format=1,fingerprint=h.hexdigest(),sources=counts,train_count=len(train),val_count=len(val),
                  local_index=str(original) if original else None,image_lists=[str(p) for p in lists],sample_checks=checks,
                  split='lowest 2000 SHA256(pid-stage1-split-3407:path); immutable IDs',
                  deduplication='exact path dedup; no cross-source pixel dedup',
                  decode_errors='explicit training error with path; never silently change the sample sequence')
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


def download():
    from huggingface_hub import snapshot_download
    config=json.loads((ROOT/'configs/pid_codec/stage1.json').read_text())
    dest=ROOT/config['dcae_dir']
    # All writes, including the Hugging Face partial-download cache, stay on data.
    snapshot_download(config['dcae_repo'],revision=config['dcae_revision'],local_dir=dest,
                      allow_patterns=['config.json','diffusion_pytorch_model.safetensors'],max_workers=2)
    meta=json.loads((ROOT/'configs/dcae_weights.json').read_text())
    assert config['dcae_repo']==meta['repo'] and config['dcae_revision']==meta['revision']
    expected=meta['sha256']
    h=hashlib.sha256()
    with (dest/'diffusion_pytorch_model.safetensors').open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    assert h.hexdigest()==expected,'DC-AE download checksum mismatch'
    (dest/'verified.json').write_text(json.dumps(dict(repo=config['dcae_repo'],revision=config['dcae_revision'],sha256=h.hexdigest()),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['data','download'])
    parser.add_argument('--image-list',action='append',default=[])
    parser.add_argument('--sqlite-index')
    args=parser.parse_args()
    prepare(args) if args.action=='data' else download()
