"""Immutable indexed split and stateless augmentation, including prefetch-safe resume."""
import hashlib
import json
import math
import random
import sqlite3
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, Sampler


class Images(Dataset):
    def __init__(self, manifest_dir, split, seed):
        self.root=Path(manifest_dir)
        self.ids=np.load(self.root/f'{split}_ids.npy',mmap_mode='r')
        self.split=split
        self.seed=seed
        self.connection=None

    def __len__(self): return len(self.ids)

    def __getstate__(self):
        state=dict(self.__dict__);state['connection']=None
        return state

    def __getitem__(self, key):
        index,draw=key if isinstance(key,tuple) else (key,key)
        if self.connection is None:
            self.connection=sqlite3.connect(f'file:{self.root}/images.sqlite?mode=ro',uri=True)
        path=self.connection.execute('SELECT path FROM images WHERE id=?',(int(self.ids[index]),)).fetchone()[0]
        # Randomness depends on committed sample position, not worker scheduling/prefetch.
        rng=random.Random((self.seed<<40)+int(draw))
        try:
            with Image.open(path) as im:
                im=ImageOps.exif_transpose(im).convert('RGB')
                w,h=im.size
                scale=512/min(w,h)
                im=im.resize((max(512,round(w*scale)),max(512,round(h*scale))),Image.Resampling.BICUBIC)
                if self.split=='train':
                    left=rng.randrange(im.width-511);top=rng.randrange(im.height-511)
                else:
                    left=(im.width-512)//2;top=(im.height-512)//2
                im=im.crop((left,top,left+512,top+512))
                if self.split=='train' and rng.random()<0.5:
                    im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                image=torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()/127.5-1
        except Exception as e:
            raise RuntimeError(f'Invalid manifest image {path}; repair/exclude explicitly before resuming') from e
        return image


class SampleStream(Sampler):
    """Global deterministic shuffles; cursor counts consumed samples across ranks.

    Workers can prefetch arbitrarily: only the training loop advances its committed
    cursor. Recreating this sampler at that cursor reproduces the next minibatch.
    """
    def __init__(self, length, cursor, micro_batch, rank, world_size, seed):
        self.length=length;self.cursor=cursor;self.micro=micro_batch
        self.rank=rank;self.world=world_size;self.seed=seed

    def __iter__(self):
        cursor=self.cursor;epoch=-1;perm=None
        while True:
            for local in range(self.micro):
                draw=cursor+self.rank*self.micro+local
                current=draw//self.length
                if current!=epoch:
                    epoch=current
                    perm=torch.randperm(self.length,generator=torch.Generator().manual_seed(self.seed+epoch))
                yield int(perm[draw%self.length]),draw
            cursor+=self.micro*self.world


def fingerprint(root):
    root=Path(root)
    return json.loads((root/'manifest.json').read_text())['fingerprint']
