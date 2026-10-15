"""Fixed calibrated per-channel scalar steps; hard forward and straight-through gradients."""
import json
from collections import Counter
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn


class MultiRateQuantizer(nn.Module):
    def __init__(self, anchors, channels=16):
        super().__init__()
        if isinstance(anchors,(str,Path)):
            anchors=json.loads(Path(anchors).read_text())['deltas']
        delta=torch.as_tensor(anchors,dtype=torch.float32)
        if delta.ndim!=2 or delta.shape[1]!=channels or not torch.isfinite(delta).all() or not (delta>0).all():
            raise ValueError(f'Expected finite positive [rates,{channels}] quantization anchors')
        self.register_buffer('deltas',delta)

    @property
    def num_rates(self):return self.deltas.shape[0]

    def forward(self,z,rates=None,return_symbols=False):
        if z.ndim!=4 or z.shape[1]!=self.deltas.shape[1]:raise ValueError('Latent channels must match quantization anchors')
        if rates is None:rates=torch.randint(self.num_rates,(z.shape[0],),device=z.device)
        rates=torch.as_tensor(rates,device=z.device,dtype=torch.long)
        if rates.ndim==0:rates=rates.expand(z.shape[0])
        if rates.shape!=(z.shape[0],):raise ValueError('One rate ID per image is required')
        delta=self.deltas[rates,:,None,None]
        with torch.autocast(device_type=z.device.type,enabled=False):
            y=z.float()/delta
            integers=torch.round(y)
            q=y+(integers-y).detach() if torch.is_grad_enabled() else integers
            result=q*delta
        return (result,integers.detach().to(torch.int64),rates) if return_symbols else result


class SymbolEntropy:
    """Pooled zero-order, per-channel empirical entropy. No coding/header cost."""
    def __init__(self,channels=16):
        self.counts=[Counter() for _ in range(channels)]
        self.pixels=0

    def update(self,symbols,image_pixels=512*512):
        a=symbols.detach().cpu().numpy() if torch.is_tensor(symbols) else np.asarray(symbols)
        for c in range(a.shape[1]):
            values,counts=np.unique(a[:,c],return_counts=True)
            self.counts[c].update({int(v):int(n) for v,n in zip(values,counts)})
        self.pixels+=len(a)*image_pixels

    def merge(self,other):
        for a,b in zip(self.counts,other.counts):a.update(b)
        self.pixels+=other.pixels

    def result(self):
        bits=0.;symbols=0;entropies=[]
        for counts in self.counts:
            n=sum(counts.values())
            h=-sum((v/n)*math.log2(v/n) for v in counts.values()) if n else 0.
            bits+=h*n;symbols+=n;entropies.append(h)
        return dict(estimated_bpp=bits/self.pixels if self.pixels else 0.,
                    empirical_entropy=bits/symbols if symbols else 0.,
                    channel_entropy_bits=entropies,symbol_count=symbols,image_pixels=self.pixels,
                    entropy_protocol='per-channel zero-order entropy pooled over evaluated images; accounts only for supplied symbols, excludes coder/header overhead')


def estimate_bpp(latents,deltas):
    symbols=np.rint(np.asarray(latents,dtype=np.float32)/np.asarray(deltas,dtype=np.float32)[None,:,None,None]).astype(np.int64)
    stat=SymbolEntropy(channels=symbols.shape[1]);stat.update(symbols);return stat.result()
