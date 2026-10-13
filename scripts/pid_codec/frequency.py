"""DeCo-style RGB/YCbCr block-DCT loss, evaluated in FP32.

Reference equations and JPEG tables: Zehong-Ma/DeCo, commit pinned in
configs/deco_source.json, training_repa_DeCo.py. No velocity supervision here.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class FrequencyLoss(nn.Module):
    def __init__(self, quality=85, mode='inv_gamma', gamma=1.0):
        super().__init__()
        if mode not in ('inv','inv_gamma'):raise ValueError(mode)
        n=torch.arange(8,dtype=torch.float32)
        k=n[:,None]
        scale=torch.full((8,),math.sqrt(2/8));scale[0]=math.sqrt(1/8)
        self.register_buffer('dct_matrix',scale[:,None]*torch.cos(math.pi*(2*n+1)*k/16))
        luminance=torch.tensor([
            [16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],
            [14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],
            [18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],
            [49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]],dtype=torch.float32)
        chrominance=torch.tensor([
            [17,18,24,47,99,99,99,99],[18,21,26,66,99,99,99,99],
            [24,26,56,99,99,99,99,99],[47,66,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99],[99,99,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99],[99,99,99,99,99,99,99,99]],dtype=torch.float32)
        quality=max(1,min(100,int(quality)))
        factor=5000/quality if quality<50 else 200-2*quality
        weights=[]
        for table in (luminance,chrominance,chrominance):
            q=torch.floor((table*factor+50)/100).clamp(1,255)
            w=1/q if mode=='inv' else (q.mean()/q)**gamma
            weights.append(w/w.mean())
        self.register_buffer('frequency_weight',torch.stack(weights)[None,:,None,None])

    @staticmethod
    def ycbcr(x):
        r,g,b=x[:,0:1],x[:,1:2],x[:,2:3]
        return torch.cat((.299*r+.587*g+.114*b,
                          -.168736*r-.331264*g+.5*b,
                          .5*r-.418688*g-.081312*b),dim=1)

    def dct(self,x):
        b,c,h,w=x.shape
        if h%8 or w%8:x=F.pad(x,(0,(-w)%8,0,(-h)%8),mode='reflect')
        bh,bw=x.shape[-2]//8,x.shape[-1]//8
        blocks=x.unfold(2,8,8).unfold(3,8,8).contiguous().view(-1,8,8)
        mat=self.dct_matrix.to(x)
        return (mat[None]@blocks@mat.t()[None]).view(b,c,bh,bw,8,8)

    def forward(self,pred,target):
        with torch.autocast(device_type=pred.device.type,enabled=False):
            error=self.dct(self.ycbcr(pred.float()))-self.dct(self.ycbcr(target.float()))
            return (self.frequency_weight*error.square()).mean()
