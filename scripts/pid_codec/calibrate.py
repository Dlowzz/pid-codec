"""Calibrate fixed quantization anchors from a frozen Stage 1B or Stage 2A encoder."""
import argparse,json,hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from .models import CompactEncoder,ROOT
from .train import load_config,resolved,setup
from .data import Images,fingerprint
from .quantization import estimate_bpp

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/pid_codec/stage2a.json')
    parser.add_argument('--checkpoint',help='Optional current Stage 2A checkpoint for anchor recalibration')
    args=parser.parse_args();cfg=load_config(args.config);setup(cfg['seed']);torch.set_grad_enabled(False)
    source=resolved(args.checkpoint or cfg['initial_checkpoint'])
    saved=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    assert saved['stage'] in ('B','2A') and not saved['debug']
    assert saved['data_fingerprint']==fingerprint(resolved(cfg['manifest_dir']))
    model=CompactEncoder(resolved(cfg['dcae_dir'])).cuda().eval().requires_grad_(False)
    state={k:v for k,v in saved['encoder'].items() if k!='quantizer.deltas'}
    model.load_state_dict(state,strict=True)
    bootstrap=None
    if saved['stage']=='2A':
        old_anchors=json.loads(resolved(saved['config']['quantization_anchors']).read_text())
        bootstrap=old_anchors.get('bootstrap_source') or {k:old_anchors[k] for k in ['source_identity','source_stage_step','data_fingerprint']}
    dest=resolved(cfg['quantization_anchors'])
    if dest.exists():raise FileExistsError('Do not overwrite anchors used by a previous run; select a new output path')
    def collect(split,indices):
        ds=Subset(Images(resolved(cfg['manifest_dir']),split,cfg['seed']),indices)
        loader=DataLoader(ds,batch_size=cfg['calibration_batch'],num_workers=cfg['workers'],pin_memory=True)
        chunks=[]
        for j,x in enumerate(loader):
            with torch.autocast('cuda',dtype=torch.bfloat16):z=model(x.cuda(non_blocking=True))
            chunks.append(z.float().cpu().numpy())
            if j%16==0:print(split,j,len(loader),flush=True)
        return np.concatenate(chunks)
    n=len(Images(resolved(cfg['manifest_dir']),'train',cfg['seed']))
    indices=torch.randperm(n,generator=torch.Generator().manual_seed(cfg['seed']))[:cfg['calibration_train_images']].tolist()
    z=collect('train',indices);std=z.transpose(1,0,2,3).reshape(16,-1).std(1).clip(1e-6)
    offset=cfg['calibration_val_offset'];vids=list(range(offset,offset+cfg['calibration_val_images']))
    val=collect('val',vids);anchors=[];reports=[]
    for target in cfg['rate_targets']:
        lo,hi=-12.,12.;best=None
        for _ in range(32):
            mid=(lo+hi)/2;delta=std*2**mid;report=estimate_bpp(val,delta)
            if best is None or abs(report['estimated_bpp']-target)<best[0]:best=(abs(report['estimated_bpp']-target),delta,report,2**mid)
            if report['estimated_bpp']>target:lo=mid
            else:hi=mid
        assert best[0]<.0003,(target,best[2])
        anchors.append(best[1].tolist());reports.append(dict(target_bpp=target,multiplier=best[3],**best[2]));print(reports[-1],flush=True)
    h=hashlib.sha256()
    with source.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    result=dict(deltas=anchors,channel_std=std.tolist(),reports=reports,train_indices=indices,val_indices=vids,
                source_checkpoint=str(source),source_sha256=h.hexdigest(),source_stage_step=saved['stage_step'],
                source_identity=saved['identity'],data_fingerprint=saved['data_fingerprint'],fixed=True)
    result['source_stage']=saved['stage']
    if bootstrap is not None:result['bootstrap_source']=bootstrap
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2)+'\n');print('Saved',dest,flush=True)
if __name__=='__main__':main()
