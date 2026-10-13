"""Actual pretrained LoRA parity, joint gradients, and fixed-sample short training."""
import argparse
import hashlib
import json
import time
import torch
import torch.nn.functional as F
from PIL import Image
from .models import load_models, JointCodec
from .data import Images
from .train import (load_config, resolved, setup, make_frequency, make_perceptual,
                    make_optimizer, optimizer_parameters, schedule)


def gradnorm(parameters):
    terms=[p.grad.float().square().sum() for p in parameters if p.grad is not None]
    return sum(terms).sqrt().item() if terms else 0.


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/pid_codec/stage2b.json')
    parser.add_argument('--short-steps',type=int,default=50)
    args=parser.parse_args();cfg=load_config(args.config);setup(cfg['seed'])
    dest=resolved(cfg['output_dir'])/'checks';dest.mkdir(parents=True,exist_ok=True)
    codec,reference,decoder=load_models(cfg,'2B')
    metric=make_perceptual(cfg,'2B');frequency=make_frequency(cfg,'2B')
    optimizer=make_optimizer(codec,decoder,cfg,'2B')
    scheduler=schedule(optimizer,cfg['stage_2b_steps'],cfg['warmup_steps'],cfg['lr_end_fraction'])
    data=Images(resolved(cfg['manifest_dir']),'val',cfg['seed'])
    saved=torch.load(resolved(cfg['initial_checkpoint']),map_location='cpu',weights_only=False,mmap=True)
    for name,p in codec.state_dict().items():torch.testing.assert_close(p.cpu(),saved['encoder'][name],rtol=0,atol=0)
    del saved
    frozen={n:(p,p._version) for n,p in decoder.named_parameters() if not p.requires_grad}
    parameter_ids={id(p) for p in optimizer_parameters(optimizer)}
    assert all(id(p) not in parameter_ids for p,_ in frozen.values())
    assert all(p.dtype==torch.float32 for p in decoder.lora_parameters())
    report=dict(source=codec.initialization,initialization_exact=True,lora=decoder.lora_metadata,
                lora_parameters=sum(p.numel() for p in decoder.lora_parameters()),
                encoder_parameters=sum(p.numel() for p in codec.parameters()),checks=[])
    x=data[0][None].cuda()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        z=codec(x,5)
        enabled=decoder(z)
        with decoder.disable_lora():disabled=decoder(z)
    torch.testing.assert_close(enabled,disabled,rtol=0,atol=0)
    report['zero_lora_output_exact']=True
    (dest/'initialization.json').write_text(json.dumps(report,indent=2)+'\n')
    def picture(t,path):
        a=((t[0].clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).cpu().numpy()
        Image.fromarray(a).save(path)
    picture(x,dest/'original.png');picture(enabled,dest/'before.png')
    del enabled,disabled,z
    for rate in [0,5]:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            zhat,z,symbols,rates=codec(x,rate,True)
            pred=decoder(zhat)
        assert zhat.shape==(1,16,16,16)
        mse=F.mse_loss(pred.float(),x)
        lp=metric(pred.float(),x).mean()*cfg['lambda_lpips']
        freq=frequency(pred,x)*cfg['lambda_frequency']
        di=metric.dists_loss(pred.float(),x).mean()*cfg['lambda_dists']
        loss_parts=dict(mse=mse,lpips=lp,jpeg=freq,dists=di)
        probe=next(p for n,p in decoder.model.net.named_parameters() if '.lora_B.' in n)
        contributions={}
        for name,loss in loss_parts.items():
            grads=torch.autograd.grad(loss,(zhat,codec.projection.weight,probe),retain_graph=True)
            assert all(torch.isfinite(g).all() and g.norm()>0 for g in grads),name
            contributions[name]=dict(weighted_loss=float(loss.detach()),latent_grad=float(grads[0].norm()),
                                     projection_grad=float(grads[1].norm()),lora_probe_grad=float(grads[2].norm()))
        # Reconstruction alone, without reference alignment, must train both ends.
        sum(loss_parts.values()).backward()
        enc=gradnorm(codec.encoder.parameters());proj=gradnorm(codec.projection.parameters());lo=gradnorm(decoder.lora_parameters())
        assert min(enc,proj,lo)>0
        assert all(p.grad is None and not p.requires_grad and p._version==version for p,version in frozen.values())
        assert all(p.grad is None and not p.requires_grad for p in metric.parameters())
        report['checks'].append(dict(rate=rate,encoder_grad=enc,projection_grad=proj,lora_grad=lo,contributions=contributions))
        print(json.dumps(report['checks'][-1]),flush=True)
        del pred,zhat,z,mse,lp,freq,di,loss_parts,grads,loss
    optimizer.zero_grad(set_to_none=True)
    (dest/'gradient.json').write_text(json.dumps(report,indent=2)+'\n')
    # One-image microbatches cycle through a fixed 16-image diagnostic set and all rates.
    started=time.perf_counter();rows=[]
    for step in range(args.short_steps):
        optimizer.zero_grad(set_to_none=True)
        x=data[step%16][None].cuda();rate=step%6;ref=reference(x)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            zhat=codec(x,rate);pred=decoder(zhat)
        losses=dict(mse=F.mse_loss(pred.float(),x),lpips=metric(pred.float(),x).mean(),
                    frequency=frequency(pred,x),dists=metric.dists_loss(pred.float(),x).mean(),
                    alignment=F.mse_loss(zhat.float(),ref.float()))
        total=losses['mse']+cfg['lambda_lpips']*losses['lpips']+cfg['lambda_frequency']*losses['frequency']+cfg['lambda_dists']*losses['dists']+cfg['alignment_start']*losses['alignment']
        assert torch.isfinite(total)
        total.backward()
        norm=torch.nn.utils.clip_grad_norm_(optimizer_parameters(optimizer),cfg['grad_clip'],error_if_nonfinite=True)
        if step>0:
            assert gradnorm([p for n,p in decoder.model.net.named_parameters() if '.lora_A.' in n])>0
        optimizer.step();scheduler.step()
        row=dict(step=step+1,rate=rate,loss=float(total.detach()),grad_norm=float(norm),
                 **{k:float(v.detach()) for k,v in losses.items()})
        rows.append(row)
        with (dest/'short_train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    assert all(p.grad is None and not p.requires_grad and p._version==version for p,version in frozen.values())
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        x=data[0][None].cuda();after=decoder(codec(x,5))
    picture(after,dest/'after.png')
    report.update(short_steps=args.short_steps,seconds=time.perf_counter()-started,
                  frozen_parameters_unchanged=True,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,short_losses=rows)
    (dest/'short_train.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Stage 2B gradient and short training checks passed',flush=True)

if __name__=='__main__':main()
