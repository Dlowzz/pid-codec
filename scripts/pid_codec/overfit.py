"""Four fixed training images: test memorization before the formal Stage 2B run.

Uses the same joint forward, losses, optimizers and quantizer as production.
Only the diagnostic sampling/batch and warmup duration are shortened. Never
initializes production from this diagnostic checkpoint.
"""
import argparse
import json
import time
from pathlib import Path
import torch
from torch import nn
from PIL import Image,ImageDraw
from .models import load_models
from .data import Images,fingerprint
from .quantization import SymbolEntropy
from .train import (load_config,resolved,setup,make_trainable,make_optimizer,
                    make_perceptual,make_frequency,compute_loss,optimizer_parameters,
                    schedule,atomic_save,rng_state,restore_rng)


class FixedRate(nn.Module):
    def __init__(self,joint,rate):
        super().__init__();self.joint=joint;self.rate=rate
    @property
    def quantizer(self):return self.joint.codec.quantizer
    def forward(self,x,return_details=False):
        return self.joint(x,rates=self.rate,return_details=return_details)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/pid_codec/stage2b.json')
    parser.add_argument('--max-steps',type=int,default=500)
    parser.add_argument('--resume',action='store_true',help='Continue diagnostic weights without changing pass criteria')
    args=parser.parse_args();cfg=load_config(args.config);setup(cfg['seed'])
    out=resolved(cfg['output_dir'])/'overfit_4'
    if out.exists() and not args.resume:raise RuntimeError(f'Preserve existing diagnostic run: {out}')
    out.mkdir(parents=True,exist_ok=args.resume)
    codec,reference,decoder=load_models(cfg,'2B')
    metric=make_perceptual(cfg,'2B');frequency=make_frequency(cfg,'2B')
    optimizer=make_optimizer(codec,decoder,cfg,'2B')
    # Peak LR, loss coefficients, AdamW, clip, AMP, anchors, and noise match production.
    # Twenty warmup steps let this short diagnostic reach those LRs promptly.
    scheduler=schedule(optimizer,args.max_steps,20,cfg['lr_end_fraction'])
    rate=codec.quantizer.num_rates-1
    joint=FixedRate(make_trainable(codec,decoder,'2B'),rate)
    data=Images(resolved(cfg['manifest_dir']),'train',cfg['seed'])
    indices=list(range(4));x=torch.stack([data[(i,i)] for i in indices]).cuda()
    zref=reference(x)
    frozen={n:(p,p._version) for n,p in decoder.named_parameters() if not p.requires_grad}
    metadata=dict(training_indices=indices,augmentation_draws=indices,augmentation='cached once; no new crop/flip',
                  split='train',data_fingerprint=fingerprint(resolved(cfg['manifest_dir'])),
                  rate=rate,micro_batch=4,gradient_accumulation=1,warmup_steps=20,
                  lr_encoder=cfg['stage_2b_lr'],lr_pid_lora=cfg['stage_2b_lora_lr'],
                  decoder_seed=cfg['decoder_seed'],max_steps=args.max_steps,production_initialization=codec.initialization,
                  criteria=dict(minimum_steps=100,mean_psnr_gain_db=3.,lpips_relative_drop=.20,
                                reconstruction_objective_relative_drop=.30,each_image_psnr_gain_db=.5))
    (out/'settings.json').write_text(json.dumps(metadata,indent=2)+'\n')
    def pic(t):
        return Image.fromarray(((t.clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).cpu().numpy())
    @torch.no_grad()
    def evaluate(step):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            zhat,z,symbols,rates,pred=joint(x,return_details=True)
        psnr=(-10*torch.log10(((pred.float()-x)/2).square().mean((1,2,3)).clamp_min(1e-12)))
        mse=(pred.float()-x).square().mean();lp=metric(pred,x).mean()
        freq=frequency(pred,x);di=metric.dists_loss(pred,x).mean()
        align=(zhat.float()-zref).square().mean()
        rec=mse+cfg['lambda_lpips']*lp+cfg['lambda_frequency']*freq+cfg['lambda_dists']*di
        entropy=SymbolEntropy();entropy.update(symbols)
        row=dict(step=step,psnr=psnr.mean().item(),per_image_psnr=psnr.tolist(),mse=mse.item(),
                 lpips=lp.item(),frequency=freq.item(),dists=di.item(),alignment=align.item(),
                 reconstruction_objective=rec.item(),total=(rec+cfg['alignment_start']*align).item(),
                 estimated_bpp=entropy.result()['estimated_bpp'])
        with (out/'evaluation.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        return row,pred.detach().cpu()
    initial,before=evaluate(0);history=[initial];passed=False;start_step=0;prior_elapsed=0.
    if args.resume:
        saved=torch.load(out/'diagnostic.pt',map_location='cpu',weights_only=False)
        assert saved['debug'] and saved['config']==cfg
        assert not saved['report']['passed'], 'Diagnostic already passed'
        codec.load_state_dict(saved['encoder'],strict=True)
        decoder.load_lora_state_dict(saved['pid_lora'])
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        restore_rng(saved['rng'])
        start_step=saved['stage_step'];assert args.max_steps>start_step
        initial=saved['report']['initial'];history=saved['report']['history']
        prior_elapsed=saved['report']['elapsed_seconds']
        metadata['continuation_from_step']=start_step
        metadata['continuation_note']='Same images, objective, peak LRs, and success thresholds; extend cosine horizon'
        (out/'settings.json').write_text(json.dumps(metadata,indent=2)+'\n')
        del saved
    for i in indices:pic(x[i].cpu()).save(out/f'original_{i}.png')
    started=time.perf_counter()
    for step in range(start_step+1,args.max_steps+1):
        optimizer.zero_grad(set_to_none=True)
        loss,parts=compute_loss(joint,lambda _:zref,decoder,metric,x,'2B',cfg['alignment_start'],frequency,cfg['lambda_frequency'])
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite overfit loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(optimizer_parameters(optimizer),cfg['grad_clip'],error_if_nonfinite=True)
        optimizer.step();scheduler.step()
        row=dict(step=step,gradient_norm=float(norm),**{k:float(v) for k,v in parts.items()})
        with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if step%25==0 or step==args.max_steps:
            final,after=evaluate(step);history.append(final)
            gains=[a-b for a,b in zip(final['per_image_psnr'],initial['per_image_psnr'])]
            passed=(step>=100 and final['psnr']-initial['psnr']>=3 and
                    final['lpips']<=initial['lpips']*.8 and
                    final['reconstruction_objective']<=initial['reconstruction_objective']*.7 and min(gains)>=.5)
            canvas=Image.new('RGB',(1536,4*542),'white');draw=ImageDraw.Draw(canvas)
            for i in indices:
                for j,(t,label) in enumerate([(x[i].cpu(),'Original'),(before[i],'Before'),(after[i],f'After {step} steps')]):
                    canvas.paste(pic(t),(512*j,542*i+30));draw.text((512*j+8,542*i+8),label,fill='black')
            canvas.save(out/'comparison.jpg',quality=94)
            assert all(p.grad is None and not p.requires_grad and p._version==version for p,version in frozen.values())
            report=dict(**metadata,passed=passed,step=step,initial=initial,final=final,history=history,
                        elapsed_seconds=prior_elapsed+time.perf_counter()-started,frozen_pid_unchanged=True,
                        psnr_gain_db=final['psnr']-initial['psnr'],
                        lpips_relative_drop=1-final['lpips']/initial['lpips'],
                        reconstruction_objective_relative_drop=1-final['reconstruction_objective']/initial['reconstruction_objective'])
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            if passed or step==args.max_steps:
                atomic_save(out/'diagnostic.pt',dict(format='pid-codec-overfit-debug-v1',debug=True,stage='2B',stage_step=step,
                    encoder=codec.state_dict(),pid_lora=decoder.lora_state_dict(),pid_lora_config=decoder.lora_metadata,
                    optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(),config=cfg,report=report))
                break
    if not passed:raise RuntimeError('Tiny-set overfit did not meet predeclared criteria; do not start production')
    print('Four-image overfit PASSED; diagnostic weights must NOT initialize production.',flush=True)

if __name__=='__main__':main()
