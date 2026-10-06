"""Two explicit codec-encoder stages; official models/sampler, no entropy or GAN."""
import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .data import Images, SampleStream, fingerprint
from .models import ROOT, FrozenPiD, load_models


def resolved(value):
    p=Path(value)
    return p if p.is_absolute() else ROOT/p


def load_config(path):
    cfg=json.loads(resolved(path).read_text())
    for key in ['output_dir','dcae_dir','manifest_dir']:
        if not resolved(cfg[key]).resolve().is_relative_to(ROOT):
            raise ValueError(f'{key} must stay under {ROOT}; external data roots are read-only')
    if cfg.get('ema'): raise ValueError('EMA is disabled in this experiment')
    return cfg


def setup(seed,local_rank=0):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.cuda.set_device(local_rank)
    # Leave room for CUDA context/libraries in addition to allocator tensors.
    torch.cuda.set_per_process_memory_fraction(0.94)
    torch.set_num_threads(4)
    torch.set_grad_enabled(True)
    torch.set_autocast_cache_enabled(False)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    os.environ.setdefault('TORCH_HOME',str(ROOT/'checkpoints/.torch-cache'))


def schedule(optimizer,total,warmup,end_fraction):
    def factor(step):
        if step<warmup:return (step+1)/max(warmup,1)
        progress=min(1.,(step-warmup)/max(1,total-1-warmup))
        return end_fraction+(1-end_fraction)*(1+math.cos(math.pi*progress))/2
    return torch.optim.lr_scheduler.LambdaLR(optimizer,factor)


def align_weight(step,total,cfg):
    progress=min(1.,max(0.,step/max(total-1,1)))
    return cfg['alignment_start']+(cfg['alignment_end']-cfg['alignment_start'])*progress


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda'])


def atomic_save(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    def tensor_bytes(value):
        if torch.is_tensor(value):return value.numel()*value.element_size()
        if isinstance(value,dict):return sum(tensor_bytes(v) for v in value.values())
        if isinstance(value,(list,tuple)):return sum(tensor_bytes(v) for v in value)
        return 0
    required=int(tensor_bytes(payload)*1.15)+2**20
    if shutil.disk_usage(path.parent).free<required:
        raise RuntimeError('Insufficient data-disk space for an atomic checkpoint; previous latest remains valid')
    try:
        torch.save(payload,temp)
        with temp.open('rb') as f:os.fsync(f.fileno())
        os.replace(temp,path)
    except BaseException:
        if temp.exists():temp.unlink()
        raise


def alias(source,destination):
    temp=destination.with_suffix('.linktmp')
    if temp.exists():temp.unlink()
    os.link(source,temp);os.replace(temp,destination)


def training_identity(cfg,stage,data_hash):
    keys=['seed','decoder_seed','caption','dcae_repo','dcae_revision','stage_a_steps','stage_b_steps',
          'stage_a_lr','stage_b_lr','warmup_steps','lr_end_fraction','weight_decay','grad_clip',
          'alignment_start','alignment_end','encoder_activation_checkpoint','pid_activation_checkpoint']
    payload={k:cfg[k] for k in keys}
    payload.update(stage=stage,data_hash=data_hash)
    assets=[ROOT/cfg['pid_args'],ROOT/cfg['dcae_dir']/'verified.json',ROOT/'configs/pid_weights.json']
    payload['asset_metadata_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in assets}
    # Changes to forward/sampler/loss code cannot silently reuse an optimizer cursor.
    source_paths=[Path(__file__),Path(__file__).with_name('models.py'),Path(__file__).with_name('data.py'),
                  ROOT/'third_party/PiD/pid/_src/models/pid_distill_model_infer.py']
    payload['code_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def make_loader(cfg,micro,cursor,rank,world,workers=None):
    data=Images(resolved(cfg['manifest_dir']),'train',cfg['seed'])
    sampler=SampleStream(len(data),cursor,micro,rank,world,cfg['seed'])
    workers=cfg['workers'] if workers is None else workers
    opts=dict(batch_size=micro,sampler=sampler,num_workers=workers,pin_memory=True,drop_last=True,
              generator=torch.Generator().manual_seed(cfg['seed']+rank))
    if workers:opts.update(persistent_workers=True,prefetch_factor=cfg['prefetch_factor'])
    return DataLoader(data,**opts)


def compute_loss(codec,reference,decoder,perceptual,x,stage,weight):
    zref=reference(x)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        z=codec(x)
        align=F.mse_loss(z.float(),zref.float())
        if stage=='A':
            total=align;l1=align.new_zeros(());lp=align.new_zeros(())
        else:
            reconstruction=decoder(z)
            l1=F.l1_loss(reconstruction.float(),x.float())
            # Frozen LPIPS params, but retain its input gradient.
            with torch.autocast('cuda',enabled=False):
                lp=perceptual(reconstruction.float(),x.float()).mean()
            total=l1+0.5*lp+weight*align
    return total,{'loss':total.detach(),'alignment':align.detach(),'l1':l1.detach(),'lpips':lp.detach()}


@torch.no_grad()
def validate(codec,decoder,cfg,stage,step,out,full=False):
    import lpips
    from PIL import Image,ImageDraw
    dataset=Images(resolved(cfg['manifest_dir']),'val',cfg['seed'])
    count=len(dataset) if full else min(cfg['validation_images'],len(dataset))
    rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    owned=decoder is None
    if owned:decoder=FrozenPiD(cfg)
    metric=lpips.LPIPS(net='vgg',version='0.1',verbose=False).cuda().eval().requires_grad_(False)
    codec.eval()
    sums=dict(alignment=0.,psnr=0.,lpips=0.,l1=0.,baseline_psnr=0.,baseline_lpips=0.)
    images=out/'validation'/f'step{step:06d}';images.mkdir(parents=True,exist_ok=True)
    for i in range(rank,count,world):
        x=dataset[i].unsqueeze(0).cuda()
        reference=decoder.reference(x)
        with torch.autocast('cuda',dtype=torch.bfloat16): z=codec(x)
        pred=decoder(z).float();base=decoder(reference).float()
        sums['alignment']+=F.mse_loss(z.float(),reference).item()
        sums['l1']+=F.l1_loss(pred,x).item()
        sums['psnr']+=-10*torch.log10(F.mse_loss((pred+1)/2,(x+1)/2).clamp_min(1e-12)).item()
        sums['baseline_psnr']+=-10*torch.log10(F.mse_loss((base+1)/2,(x+1)/2).clamp_min(1e-12)).item()
        sums['lpips']+=metric(pred,x).mean().item();sums['baseline_lpips']+=metric(base,x).mean().item()
        if i<cfg['validation_visuals']:
            canvas=Image.new('RGB',(1536,542),'white');draw=ImageDraw.Draw(canvas)
            for j,(t,label) in enumerate([(x,'Original512'),(base,'Official downsample + VAE -> PiD'),(pred,'Codec encoder -> PiD')]):
                arr=((t[0].clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).cpu().numpy()
                canvas.paste(Image.fromarray(arr),(512*j,30));draw.text((512*j+6,8),label,fill='black')
            canvas.save(images/f'val{i:04d}.jpg',quality=92)
    if world>1:
        values=torch.tensor(list(sums.values()),device='cuda',dtype=torch.float64)
        dist.all_reduce(values);sums=dict(zip(sums,values.tolist()))
    result={k:v/count for k,v in sums.items()}
    result.update(step=step,count=count,stage=stage,psnr_protocol='unrounded float RGB [0,1]',l1_range='[-1,1]')
    result['selection_loss']=result['alignment'] if stage=='A' else result['l1']+0.5*result['lpips']
    if rank==0:(out/f'validation_{step:06d}.json').write_text(json.dumps(result,indent=2)+'\n')
    codec.train();del metric
    if owned:del decoder
    gc.collect();torch.cuda.empty_cache()
    if rank==0:
        for old in sorted((out/'validation').glob('step*'))[:-3]:shutil.rmtree(old)
    return result


def trial(cfg,args):
    """A real optimizer-step trial; no checkpoints, no permanent model mutation."""
    setup(cfg['seed'])
    total=torch.cuda.get_device_properties(0).total_memory
    target=cfg['stage_a_effective_batch_target'] if args.stage=='A' else cfg['stage_b_effective_batch_target']
    accum=max(1,math.ceil(target/(args.micro_batch*args.profile_world_size)))
    result=dict(stage=args.stage,micro_batch=args.micro_batch,gradient_accumulation=accum,
                effective_batch=args.micro_batch*accum*args.profile_world_size,gpu=torch.cuda.get_device_name(),total_bytes=total,workers=args.workers)
    try:
        codec,reference,decoder=load_models(cfg,args.stage)
        optimizer=torch.optim.AdamW(codec.parameters(),lr=cfg['stage_a_lr'] if args.stage=='A' else cfg['stage_b_lr'],weight_decay=cfg['weight_decay'])
        perceptual=None
        if decoder is not None:
            import lpips
            perceptual=lpips.LPIPS(net='vgg',verbose=False).cuda().eval().requires_grad_(False)
        loader=iter(make_loader(cfg,args.micro_batch,0,0,1,args.workers))
        torch.cuda.reset_peak_memory_stats()
        times=[];waits=[];device_peak=0
        for i in range(2+cfg['profile_measured_steps']):
            torch.cuda.synchronize();start=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            waited=0.;mean_loss=0.
            for _ in range(accum):
                begin=time.perf_counter()
                x=next(loader).cuda(non_blocking=True);torch.cuda.synchronize()
                waited+=time.perf_counter()-begin
                loss,_=compute_loss(codec,reference,decoder,perceptual,x,args.stage,0.5)
                (loss/accum).backward();mean_loss+=loss.detach().item()/accum
            norm=torch.nn.utils.clip_grad_norm_(codec.parameters(),cfg['grad_clip'],error_if_nonfinite=True)
            assert norm.item()>0 and torch.isfinite(loss)
            optimizer.step();torch.cuda.synchronize()
            free_now,_=torch.cuda.mem_get_info();device_peak=max(device_peak,total-free_now)
            if i>=2:times.append(time.perf_counter()-start);waits.append(waited)
        result.update(ok=True,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                      device_used_peak=device_peak,
                      images_per_second=args.micro_batch*accum/(sum(times)/len(times)),data_wait_fraction=sum(waits)/sum(times),
                      loss=mean_loss,grad_norm=float(norm))
    except torch.OutOfMemoryError as e:
        result.update(ok=False,oom=True,error=str(e))
    except Exception as e:
        result.update(ok=False,oom=False,error=repr(e))
        resolved(args.result).write_text(json.dumps(result,indent=2));raise
    resolved(args.result).write_text(json.dumps(result,indent=2)+'\n')


def train(cfg,args):
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    setup(cfg['seed']+rank,local)
    if world>1:dist.init_process_group('nccl')
    out=resolved(args.output) if args.output else resolved(cfg['output_dir'])/f'stage_{args.stage}'
    if not out.resolve().is_relative_to(ROOT):raise ValueError('Training writes must remain under the project directory')
    out.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(out).free<cfg['minimum_free_disk_gib']*2**30:
        raise RuntimeError(f"Need at least {cfg['minimum_free_disk_gib']} GiB free on data before training/checkpoint writes")
    profile=json.loads(resolved(args.profile).read_text())
    if profile['stage']!=args.stage or profile['world_size']!=world:raise ValueError('Re-profile stage/current GPU count')
    micro=profile['micro_batch'];accum=profile['gradient_accumulation'];effective=micro*accum*world
    cfg['workers']=profile['workers']
    if args.validation_images is not None:cfg['validation_images']=args.validation_images
    data_hash=fingerprint(resolved(cfg['manifest_dir']))
    identity=training_identity(cfg,args.stage,data_hash)
    if profile.get('identity')!=identity:raise ValueError('Code/data/training config changed since profiling; re-profile')
    hardware=profile['hardware'][local]
    if torch.cuda.get_device_name(local)!=hardware['name'] or torch.cuda.get_device_properties(local).total_memory!=hardware['total']:
        raise ValueError('GPU differs from profile; re-profile')
    free_now,_=torch.cuda.mem_get_info()
    required=profile['measured']['peak_reserved']+2*2**30
    if free_now<required:raise RuntimeError('GPU memory occupied since profiling; wait before starting training')
    steps=cfg['stage_a_steps'] if args.stage=='A' else cfg['stage_b_steps']
    stop=min(steps,args.max_steps) if args.max_steps else steps
    codec,reference,decoder=load_models(cfg,args.stage)
    optimizer=torch.optim.AdamW(codec.parameters(),lr=cfg['stage_a_lr'] if args.stage=='A' else cfg['stage_b_lr'],weight_decay=cfg['weight_decay'])
    scheduler=schedule(optimizer,steps,cfg['warmup_steps'],cfg['lr_end_fraction'])
    perceptual=None
    if args.stage=='B':
        import lpips
        perceptual=lpips.LPIPS(net='vgg',verbose=False).cuda().eval().requires_grad_(False)
    step=0;cursor=0;best=float('inf');resume_rng=None
    global_step_offset=0;transition=None
    if args.resume:
        path=out/'latest.pt' if args.resume=='auto' else resolved(args.resume)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert saved['identity']==identity and saved['stage']==args.stage,'Changed code/config/data: refusing inaccurate resume'
        assert saved['batch']==dict(micro=micro,accum=accum,world=world)
        assert saved['debug']==args.debug,'Do not mix debug and production checkpoints'
        codec.load_state_dict(saved['encoder'],strict=True);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        step=saved['stage_step'];cursor=saved['sample_cursor'];best=saved['best'];resume_rng=saved['rng_by_rank'][rank]
        global_step_offset=saved['global_optimizer_step']-step
        transition=saved.get('transition')
        del saved
    elif args.init_from:
        saved=torch.load(resolved(args.init_from),map_location='cpu',weights_only=False)
        assert args.stage=='B' and saved['stage']=='A'
        if not args.debug:
            assert not saved['debug'],'Production cannot initialize from a debug checkpoint'
            assert saved['stage_complete'] or args.allow_early_transition, 'Incomplete A requires explicit --allow-early-transition'
        global_step_offset=saved['global_optimizer_step']
        transition=dict(source_checkpoint=str(resolved(args.init_from)),source_stage_step=saved['stage_step'],
                        early=not saved['stage_complete'],explicitly_allowed=args.allow_early_transition)
        assert saved['data_fingerprint']==data_hash
        codec.load_state_dict(saved['encoder'],strict=True);del saved
    elif args.stage=='B':raise ValueError('Stage B requires --init-from stage A or --resume')
    codec.train();wrapped=DDP(codec,device_ids=[local],gradient_as_bucket_view=True) if world>1 else codec
    loader=iter(make_loader(cfg,micro,cursor,rank,world))
    if resume_rng:restore_rng(resume_rng)
    if rank==0:
        (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        (out/'batch.json').write_text(json.dumps(profile,indent=2)+'\n')
    interrupted=False
    # SIGTERM is handled between optimizer steps, not in the middle of an update.
    import signal
    def request_stop(*unused):
        nonlocal interrupted
        interrupted=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    while step<stop:
        torch.cuda.synchronize();start=time.perf_counter();data_wait=0.
        optimizer.zero_grad(set_to_none=True);totals={k:0. for k in ['loss','alignment','l1','lpips']}
        weight=align_weight(step,steps,cfg) if args.stage=='B' else 1.
        torch.cuda.reset_peak_memory_stats()
        for i in range(accum):
            begin=time.perf_counter();x=next(loader).cuda(non_blocking=True);data_wait+=time.perf_counter()-begin
            sync=wrapped.no_sync() if world>1 and i<accum-1 else contextlib.nullcontext()
            with sync:
                loss,parts=compute_loss(wrapped,reference,decoder,perceptual,x,args.stage,weight)
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss; last complete checkpoint retained')
                (loss/accum).backward()
            for k,v in parts.items():totals[k]+=float(v)/accum
        norm=torch.nn.utils.clip_grad_norm_(codec.parameters(),cfg['grad_clip'],error_if_nonfinite=True)
        used_lr=optimizer.param_groups[0]['lr']
        optimizer.step();scheduler.step();step+=1;cursor+=effective
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        if world>1:
            t=torch.tensor(list(totals.values()),device='cuda');dist.all_reduce(t);totals=dict(zip(totals,(t/world).tolist()))
            stop_flag=torch.tensor(int(interrupted),device='cuda');dist.all_reduce(stop_flag,op=dist.ReduceOp.MAX);interrupted=bool(stop_flag.item())
        row=dict(stage=args.stage,step=step,global_optimizer_step=global_step_offset+step,
                 **totals,lr=used_lr,lambda_align=weight,gradient_norm=float(norm),
                 peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                 device_used_gib=(torch.cuda.get_device_properties(local).total_memory-torch.cuda.mem_get_info()[0])/2**30,
                 images_per_second=effective/elapsed,step_seconds=elapsed,data_wait_seconds=data_wait,sample_cursor=cursor,
                 micro_batch=micro,world_size=world,gradient_accumulation=accum,effective_batch=effective)
        if rank==0:
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        validation_due=step%cfg['validation_every']==0 or step==stop
        save_due=step%cfg['checkpoint_every']==0 or validation_due or interrupted
        is_best=False
        if validation_due:
            # Release update-only buffers before loading the Stage A decoder.
            # Optimizer state stays resident; no training graph is needed here.
            optimizer.zero_grad(set_to_none=True)
            del x,loss,parts
            torch.cuda.empty_cache()
            if world>1:dist.barrier()
            training_rng=rng_state()
            result=validate(codec,decoder,cfg,args.stage,step,out,full=False)
            if step==steps and cfg['full_validation_at_stage_end']:
                # Best selection always uses the same fixed subset. The full 2k
                # holdout is a separate report, never mixed with subset scores.
                validate(codec,decoder,cfg,args.stage,step,out/'full_validation',full=True)
            restore_rng(training_rng)
            is_best=result['selection_loss']<best
            if is_best:best=result['selection_loss']
            if world>1:dist.barrier()
        if save_due:
            rngs=[None]*world
            if world>1:dist.all_gather_object(rngs,rng_state())
            else:rngs[0]=rng_state()
            if rank==0:
                saved=dict(format='pid-codec-stage1-v1',encoder=codec.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    amp_scaler=None,amp_dtype='bfloat16',ema=None,stage=args.stage,stage_step=step,transition=transition,
                    global_optimizer_step=row['global_optimizer_step'],sample_cursor=cursor,rng_by_rank=rngs,
                    stage_complete=step==steps,best=best,identity=identity,data_fingerprint=data_hash,
                    batch=dict(micro=micro,accum=accum,world=world),config=cfg,debug=args.debug)
                path=out/f'step_{step:06d}.pt';atomic_save(path,saved);alias(path,out/'latest.pt')
                if is_best:alias(path,out/'best.pt')
                for old in sorted(out.glob('step_*.pt'))[:-cfg['keep_step_checkpoints']]:old.unlink()
                del saved
            if world>1:dist.barrier()
        if interrupted:break
    if world>1:dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/pid_codec/stage1.json')
    parser.add_argument('--stage',choices=['A','B'],required=True)
    parser.add_argument('--profile')
    parser.add_argument('--output');parser.add_argument('--resume');parser.add_argument('--init-from')
    parser.add_argument('--allow-early-transition',action='store_true')
    parser.add_argument('--max-steps',type=int);parser.add_argument('--validation-images',type=int)
    parser.add_argument('--debug',action='store_true');parser.add_argument('--trial',action='store_true')
    parser.add_argument('--profile-world-size',type=int,default=1)
    parser.add_argument('--micro-batch',type=int);parser.add_argument('--workers',type=int,default=8);parser.add_argument('--result')
    args=parser.parse_args();cfg=load_config(args.config)
    if args.allow_early_transition and (args.stage!='B' or not args.init_from or args.resume):
        parser.error('--allow-early-transition requires Stage B --init-from, without --resume')
    if args.trial:trial(cfg,args)
    else:
        if not args.profile:parser.error('--profile from measured profiling is required')
        if args.max_steps and not args.debug:parser.error('--max-steps requires --debug; production uses full stage length')
        train(cfg,args)


if __name__=='__main__':main()
