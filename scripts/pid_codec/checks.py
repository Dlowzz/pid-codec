"""Meaningful resume, official-sampler parity, and reconstruction-gradient checks."""
import argparse
import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from .data import Images,SampleStream,fingerprint
from .models import ROOT,CompactEncoder,FrozenPiD
from .train import atomic_save,load_config,restore_rng,rng_state,schedule,setup,align_weight


def basic(cfg):
    setup(cfg['seed'])
    n=37;micro=3;world=2
    streams=[iter(SampleStream(n,0,micro,r,world,123)) for r in range(world)]
    draws=[]
    for _ in range(5):
        for r in range(world):draws.extend(next(streams[r]) for _ in range(micro))
    # Resume at a committed cursor even though the original stream is prefetched.
    expected=[next(streams[0]) for _ in range(micro)]
    restored=iter(SampleStream(n,5*micro*world,micro,0,world,123))
    assert expected==[next(restored) for _ in range(micro)]
    assert len({d for _,d in draws})==len(draws)
    dataset=Images(ROOT/cfg['manifest_dir'],'train',cfg['seed'])
    a=dataset[(0,123)];b=dataset[(0,123)]
    assert torch.equal(a,b) and a.shape==(3,512,512)
    model=torch.nn.Linear(4,3).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    sched=schedule(opt,8,2,.1)
    def update(m,o,s):
        x=torch.randn(2,4,device='cuda');o.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=m(x).float().square().mean()
        loss.backward();o.step();s.step();return loss.item()
    for _ in range(2):update(model,opt,sched)
    with tempfile.TemporaryDirectory(prefix='pid-resume-') as tmp:
        p=Path(tmp)/'checkpoint.pt'
        atomic_save(p,dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng=rng_state()))
        expected_loss=[update(model,opt,sched) for _ in range(2)]
        expected_params={k:v.clone() for k,v in model.state_dict().items()}
        saved=torch.load(p,map_location='cpu',weights_only=False)
        resumed=torch.nn.Linear(4,3).cuda();resumed.load_state_dict(saved['model'])
        optimizer=torch.optim.AdamW(resumed.parameters(),lr=1e-4);optimizer.load_state_dict(saved['optimizer'])
        scheduler=schedule(optimizer,8,2,.1);scheduler.load_state_dict(saved['scheduler'])
        # LambdaLR construction modifies optimizer lr; restore optimizer state
        # after scheduler construction, as the production trainer does.
        optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
        actual=[update(resumed,optimizer,scheduler) for _ in range(2)]
        assert expected_loss==actual
        assert all(torch.equal(v,resumed.state_dict()[k]) for k,v in expected_params.items())
    assert align_weight(0,50000,cfg)==.5
    assert abs(align_weight(49999,50000,cfg)-.05)<1e-12
    return dict(sample_cursor_resume=True,stateless_augmentation=True,optimizer_scheduler_rng_resume_bit_exact=True,
                alignment_endpoints=True,data_fingerprint=fingerprint(ROOT/cfg['manifest_dir']))


def gradient(cfg,encoder_test=False,diagnostic_random=False):
    import os
    import lpips
    setup(cfg['seed']);os.chdir(ROOT/'third_party/PiD')
    flags=(torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,
           torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32)
    decoder=FrozenPiD(cfg)
    assert flags==(torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,
                   torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32)
    dataset=Images(ROOT/cfg['manifest_dir'],'val',cfg['seed'])
    x=dataset[0].unsqueeze(0).cuda()
    zref=decoder.reference(x)
    assert zref.shape==(1,16,16,16)
    with torch.no_grad():
        official=decoder.model.generate_samples_from_batch({decoder.model.config.input_caption_key:[cfg['caption']],
                         'LQ_latent':zref,'degrade_sigma':torch.zeros(1,device='cuda')},
                         cfg_scale=1,num_steps=4,seed=cfg['decoder_seed'],image_size=(512,512))[:,:,0]
        ours=decoder(zref)
        torch.testing.assert_close(ours,official,atol=1e-6,rtol=1e-6)
        twice=decoder(zref)
        assert torch.equal(ours,twice)
        batched=decoder(zref.expand(2,-1,-1,-1).contiguous())
        # Batched GEMMs may have BF16 rounding differences; fixed noise is exact.
        batch_difference=(batched[0]-ours[0]).abs().max().item()
        batch_mean_difference=(batched[0]-ours[0]).abs().mean().item()
        torch.testing.assert_close(batched[0],batched[1],atol=1e-6,rtol=1e-6)
    metric=lpips.LPIPS(net='vgg',verbose=False).cuda().eval().requires_grad_(False)
    codec=None
    if encoder_test:
        codec=CompactEncoder(ROOT/cfg['dcae_dir'],cfg['encoder_activation_checkpoint'],diagnostic_random).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):z=codec(x)
        assert z.shape==(1,16,16,16)
        z.retain_grad()
    else:z=zref.detach().clone().requires_grad_(True)
    pred=decoder(z)
    loss=(pred-x).abs().mean()+0.5*metric(pred.float(),x.float()).mean()
    loss.backward()
    norm=z.grad.float().norm().item()
    assert norm>0 and torch.isfinite(z.grad).all()
    assert all(p.grad is None and not p.requires_grad for p in decoder.parameters())
    result=dict(official_four_step_parity=True,repeated_output_exact=True,batch_max_abs_difference=batch_difference,
                batch_mean_abs_difference=batch_mean_difference,identical_samples_within_batch=True,
                latent_gradient_norm=norm,reconstruction_loss=float(loss.detach()),pid_frozen=True,
                alignment_loss_in_gradient_test=False,encoder_test=encoder_test,diagnostic_random=diagnostic_random,
                backend_flags_preserved=True,reference_shape=list(zref.shape),codec_latent_shape=list(z.shape))
    if codec is not None:
        for name,module in [('encoder',codec.encoder),('projection',codec.projection)]:
            grads=[p.grad for p in module.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            result[name+'_gradient_norm']=sum(g.float().square().sum() for g in grads).sqrt().item()
            assert result[name+'_gradient_norm']>0
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['basic','gradient','encoder-gradient'])
    p.add_argument('--diagnostic-random',action='store_true');args=p.parse_args()
    cfg=load_config('configs/pid_codec/stage1.json')
    result=basic(cfg) if args.mode=='basic' else gradient(cfg,args.mode=='encoder-gradient',args.diagnostic_random)
    dest=ROOT/'runs/pid_codec/checks';dest.mkdir(parents=True,exist_ok=True)
    (dest/(args.mode+('_random' if args.diagnostic_random else '')+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
