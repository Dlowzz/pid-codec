"""Real full-path checks for GAN finetuning, detached D updates and validation cache."""
import argparse
import json
from pathlib import Path

import torch

from .models import load_models,JointCodec
from .data import Images
from .train import (setup,load_config,resolved,make_optimizer,make_perceptual,make_frequency,
                    make_adversary,backward_batch,optimizer_parameters)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/pid_codec/stage2c_gan_fit.json')
    a=p.parse_args();cfg=load_config(a.config);setup(cfg['seed'])
    out=resolved(cfg['output_dir'])/'checks';out.mkdir(parents=True,exist_ok=True)
    saved=torch.load(resolved(cfg['initial_checkpoint']),map_location='cpu',weights_only=False,mmap=True)
    # A complete checkpoint must work even if bootstrap/anchor files are gone.
    restore_cfg=dict(cfg,initial_checkpoint='/missing/bootstrap.pt',quantization_anchors='/missing/main.json',side_quantization_anchors='/missing/side.json')
    cwd=Path.cwd();codec,reference,decoder=load_models(restore_cfg,'2C',checkpoint_state=saved)
    assert Path.cwd()==cwd
    for key,value in codec.state_dict().items():torch.testing.assert_close(value.cpu(),saved['encoder'][key],rtol=0,atol=0)
    for key,value in decoder.side_branch.state_dict().items():torch.testing.assert_close(value.cpu(),saved['side_branch'][key],rtol=0,atol=0)
    del saved
    joint=JointCodec(codec,decoder).train();metric=make_perceptual(cfg,'2C');frequency=make_frequency(cfg,'2C')
    assert frequency is None and cfg['lambda_frequency']==0
    optimizer=make_optimizer(codec,decoder,cfg,'2C');disc,dopt,_=make_adversary(cfg,'2C')
    frozen=[(p,p._version) for module in [decoder.model,metric] for p in module.parameters() if not p.requires_grad]
    data=Images(resolved(cfg['manifest_dir']),'fit',cfg['seed']);x=data[0][None].cuda()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        highest_rate=codec.quantizer.num_rates-1
        direct=joint(x,highest_rate,True)
        z=codec.encode(x);fx=decoder.full_vae(x.float())
        cached=joint(x,highest_rate,True,encoded=z,target_features=fx)
        for key in ['zhat','side','base','conditioned','prediction']:
            torch.testing.assert_close(direct[key],cached[key],rtol=0,atol=0)
    report=dict(checkpoint_restored_without_bootstrap=True,checkpoint_weights_exact=True,cwd_unchanged=True,
                cached_validation_bit_exact=True,frequency_disabled=True,discriminator_parameters=sum(p.numel() for p in disc.parameters()),steps=[])
    del direct,cached,z,fx
    for step in range(3):
        x=data[step][None].cuda();optimizer.zero_grad(set_to_none=True);dopt.zero_grad(set_to_none=True)
        rate=[0,codec.quantizer.num_rates//2,codec.quantizer.num_rates-1][step]
        encoder_calls=[];query_sg=[];decoder_grad=[]
        handles=[codec.encoder.register_forward_hook(lambda *args:encoder_calls.append(1)),
                 codec.quantizer.register_forward_pre_hook(lambda module,args:(args[0],rate)+args[2:]),
                 decoder.side_branch.query.register_forward_pre_hook(lambda module,args:query_sg.append([v.requires_grad for v in args])),
                 decoder.register_forward_hook(lambda module,args,output:decoder_grad.append(output.requires_grad))]
        loss,parts=backward_batch(joint,reference,decoder,metric,x,'2C',0.,frequency,cfg,1,disc)
        for handle in handles:handle.remove()
        assert len(encoder_calls)==1 and query_sg==[[False,False]] and decoder_grad==[False,True]
        norms={g['name']:float(torch.stack([p.grad.float().square().sum() for p in g['params'] if p.grad is not None]).sum().sqrt()) for g in optimizer.param_groups}
        assert all(value>0 for value in norms.values()),norms
        assert all(p.grad is None and p._version==version for p,version in frozen)
        assert decoder.side_branch.adaln._features is None
        norm=torch.nn.utils.clip_grad_norm_(optimizer_parameters(optimizer),cfg['grad_clip'],error_if_nonfinite=True)
        dnorm=torch.nn.utils.clip_grad_norm_(optimizer_parameters(dopt),cfg['grad_clip'],error_if_nonfinite=True)
        assert dnorm>0
        optimizer.step();dopt.step()
        row=dict(step=step+1,rate=rate,**{k:float(v) for k,v in parts.items()},group_gradient_norms=norms,
                 gradient_norm=float(norm),discriminator_gradient_norm=float(dnorm),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        report['steps'].append(row);print(json.dumps(row),flush=True)
    report.update(passed=True,sg_paths_verified=True,all_trainable_groups_have_gradients=True,frozen_parameters_unchanged=True)
    (out/'gan.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
