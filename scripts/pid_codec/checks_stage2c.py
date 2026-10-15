"""Real pretrained Stage 2C shape, SG, zero-init, gradient and frozen-state checks."""
import argparse,json,time
import torch
import torch.nn.functional as F
from .models import load_models,JointCodec
from .data import Images
from .train import (load_config,resolved,setup,make_perceptual,make_frequency,make_optimizer,
                    optimizer_parameters,compute_loss,schedule)
from .checks_stage2b import gradnorm


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/pid_codec/stage2c.json');p.add_argument('--short-steps',type=int,default=12)
    args=p.parse_args();cfg=load_config(args.config);setup(cfg['seed'])
    out=resolved(cfg['output_dir'])/'checks';out.mkdir(parents=True,exist_ok=True)
    codec,reference,decoder=load_models(cfg,'2C');joint=JointCodec(codec,decoder).train()
    joint_mode=decoder.joint_finetune
    metric=make_perceptual(cfg,'2C');frequency=make_frequency(cfg,'2C');optimizer=make_optimizer(codec,decoder,cfg,'2C')
    scheduler=schedule(optimizer,cfg['stage_2c_steps'],cfg['warmup_steps'],cfg['lr_end_fraction'])
    vae=decoder.model.vae_encoder.model.model
    frozen=[(p,p._version) for mod in [codec,decoder.model,metric] for p in mod.parameters() if not p.requires_grad]
    assert not any(p.requires_grad for p in vae.parameters())
    assert all(p.requires_grad for p in decoder.side_branch.parameters())
    expected={id(p) for p in decoder.side_branch.parameters()}
    if joint_mode:
        assert all(p.requires_grad for p in codec.parameters()) and codec.training
        expected.update(id(p) for p in codec.parameters())
        expected.update(id(p) for p in decoder.lora_parameters())
        assert all(('.lora_A.' in n or '.lora_B.' in n)==p.requires_grad for n,p in decoder.model.named_parameters())
    else:
        assert not any(p.requires_grad for p in codec.parameters())
        assert not any(p.requires_grad for p in decoder.model.parameters())
    assert {id(p) for p in optimizer_parameters(optimizer)}==expected
    from peft.tuners.tuners_utils import BaseTunerLayer
    assert all(not m.disable_adapters for m in decoder.model.net.modules() if isinstance(m,BaseTunerLayer))
    saved=torch.load(resolved(cfg['initial_checkpoint']),map_location='cpu',weights_only=False,mmap=True)
    assert saved['stage']==('2C' if joint_mode else '2B')
    source_step=saved['stage_step']
    for n,v in codec.state_dict().items():torch.testing.assert_close(v.cpu(),saved['encoder'][n],rtol=0,atol=0)
    for n,v in decoder.lora_state_dict().items():torch.testing.assert_close(v,saved['pid_lora'][n],rtol=0,atol=0)
    if joint_mode:
        mapping=decoder.side_initialization['key_mapping']
        for n,v in saved['side_branch'].items():torch.testing.assert_close(decoder.side_branch.state_dict()[mapping[n]].cpu(),v,rtol=0,atol=0)
        source_side_state=saved['side_branch'];source_side_cfg=saved['side_config']
    del saved
    data=Images(resolved(cfg['manifest_dir']),'train',cfg['seed']);x=data[(0,0)][None].cuda()
    calls=[];h=codec.encoder.register_forward_hook(lambda *unused:calls.append(1))
    modulation_calls=[0]*len(decoder.model.net.patch_blocks);handles=[]
    for i,block in enumerate(decoder.model.net.patch_blocks):
        def count(module,args,output,index=i):
            if decoder.side_branch.adaln._features is not None:modulation_calls[index]+=1
        handles.append(block.adaLN_modulation_img.register_forward_hook(count))
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):d=joint(x,5,True)
    h.remove()
    for handle in handles:handle.remove()
    assert len(calls)==1
    assert d['fx'].shape==(1,16,64,64) and d['fbase'].shape==(1,16,64,64)
    assert d['side'].shape==(1,8,16,16) and d['residual'].shape==(1,3,512,512)
    side_quantized=decoder.side_branch.quantizer is not None
    if side_quantized:
        side_steps=decoder.side_branch.quantizer.deltas.clone()
        assert side_steps.shape==(6,8) and not list(decoder.side_branch.quantizer.parameters())
        delta=side_steps[d['rates'],:,None,None]
        expected=torch.round(d['side_raw'].float()/delta)
        torch.testing.assert_close(d['side_symbols'],expected.to(torch.int64),rtol=0,atol=0)
        torch.testing.assert_close(d['side'],expected*delta,rtol=0,atol=0)
    if not joint_mode:
        assert torch.count_nonzero(d['residual'])==0
        torch.testing.assert_close(d['base'],d['conditioned'],rtol=0,atol=0)
        torch.testing.assert_close(d['base'],d['prediction'],rtol=0,atol=0)
    torch.testing.assert_close(d['fx'],decoder.model.encode_lq_latent(x).float(),rtol=0,atol=0)
    selected=decoder.side_branch.adaln.block_indices
    assert all(modulation_calls[i]==4 for i in selected)
    expansion_exact=False
    if joint_mode and decoder.side_initialization['added_zero_blocks']:
        from .side import SideBranch
        current=decoder.side_branch
        baseline=SideBranch(source_side_cfg,decoder.model.net).cuda()
        baseline.load_state_dict(source_side_state,strict=True)
        baseline.quantizer=current.quantizer
        try:
            decoder.side_branch=baseline
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):previous=joint(x,5,True)
            for key in ['side','conditioned','residual','prediction']:
                torch.testing.assert_close(d[key],previous[key],rtol=0,atol=0)
            expansion_exact=True
        finally:
            decoder.side_branch=current
            for handle in baseline.adaln._handles:handle.remove()
        del baseline,previous,source_side_state
    with torch.no_grad():
        with decoder.disable_lora():without_lora=decoder(d['zhat'])
    lora_effect=(d['base']-without_lora).abs().mean().item();assert lora_effect>0
    probe=d['base'].detach().float().requires_grad_(True)
    vloss=F.mse_loss(decoder.full_vae(probe).float(),d['fx'].float())
    vg=torch.autograd.grad(vloss,probe)[0];assert vg.norm()>0 and torch.isfinite(vg).all()
    report=dict(initialization_exact=True,joint_finetune=joint_mode,source_step=source_step,source=codec.initialization,parameter_counts=decoder.side_branch.parameter_counts(),
        adaln_expansion_reconstruction_exact=expansion_exact,
        side_quantized=side_quantized,side_hard_round_exact=side_quantized,
        trainable_parameter_counts={g['name']:sum(p.numel() for p in g['params']) for g in optimizer.param_groups},
        main_encoder_calls=len(calls),feature_shape=list(d['fx'].shape),side_shape=list(d['side'].shape),patch_grid=decoder.side_branch.adaln.patch_grid,
        adapted_blocks=selected,zero_init_reconstruction_exact=not joint_mode,full_vae_matches_official=True,full_vae_input_grad=float(vg.norm()),
        lora_enabled=True,lora_output_effect=lora_effect,side_condition_steps=4,checks=[])
    del d,probe,vg,vloss,without_lora
    (out/'initialization.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    start=time.perf_counter()
    for step in range(args.short_steps):
        optimizer.zero_grad(set_to_none=True);x=data[(step%4,step%4)][None].cuda()
        loss,parts=compute_loss(joint,reference,decoder,metric,x,'2C',0.,frequency,cfg['lambda_frequency'])
        assert torch.isfinite(loss);loss.backward()
        norms={g['name']:gradnorm(g['params']) for g in optimizer.param_groups}
        if step>0 or joint_mode:assert min(norms.values())>0,norms
        head_norms={str(index):gradnorm(head.parameters()) for index,head in zip(selected,decoder.side_branch.adaln.heads)}
        assert min(head_norms.values())>0,head_norms
        assert all(p.grad is None and p._version==v and not p.requires_grad for p,v in frozen)
        assert decoder.side_branch.adaln._features is None
        norm=torch.nn.utils.clip_grad_norm_(optimizer_parameters(optimizer),cfg['grad_clip'],error_if_nonfinite=True)
        probes={}
        for group in optimizer.param_groups:
            param=max((p for p in group['params'] if p.grad is not None),key=lambda p:float(p.grad.abs().max()))
            index=int(param.grad.abs().flatten().argmax())
            probes[group['name']]=(param,index,float(param.detach().flatten()[index]))
        optimizer.step();scheduler.step()
        updates={name:float(param.detach().flatten()[index])-before for name,(param,index,before) in probes.items()}
        if joint_mode:assert all(value!=0 for value in updates.values()),updates
        row=dict(step=step+1,gradients=norms,adaln_head_gradients=head_norms,parameter_update_probes=updates,gradient_norm=float(norm),**{k:float(v) for k,v in parts.items()})
        report['checks'].append(row);print(json.dumps(row),flush=True)
    # Separate paths must each propagate into the query, not just the RGB shortcut.
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):d=joint(x,0,True)
    for key in ['base','fx','fbase']:assert not d[key].requires_grad,key
    for key in ['zhat','z']:assert d[key].requires_grad==joint_mode,key
    assert d['side'].requires_grad and d['conditioned'].requires_grad
    if side_quantized:
        ste=torch.autograd.grad(d['side'].sum(),d['side_raw'],retain_graph=True)[0]
        torch.testing.assert_close(ste,torch.ones_like(ste),rtol=0,atol=0)
        torch.testing.assert_close(decoder.side_branch.quantizer.deltas,side_steps,rtol=0,atol=0)
        report.update(side_ste_gradient_exact=True,side_anchors_fixed=True)
    query=decoder.side_branch.query.output_projection.weight
    ga=torch.autograd.grad(F.mse_loss(d['conditioned'].float(),x),(query,),retain_graph=True)[0]
    gr=torch.autograd.grad(d['residual'].square().mean(),(query,),retain_graph=True)[0]
    assert ga.norm()>0 and gr.norm()>0
    if joint_mode:
        ge=torch.autograd.grad(F.mse_loss(d['conditioned'].float(),x),codec.projection.weight,retain_graph=True)[0]
        assert ge.norm()>0 and torch.isfinite(ge).all()
        report['conditioned_to_encoder_gradient']=float(ge.norm())
    # Compare checkpoint ON/OFF with identical grad mode. Separately record
    # inference differences: BF16 kernels may change with gradient requirements.
    with torch.autocast('cuda',dtype=torch.bfloat16):
        decoder.activation_checkpoint=False;direct=joint(x,0,True);decoder.activation_checkpoint=True
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):inference=joint(x,0,True)
    parity={k:dict(checkpoint_max=float((d[k].detach()-direct[k].detach()).abs().max()),
                   inference_max=float((d[k].detach()-inference[k]).abs().max()),
                   inference_mean=float((d[k].detach()-inference[k]).abs().mean()))
            for k in ['zhat','fx','fbase','base','side','conditioned','residual','prediction']}
    print('PARITY',json.dumps(parity),flush=True)
    (out/'parity.json').write_text(json.dumps(parity,indent=2)+'\n')
    torch.testing.assert_close(d['prediction'].detach(),direct['prediction'].detach(),rtol=0,atol=0)
    # Repeated inference must remain exactly reproducible.
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):repeat=joint(x,0)
    torch.testing.assert_close(inference['prediction'],repeat,rtol=0,atol=0)
    report.update(frozen_parameters_unchanged=True,sg_paths_detached=True,adaln_to_query_grad=float(ga.norm()),rgb_to_query_grad=float(gr.norm()),
        checkpoint_forward_exact=True,inference_repeat_exact=True,mode_parity=parity,short_steps=args.short_steps,seconds=time.perf_counter()-start,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    (out/'gradient.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Stage 2C real-model checks PASSED',flush=True)

if __name__=='__main__':main()
