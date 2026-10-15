"""All-rate validation integrated with the existing training/checkpoint loop."""
import gc,json,shutil,contextlib,math
import torch
import torch.distributed as dist
import torch.nn.functional as F
from .data import Images
from .train import resolved,make_frequency,make_perceptual,write_progress
from .quantization import SymbolEntropy

@torch.no_grad()
def validate_rates(codec,decoder,cfg,step,out,full=False,stage="2A",metric=None,frequency=None,progress_dir=None):
    if stage=='2C':return validate_side_rates(codec,decoder,cfg,step,out,full,metric=metric,frequency=frequency,progress_dir=progress_dir)
    from PIL import Image,ImageDraw
    dataset=Images(resolved(cfg['manifest_dir']),'val',cfg['seed'])
    count=len(dataset) if full else min(cfg['validation_images'],len(dataset))
    rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    if metric is None:metric=make_perceptual(cfg,stage)
    if frequency is None:frequency=make_frequency(cfg,'2A')
    codec.eval();rates=codec.quantizer.num_rates
    sums=[dict(alignment=0.,psnr=0.,lpips=0.,l1=0.,frequency=0.) for _ in range(rates)]
    if stage=='2B':
        for row in sums:row.update(mse=0.,dists=0.)
    baseline=out/'baseline'
    if stage=='2B' and step==0:baseline.mkdir(parents=True,exist_ok=True)
    entropy=[SymbolEntropy() for _ in range(rates)]
    images=out/'validation'/f'step{step:06d}';images.mkdir(parents=True,exist_ok=True)
    def pil(t):return Image.fromarray(((t[0].clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).cpu().numpy())
    for i in range(rank,count,world):
        write_progress(progress_dir or out,'validation',step,image=i,count=count)
        x=dataset[i].unsqueeze(0).cuda();reference=decoder.reference(x)
        with torch.autocast('cuda',dtype=torch.bfloat16):z=codec.encode(x)
        tiles=[(pil(x),'Original 512')]
        if i<cfg['validation_visuals']:
            with decoder.disable_lora() if stage=='2B' else contextlib.nullcontext():
                tiles.append((pil(decoder(reference)),'Downsample + Qwen VAE + original PiD'))
        for r in range(rates):
            zhat,symbols,_=codec.quantizer(z,r,return_symbols=True);entropy[r].update(symbols)
            pred=decoder(zhat).float()
            vals=dict(alignment=F.mse_loss(zhat.float(),reference).item(),
                      psnr=-10*torch.log10(F.mse_loss((pred+1)/2,(x+1)/2).clamp_min(1e-12)).item(),
                      lpips=metric(pred,x).mean().item(),l1=F.l1_loss(pred,x).item(),frequency=frequency(pred,x).item() if frequency is not None else 0.)
            if stage=='2B':
                vals.update(mse=F.mse_loss(pred,x).item(),dists=metric.dists_loss(pred,x).mean().item())
            for k,v in vals.items():sums[r][k]+=v
            if i<cfg['validation_visuals']:
                pic=pil(pred);pic.save(images/f'val{i:04d}_rate{r}.png')
                if stage=='2B' and step==0:pic.save(baseline/f'val{i:04d}_rate{r}.png')
                tiles.append((pic,f'Rate {r}; PSNR {vals["psnr"]:.2f}; bpp in JSON (estimated)'))
        if i<cfg['validation_visuals']:
            if stage=='2B' and step>0:
                for r in range(rates):
                    source=baseline/f'val{i:04d}_rate{r}.png'
                    if source.exists():
                        with Image.open(source) as im:tiles.append((im.copy(),f'Stage 2A source, rate {r}'))
            canvas=Image.new('RGB',(512*4,542*math.ceil(len(tiles)/4)),'white');draw=ImageDraw.Draw(canvas)
            for j,(pic,label) in enumerate(tiles):
                left=(j%4)*512;top=(j//4)*542
                canvas.paste(pic,(left,top+30));draw.text((left+6,top+8),label,fill='black')
            canvas.save(images/f'val{i:04d}.jpg',quality=92)
    if world>1:
        values=torch.tensor([list(s.values()) for s in sums],device='cuda',dtype=torch.float64)
        dist.all_reduce(values);sums=[dict(zip(s,vs)) for s,vs in zip(sums,values.tolist())]
        gathered=[None]*world;dist.all_gather_object(gathered,entropy)
        entropy=[SymbolEntropy() for _ in range(rates)]
        for group in gathered:
            for a,b in zip(entropy,group):a.merge(b)
    reports=[dict(rate=r,target_bpp=cfg['rate_targets'][r],**{k:v/count for k,v in sums[r].items()},**entropy[r].result()) for r in range(rates)]
    result=dict(step=step,count=count,stage=stage,rates=reports,psnr_protocol='unrounded float RGB [0,1]',l1_range='[-1,1]',
                selection_loss=sum(s['l1']+.5*s['lpips'] for s in reports)/rates)
    if stage=='2B':
        result['selection_loss']=sum(s['mse']+cfg['lambda_lpips']*s['lpips']+cfg['lambda_frequency']*s['frequency']+cfg['lambda_dists']*s['dists'] for s in reports)/rates
        result.update(mse_range='[-1,1]',dists_range='[0,1]',selection_objective='mse + lpips + jpeg + 0.2*dists, mean over rates')
    if rank==0:(out/f'validation_{step:06d}.json').write_text(json.dumps(result,indent=2)+'\n')
    codec.train();del metric,frequency;gc.collect();torch.cuda.empty_cache()
    if rank==0:
        for old in sorted((out/'validation').glob('step*'))[:-3]:shutil.rmtree(old)
    return result


@torch.no_grad()
def validate_side_rates(codec,decoder,cfg,step,out,full=False,metric=None,frequency=None,split='val',progress_dir=None):
    """Same fixed split/rates; evaluate the complete side-conditioned model."""
    from PIL import Image,ImageDraw
    from .models import JointCodec
    dataset=Images(resolved(cfg['manifest_dir']),split,cfg['seed'])
    count=len(dataset) if full else min(cfg['validation_images'],len(dataset))
    rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    if metric is None:metric=make_perceptual(cfg,'2C')
    if frequency is None:frequency=make_frequency(cfg,'2C')
    joint=JointCodec(codec,decoder).eval();rates=codec.quantizer.num_rates
    side_quantized=decoder.side_branch.quantizer is not None
    side_entropy=[SymbolEntropy(channels=cfg['side_branch']['channels']) for _ in range(rates)]
    stages=['base','conditioned','final'];keys=['psnr','lpips','dists','mse','l1','frequency','vae_consistency']
    sums=[{branch:{k:0. for k in keys} for branch in stages} for _ in range(rates)]
    align=[0.]*rates;residual=[0.]*rates;entropy=[SymbolEntropy() for _ in range(rates)]
    images=out/'validation'/f'step{step:06d}';images.mkdir(parents=True,exist_ok=True)
    def pil(t):return Image.fromarray(((t[0].clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).cpu().numpy())
    per_image=[]
    for i in range(rank,count,world):
        write_progress(progress_dir or out,'validation',step,image=i,count=count,split=split)
        x=dataset[i][None].cuda();ref=decoder.reference(x);tiles=[]
        with torch.autocast('cuda',dtype=torch.bfloat16):z=codec.encode(x)
        fx=decoder.full_vae(x.float()).detach()
        for rate in range(rates):
            with torch.autocast('cuda',dtype=torch.bfloat16):d=joint(x,rate,True,encoded=z,target_features=fx)
            entropy[rate].update(d['symbols'])
            if side_quantized:side_entropy[rate].update(d['side_symbols'])
            align[rate]+=F.mse_loss(d['zhat'].float(),ref).item()
            residual[rate]+=d['residual'].float().square().mean().sqrt().item()
            records={}
            for branch,pred in [('base',d['base']),('conditioned',d['conditioned']),('final',d['prediction'])]:
                pred=pred.float()
                latent=d['fbase'] if branch=='base' else decoder.full_vae(pred)
                mse=F.mse_loss(pred,x).item()
                row=dict(psnr=-10*math.log10(max(mse/4,1e-12)),mse=mse,lpips=metric(pred,x).mean().item(),
                         dists=metric.dists_loss(pred,x).mean().item(),l1=F.l1_loss(pred,x).item(),frequency=frequency(pred,x).item() if frequency is not None else 0.,
                         vae_consistency=F.mse_loss(latent.float(),d['fx'].float()).item())
                records[branch]=row
                for k,v in row.items():sums[rate][branch][k]+=v
            if i<cfg['validation_visuals']:
                # A fixed 4x visual gain, zero residual = gray; not a reconstruction.
                base_label='Base (current encoder + LoRA)' if decoder.joint_finetune else 'Base (frozen Stage 2B)'
                pictures=[(x,'Original'),(d['base'],base_label),(d['conditioned'],'AdaLN conditioned'),
                          ((4*d['residual']).clamp(-1,1),'Signed RGB residual x4 (gray=0)'),(d['prediction'],'Final')]
                for tensor,label in pictures:tiles.append((pil(tensor),f'Rate {rate}: {label}'))
                pil(d['prediction']).save(images/f'val{i:04d}_rate{rate}_final.png')
                per_image.append(dict(index=i,rate=rate,**records))
            del d
        if i<cfg['validation_visuals']:
            canvas=Image.new('RGB',(5*512,rates*542),'white');draw=ImageDraw.Draw(canvas)
            for j,(pic,label) in enumerate(tiles):
                left=(j%5)*512;top=(j//5)*542
                canvas.paste(pic,(left,top+30));draw.text((left+6,top+8),label,fill='black')
            canvas.save(images/f'val{i:04d}.jpg',quality=92)
    if world>1:
        v=torch.tensor([sums[r][b][k] for r in range(rates) for b in stages for k in keys]+align+residual,device='cuda',dtype=torch.float64)
        dist.all_reduce(v);vs=iter(v.tolist())
        sums=[{b:{k:next(vs) for k in keys} for b in stages} for _ in range(rates)]
        align=[next(vs) for _ in range(rates)];residual=[next(vs) for _ in range(rates)]
        gathered=[None]*world;dist.all_gather_object(gathered,entropy)
        entropy=[SymbolEntropy() for _ in range(rates)]
        for group in gathered:
            for a,b in zip(entropy,group):a.merge(b)
        if side_quantized:
            gathered=[None]*world;dist.all_gather_object(gathered,side_entropy)
            side_entropy=[SymbolEntropy(channels=cfg['side_branch']['channels']) for _ in range(rates)]
            for group in gathered:
                for a,b in zip(side_entropy,group):a.merge(b)
        gathered_rows=[None]*world;dist.all_gather_object(gathered_rows,per_image)
        per_image=[r for part in gathered_rows for r in part]
    reports=[]
    for r in range(rates):
        stats=entropy[r].result();bpp=stats.pop('estimated_bpp')
        reports.append(dict(rate=r,target_main_bpp=cfg['rate_targets'][r],main_estimated_bpp=bpp,main_entropy=stats,
            alignment=align[r]/count,residual_rms=residual[r]/count,**{b:{k:v/count for k,v in sums[r][b].items()} for b in stages}))
        if side_quantized:
            side_stats=side_entropy[r].result();side_bpp=side_stats.pop('estimated_bpp')
            reports[-1].update(target_side_bpp=cfg['side_rate_targets'][r],side_estimated_bpp=side_bpp,
                               side_entropy=side_stats,total_estimated_bpp=bpp+side_bpp)
    def objective(m):return cfg['lambda_mse']*m['mse']+cfg['lambda_lpips']*m['lpips']+cfg['lambda_frequency']*m['frequency']+cfg['lambda_dists']*m['dists']+cfg['lambda_vae']*m['vae_consistency']
    result=dict(stage='2C',step=step,count=count,rates=reports,selection_loss=sum(objective(r['final']) for r in reports)/rates,
        joint_finetune=decoder.joint_finetune,
        split=split,seen_in_training=split=='fit',frequency_computed=frequency is not None,
        selection_objective={k:cfg[k] for k in ['lambda_mse','lambda_lpips','lambda_frequency','lambda_dists','lambda_vae']},
        psnr_protocol='unrounded float RGB [0,1]',mse_range='[-1,1]',side_latent_shape=[8,16,16],side_quantized=side_quantized,side_rate_accounted=side_quantized,
        rate_note=('Main + side per-channel empirical entropy; excludes entropy coder and headers. No actual bitstream bpp.' if side_quantized else
                   'Only main stream zero-order empirical entropy; continuous side latent is NOT included. No total or actual bitstream bpp.'),
        side_parameter_counts=decoder.side_branch.parameter_counts())
    if rank==0:
        (out/f'validation_{step:06d}.json').write_text(json.dumps(result,indent=2)+'\n')
        (images/'metrics.json').write_text(json.dumps(per_image,indent=2)+'\n')
        for old in sorted((out/'validation').glob('step*'))[:-3]:shutil.rmtree(old)
    joint.train();del metric,frequency,joint;gc.collect();torch.cuda.empty_cache()
    return result
