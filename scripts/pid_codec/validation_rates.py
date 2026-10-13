"""All-rate validation integrated with the existing training/checkpoint loop."""
import gc,json,shutil,contextlib,math
import torch
import torch.distributed as dist
import torch.nn.functional as F
from .data import Images
from .train import resolved,make_frequency,make_perceptual
from .quantization import SymbolEntropy

@torch.no_grad()
def validate_rates(codec,decoder,cfg,step,out,full=False,stage="2A"):
    import lpips
    from PIL import Image,ImageDraw
    dataset=Images(resolved(cfg['manifest_dir']),'val',cfg['seed'])
    count=len(dataset) if full else min(cfg['validation_images'],len(dataset))
    rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    metric=make_perceptual(cfg,stage)
    frequency=make_frequency(cfg,'2A');codec.eval();rates=codec.quantizer.num_rates
    sums=[dict(alignment=0.,psnr=0.,lpips=0.,l1=0.,frequency=0.) for _ in range(rates)]
    if stage=='2B':
        for row in sums:row.update(mse=0.,dists=0.)
    baseline=out/'baseline'
    if stage=='2B' and step==0:baseline.mkdir(parents=True,exist_ok=True)
    entropy=[SymbolEntropy() for _ in range(rates)]
    images=out/'validation'/f'step{step:06d}';images.mkdir(parents=True,exist_ok=True)
    def pil(t):return Image.fromarray(((t[0].clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).cpu().numpy())
    for i in range(rank,count,world):
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
                      lpips=metric(pred,x).mean().item(),l1=F.l1_loss(pred,x).item(),frequency=frequency(pred,x).item())
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
