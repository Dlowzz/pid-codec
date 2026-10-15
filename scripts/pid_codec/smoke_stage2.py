"""Actual Stage 2 optimizer updates, all-rate validation, exact resume check."""
import argparse,json,subprocess,sys
import torch
from .models import ROOT
from .smoke import same

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['2A','2B','2C'],default='2A')
    parser.add_argument('--config')
    parser.add_argument('--compare-only',action='store_true')
    args=parser.parse_args()
    from .train import load_config,resolved
    config=args.config or f'configs/pid_codec/stage{args.stage.lower()}.json'
    cfg=load_config(config)
    base=resolved(cfg['output_dir'])
    profile=base/f'profiles/{args.stage}/selected.json';world=json.loads(profile.read_text())['world_size']
    resumed=base/'smoke_resumed';continuous=base/'smoke_continuous'
    if not args.compare_only and (resumed.exists() or continuous.exists()):
        raise RuntimeError('Inspect existing smoke outputs before rerunning; --compare-only rechecks completed outputs')
    prefix=[sys.executable,'-u','-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',
            '--module','scripts.pid_codec.train','--stage',args.stage,'--config',str(resolved(config)),
            '--profile',str(profile),'--debug','--validation-images','2']
    jobs=[] if args.compare_only else [('initial',resumed,['--max-steps','2']),('resume',resumed,['--resume','auto','--max-steps','4']),
                            ('continuous',continuous,['--max-steps','4'])]
    for label,out,extra_args in jobs:
        log=base/f'smoke_{label}.log';print('Starting',label,log,flush=True)
        with log.open('w') as f:subprocess.run(prefix+['--output',str(out)]+extra_args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    a=torch.load(resumed/'latest.pt',map_location='cpu',weights_only=False,mmap=True)
    b=torch.load(continuous/'latest.pt',map_location='cpu',weights_only=False,mmap=True)
    keys=['encoder','optimizer','scheduler','stage_step','sample_cursor','identity','batch','global_optimizer_step','rng_by_rank']
    if args.stage in ('2B','2C'):keys+=['pid_lora','pid_lora_config','pid_base_weights']
    if args.stage=='2C':keys+=['side_branch','side_config']
    if cfg.get('gan'):keys+=['discriminator','discriminator_optimizer','discriminator_scheduler']
    for key in keys:
        same(a[key],b[key],key)
    result=dict(stage=args.stage,lora_resume_bit_exact=args.stage in ('2B','2C'),side_resume_bit_exact=args.stage=='2C',actual_pretrained_training=True,encoder_optimizer_scheduler_resume_bit_exact=True,
                stage_step=a['stage_step'],sample_cursor=a['sample_cursor'],batch=a['batch'],all_rates_validated=True,rate_count=len(cfg['rate_targets']))
    if cfg.get('gan'):result['discriminator_optimizer_scheduler_resume_bit_exact']=True
    if args.stage=='2C':
        quantized='quantizer.deltas' in a['side_branch']
        result.update(side_quantized=quantized,side_quantizer_resume_bit_exact=quantized,
                      joint_finetune=cfg.get('stage_2c_joint_finetune',False))
        relative=f"validation/step{a['stage_step']:06d}"
        images=sorted((resumed/relative).glob('*_final.png'))
        assert len(images)==2*len(cfg['rate_targets']),'Missing reconstruction outputs for configured rates'
        for image in images:
            assert image.read_bytes()==(continuous/relative/image.name).read_bytes(),f'Resumed reconstruction differs: {image.name}'
        same(json.loads((resumed/relative/'metrics.json').read_text()),
             json.loads((continuous/relative/'metrics.json').read_text()),'validation_metrics')
        rates_a=json.loads((resumed/f"validation_{a['stage_step']:06d}.json").read_text())
        rates_b=json.loads((continuous/f"validation_{b['stage_step']:06d}.json").read_text())
        same(rates_a,rates_b,'all_rate_metrics_and_entropy')
        if quantized:
            assert rates_a['side_rate_accounted'] and rates_a['side_quantized']
            for row in rates_a['rates']:
                assert row['total_estimated_bpp']==row['main_estimated_bpp']+row['side_estimated_bpp']
        result.update(reconstruction_resume_bit_exact=True,reconstruction_images_compared=len(images))
    (base/'checks/resume.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
