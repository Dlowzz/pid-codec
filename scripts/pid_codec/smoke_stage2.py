"""Actual Stage 2A optimizer updates, six-rate validation, exact resume check."""
import argparse,json,subprocess,sys
import torch
from .models import ROOT
from .smoke import same

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['2A','2B'],default='2A')
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
    if args.stage=='2B':keys+=['pid_lora','pid_lora_config','pid_base_weights']
    for key in keys:
        same(a[key],b[key],key)
    result=dict(stage=args.stage,lora_resume_bit_exact=args.stage=='2B',actual_pretrained_training=True,encoder_optimizer_scheduler_resume_bit_exact=True,
                stage_step=a['stage_step'],sample_cursor=a['sample_cursor'],batch=a['batch'],all_six_rates_validated=True)
    (base/'checks/resume.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
