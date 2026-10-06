"""Isolated OOM-safe real training trials; select measured batch, never a preset."""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch

from .models import ROOT
from .train import load_config,resolved,training_identity
from .data import fingerprint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['A','B'],required=True)
    parser.add_argument('--config',default='configs/pid_codec/stage1.json')
    args=parser.parse_args();cfg=load_config(args.config)
    dest=resolved(cfg['output_dir'])/'profiles'/args.stage;dest.mkdir(parents=True,exist_ok=True)
    count=torch.cuda.device_count()
    if not count:raise RuntimeError('CUDA GPU required')
    hardware=[]
    for i in range(count):
        with torch.cuda.device(i):
            free,total=torch.cuda.mem_get_info()
        hardware.append(dict(index=i,name=torch.cuda.get_device_name(i),free=free,total=total))
        if free<total*0.96:raise RuntimeError(f'GPU {i} is occupied; wait for its job before 90–95% profiling')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if os.environ.get('CUDA_VISIBLE_DEVICES') else [str(i) for i in range(count)]
    trials=[];selections=[]
    def run(device,batch,workers):
        label=f'gpu{device}_batch{batch}_workers{workers}'
        result=dest/(label+'.json');log=dest/(label+'.log')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=visible[device])
        command=[sys.executable,'-u','-m','scripts.pid_codec.train','--config',str(resolved(args.config)),
                 '--stage',args.stage,'--trial','--profile-world-size',str(count),'--micro-batch',str(batch),'--workers',str(workers),'--result',str(result)]
        with log.open('w') as f:
            completed=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        if not result.exists():raise RuntimeError(f'Trial failed before metrics; inspect {log}')
        row=json.loads(result.read_text());trials.append(row)
        print(json.dumps(row),flush=True)
        if completed.returncode and not row.get('oom'):raise RuntimeError(f'Non-OOM error; inspect {log}')
        if not row.get('ok') and not row.get('oom'):raise RuntimeError(f'Non-OOM error; inspect {log}')
        return row
    for device in range(count):
        low=0;high=None;batch=1;best=None
        while True:
            row=run(device,batch,cfg['workers'])
            good=row.get('ok') and row['device_used_peak']<=hardware[device]['total']*0.95
            if good:
                low=batch;best=row
                if row['device_used_peak']>=hardware[device]['total']*cfg['profile_memory_fraction']:
                    # This meets the target; test the next size before declaring the maximum.
                    high=batch+max(1,batch//8)
                    upper=run(device,high,cfg['workers'])
                    if upper.get('ok') and upper['device_used_peak']<=hardware[device]['total']*.95:
                        low=high;best=upper;batch=high+max(1,high//8);high=None;continue
                    break
                batch*=2
                if batch>1024:raise RuntimeError('Unexpectedly large feasible batch; inspect profile rather than silently cap')
            else:
                high=batch;break
        if low==0:raise RuntimeError('Even microbatch1 failed; use configured activation checkpointing / inspect logs')
        while high-low>1:
            batch=(low+high)//2;row=run(device,batch,cfg['workers'])
            if row.get('ok') and row['device_used_peak']<=hardware[device]['total']*.95:low=batch;best=row
            else:high=batch
        selections.append(best)
    micro=min(r['micro_batch'] for r in selections)
    # Time the actual loader at the selected batch; do not infer throughput from synthetic tensors.
    variants=[run(0,micro,w) for w in sorted(set([2,4,cfg['workers']]))]
    valid=[r for r in variants if r.get('ok') and r['device_used_peak']<=hardware[0]['total']*.95]
    if not valid:raise RuntimeError('Final data-loader sweep failed')
    fastest=max(valid,key=lambda r:r['images_per_second'])
    target=cfg['stage_a_effective_batch_target'] if args.stage=='A' else cfg['stage_b_effective_batch_target']
    accumulation=max(1,math.ceil(target/(micro*count)))
    output=dict(stage=args.stage,world_size=count,micro_batch=micro,gradient_accumulation=accumulation,
                effective_batch=micro*count*accumulation,workers=fastest['workers'],hardware=hardware,
                measured=fastest,per_gpu_selected=selections,target_memory_fraction=cfg['profile_memory_fraction'],
                all_trials=trials,amp='bfloat16',checkpointing=dict(encoder=cfg['encoder_activation_checkpoint'],pid=cfg['pid_activation_checkpoint']))
    output['identity']=training_identity(cfg,args.stage,fingerprint(resolved(cfg['manifest_dir'])))
    (dest/'selected.json').write_text(json.dumps(output,indent=2)+'\n')
    print('Selected profile:',dest/'selected.json',flush=True)


if __name__=='__main__':main()
