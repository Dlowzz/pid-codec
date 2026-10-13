"""Real four-step A/B updates and uninterrupted versus resumed state comparison."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from .models import ROOT


def same(left,right,path='state'):
    if torch.is_tensor(left):
        if not torch.equal(left,right):
            raise AssertionError(f'{path}: tensor differs (max abs {(left.float()-right.float()).abs().max().item()})')
    elif isinstance(left,np.ndarray):
        assert np.array_equal(left,right),path
    elif isinstance(left,dict):
        assert left.keys()==right.keys(),path
        for key in left:same(left[key],right[key],f'{path}.{key}')
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right),path
        for i,(a,b) in enumerate(zip(left,right)):same(a,b,f'{path}.{i}')
    else:assert left==right,path


def main():
    reports={}
    for stage in ['A','B']:
        profile=ROOT/f'runs/pid_codec/stage1/profiles/{stage}/selected.json'
        world=json.loads(profile.read_text())['world_size']
        resumed=ROOT/f'runs/pid_codec/debug_{stage}'
        uninterrupted=ROOT/f'runs/pid_codec/debug_{stage}_continuous'
        if resumed.exists() or uninterrupted.exists():
            raise RuntimeError('Debug destination already exists; inspect it before rerunning smoke checks')
        init=[] if stage=='A' else ['--init-from',str(ROOT/'runs/pid_codec/debug_A/latest.pt')]
        prefix=[sys.executable,'-u','-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',
                '--module','scripts.pid_codec.train','--stage',stage,'--profile',str(profile),
                '--debug','--validation-images','2']
        for label,output,extra in [('initial',resumed,init+['--max-steps','2']),
                                    ('resume',resumed,['--resume','auto','--max-steps','4']),
                                    ('continuous',uninterrupted,init+['--max-steps','4'])]:
            log=ROOT/f'logs/pid_codec_smoke_{stage}_{label}.log'
            print(f'Start {stage} {label}: {log}',flush=True)
            with log.open('w') as f:
                subprocess.run(prefix+['--output',str(output)]+extra,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        a=torch.load(resumed/'latest.pt',map_location='cpu',weights_only=False)
        b=torch.load(uninterrupted/'latest.pt',map_location='cpu',weights_only=False)
        for key in ['encoder','optimizer','scheduler','stage_step','sample_cursor','identity','batch']:
            same(a[key],b[key],key)
        reports[stage]={'actual_pretrained_training':True,'encoder_optimizer_scheduler_resume_bit_exact':True,
                        'stage_step':a['stage_step'],'sample_cursor':a['sample_cursor'],'batch':a['batch']}
        del a,b
        (ROOT/'runs/pid_codec/checks/short_training_resume.json').write_text(json.dumps(reports,indent=2)+'\n')
        print(json.dumps(reports[stage]),flush=True)


if __name__=='__main__':main()
