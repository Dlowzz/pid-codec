"""Read-only wall-clock training health monitor; no GPU framework allocation."""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import statistics
import subprocess
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=1200)
    p.add_argument('--session',default='pid-codec-stage2b')
    args=p.parse_args();out=args.run.parent/'monitor_20min';out.mkdir(parents=True,exist_ok=True)
    log=args.run/'train.jsonl';wait_started=time.monotonic()
    while not log.exists() or not log.stat().st_size:
        if time.monotonic()-wait_started>600:raise RuntimeError('No first optimizer step within 10 minutes')
        time.sleep(5)
    started=time.monotonic();utc=datetime.now(timezone.utc).isoformat()
    start_step=None;last_step=None;last_change=started;observations=[];rows=[]
    try:
        while True:
            lines=log.read_text().splitlines();rows=[]
            for i,line in enumerate(lines):
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if i!=len(lines)-1:raise
            if not rows:raise RuntimeError('No readable training rows')
            current=rows[-1];now=time.monotonic()
            if start_step is None:start_step=current['step']
            selected=[r for r in rows if r['step']>=start_step]
            for a,b in zip(selected,selected[1:]):
                assert b['step']==a['step']+1,'Non-contiguous optimizer steps'
                assert b['sample_cursor']-a['sample_cursor']==b['effective_batch'],'Incorrect data cursor'
            for row in selected:
                for key in ['loss','mse','lpips','frequency','dists','alignment','gradient_norm','images_per_second']:
                    assert math.isfinite(row[key]),f'Nonfinite {key}'
                assert row['gradient_norm']>0 and row['images_per_second']>0
                assert math.isclose(row['group_lrs']['pid_lora'],2*row['group_lrs']['encoder'],rel_tol=1e-10)
                weighted=sum(row[k] for k in ['weighted_mse','weighted_lpips','weighted_frequency','weighted_dists','weighted_alignment'])
                assert abs(row['loss']-weighted)<1e-5,'Loss weights do not sum to total'
                assert row['loss']<max(10.,selected[0]['loss']*10),'Loss explosion'
            if current['step']!=last_step:last_change=now;last_step=current['step']
            if now-last_change>180:raise RuntimeError('Optimizer step stalled for over 3 minutes')
            if subprocess.run(['tmux','has-session','-t',args.session],capture_output=True).returncode:
                raise RuntimeError('Training tmux session exited')
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
            observation=dict(elapsed_seconds=now-started,step=current['step'],loss=current['loss'],
                             images_per_second=current['images_per_second'],gpu=gpu)
            observations.append(observation)
            with (out/'samples.jsonl').open('a') as f:f.write(json.dumps(observation)+'\n')
            print(json.dumps(observation),flush=True)
            if now-started>=args.seconds:break
            time.sleep(min(30,args.seconds-(now-started)))
        report=dict(passed=True,started_utc=utc,elapsed_seconds=time.monotonic()-started,
                    start_step=start_step,end_step=last_step,optimizer_steps_observed=last_step-start_step,
                    finite_losses_and_gradients=True,loss_formula_verified=True,data_cursor_verified=True,
                    lr_groups_verified=True,median_images_per_second=statistics.median(r['images_per_second'] for r in selected),
                    maximum_gradient_norm=max(r['gradient_norm'] for r in selected),
                    maximum_peak_allocated_gib=max(r['peak_allocated_gib'] for r in selected),
                    maximum_device_used_gib=max(r['device_used_gib'] for r in selected),
                    loss_first=selected[0]['loss'],loss_last=selected[-1]['loss'],observations=observations)
    except BaseException as exc:
        report=dict(passed=False,started_utc=utc,elapsed_seconds=time.monotonic()-started,error=repr(exc),
                    start_step=start_step,last_step=last_step,observations=observations)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        raise
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('20-minute training monitor PASSED',flush=True)

if __name__=='__main__':main()
