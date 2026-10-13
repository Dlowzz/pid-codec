"""CPU-only snapshots of existing training logs; never touches training state."""
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path(__file__).resolve().parents[2]/'runs/pid_codec/stage2B/curves'
out.mkdir(parents=True,exist_ok=True)
run=out.parent/'stage_2B'
lines=(run/'train.jsonl').read_text().splitlines()
rows=[]
for i,line in enumerate(lines):
    try: rows.append(json.loads(line))
    except json.JSONDecodeError:
        if i!=len(lines)-1: raise
vals=[json.loads(p.read_text()) for p in sorted(run.glob('validation_*.json'))]
x=np.array([r['step'] for r in rows]);window=100
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})

def smooth(y):
    return np.convolve(y,np.ones(window)/window,mode='valid')
def training(ax,key,label):
    y=np.array([r[key] for r in rows])
    ax.plot(x,y,alpha=.18,lw=.5,color='tab:blue',label='Per optimizer step')
    ax.plot(x[window-1:],smooth(y),lw=1.7,color='tab:blue',label='Trailing 100-step mean')
    ax.set(title=label,xlabel='Optimizer step',ylabel='Loss')

a,axs=plt.subplots(2,3,figsize=(15,8))
for ax,key,title in zip(axs.flat,['loss','mse','lpips','frequency','dists','alignment'],['Total training loss','Pixel MSE [-1,1], weight 1','LPIPS VGG, weight 1','JPEG frequency loss, weight 1','Raw DISTS, weight 0.2','Raw latent alignment MSE, decaying weight']):
    training(ax,key,title)
axs[0,0].legend(fontsize=8)
a.suptitle(f'Stage 2B training loss — through step {x[-1]} (mixed images and rates)')
a.tight_layout();a.savefig(out/'loss_curves.png',dpi=160);plt.close(a)

b,axs=plt.subplots(2,3,figsize=(15,8));g=np.array([r['gradient_norm'] for r in rows])
axs[0,0].plot(x,g,lw=.55,alpha=.6);axs[0,0].axhline(1,color='red',ls='--',label='Clip threshold 1.0');axs[0,0].set(yscale='log',title='Joint gradient norm BEFORE clipping',ylabel='L2 norm');axs[0,0].legend(fontsize=8)
axs[0,1].plot(x[window-1:],smooth((g>1).astype(float))*100);axs[0,1].set(title='Clipping frequency (trailing 100 steps)',ylabel='% of optimizer steps',ylim=(0,100))
keys=['weighted_mse','weighted_lpips','weighted_frequency','weighted_dists','weighted_alignment']
y=np.array([smooth(np.array([r[k] for r in rows])) for k in keys]);shares=y/y.sum(axis=0)*100
axs[0,2].stackplot(x[window-1:],shares,labels=['MSE','LPIPS','JPEG','0.2 DISTS','Alignment'],alpha=.85);axs[0,2].set(title='Weighted loss shares (not gradient shares)',ylabel='% of total loss',ylim=(0,100));axs[0,2].legend(fontsize=7,loc='lower right')
for name in ['encoder','pid_lora']:axs[1,0].plot(x,[r['group_lrs'][name] for r in rows],label=name)
axs[1,0].set(title='Optimizer group learning rates',ylabel='Learning rate');axs[1,0].legend(fontsize=8);axs[1,0].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
for key,label in [('device_used_gib','Device used'),('peak_allocated_gib','Peak allocated')]:axs[1,1].plot(x,[r[key] for r in rows],label=label)
axs[1,1].set(title='GPU memory logged at each step',ylabel='GiB');axs[1,1].legend(fontsize=8)
axs[1,2].plot(x,[r['images_per_second'] for r in rows],alpha=.6,lw=.7);axs[1,2].set(title='Training throughput',ylabel='Images / second')
for ax in axs.flat:ax.set_xlabel('Optimizer step')
b.suptitle(f'Stage 2B gradients and runtime — through step {x[-1]}')
b.tight_layout();b.savefig(out/'gradient_runtime_curves.png',dpi=160);plt.close(b)

c,axs=plt.subplots(2,3,figsize=(15,8));vx=[v['step'] for v in vals]
for ax,key,label in zip(axs.flat,['psnr','lpips','dists','mse','estimated_bpp','alignment'],['PSNR (dB), higher is better','LPIPS VGG, lower is better','DISTS, lower is better','Pixel MSE [-1,1], lower is better','Estimated empirical bpp (NO bitstream)','Latent alignment MSE']):
    for rate in range(6):
        ys=[v['rates'][rate][key] for v in vals]
        ax.plot(vx,ys,'-o',ms=4,label=f'Rate {rate} (target {vals[-1]["rates"][rate]["target_bpp"]:g})')
    ax.set(title=label,xlabel='Optimizer step');ax.set_xticks(vx)
axs[0,0].legend(fontsize=7)
c.suptitle(f'Stage 2B fixed 32-image validation — measured steps: {", ".join(map(str,vx))}\nLines connect measured checkpoints only; no per-step validation metrics')
c.tight_layout();c.savefig(out/'validation_curves.png',dpi=160);plt.close(c)

(out/'snapshot.json').write_text(json.dumps(dict(training_last_step=int(x[-1]),training_rows=len(rows),validation_steps=vx,validation_images=vals[-1]['count'],smoothing_window=window,generated_at=datetime.now().isoformat(),training_unchanged=True),indent=2)+'\n')
print((out/'snapshot.json').read_text())
