"""Official DC-AE encoder and differentiable official PiD four-step sampling."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

ROOT=Path(__file__).resolve().parents[2]


class CompactEncoder(nn.Module):
    def __init__(self, dcae_dir, activation_checkpoint=False, diagnostic_random=False):
        super().__init__()
        from diffusers.models.autoencoders.autoencoder_dc import Encoder
        from safetensors import safe_open
        directory=Path(dcae_dir)
        config_path=directory/'config.json'
        if diagnostic_random:
            config_path=ROOT/'configs/dcae_architecture.json'
        cfg=json.loads(config_path.read_text())
        assert cfg['latent_channels']==32
        kwargs=dict(in_channels=cfg['in_channels'],latent_channels=32,
                    attention_head_dim=cfg['attention_head_dim'],block_type=tuple(cfg['encoder_block_types']),
                    block_out_channels=tuple(cfg['encoder_block_out_channels']),
                    layers_per_block=tuple(cfg['encoder_layers_per_block']),
                    qkv_multiscales=tuple(tuple(x) for x in cfg['encoder_qkv_multiscales']),
                    downsample_block_type=cfg['downsample_block_type'])
        if diagnostic_random:
            self.encoder=Encoder(**kwargs)
        else:
            if not (directory/'verified.json').is_file():
                raise FileNotFoundError('Verified DC-AE weights required; run prepare.py download. No fallback encoder.')
            with torch.device('meta'):
                self.encoder=Encoder(**kwargs)
            with safe_open(directory/'diffusion_pytorch_model.safetensors',framework='pt',device='cpu') as f:
                state={k.removeprefix('encoder.'):f.get_tensor(k) for k in f.keys() if k.startswith('encoder.')}
            self.encoder.load_state_dict(state,strict=True,assign=True)
            self.encoder.float()
        # Raw DC-AE encoder features; diffusion scaling_factor belongs to its
        # original diffusion pipeline, not this learned projection/condition space.
        self.projection=nn.Conv2d(32,16,1)
        if activation_checkpoint:
            for block in self.encoder.down_blocks:
                original=block.forward
                def wrapped(x, fn=original):
                    if torch.is_grad_enabled():
                        return checkpoint(fn,x,use_reentrant=False)
                    return fn(x)
                block.forward=wrapped

    def forward(self,x):
        features=self.encoder(x)
        if features.shape[1:]!=(32,16,16):
            raise ValueError(f'Expected [B,32,16,16], got {features.shape}')
        return self.projection(features)


class ReferenceVAE:
    def __init__(self):
        from pid._src.tokenizers.qwenimage_vae import QwenImageVAEInterface
        self.vae=QwenImageVAEInterface()

    @torch.no_grad()
    def __call__(self,x):
        from pid._src.degradation import simple_downsample_image
        # Exactly the training degradation, not the earlier LANCZOS experiment.
        low=simple_downsample_image(x.float(),4)
        z=self.vae.encode(low.unsqueeze(2))
        return z[:,:,0].float() if z.ndim==5 else z.float()


class FrozenPiD(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        from pid._src.inference.decoder import load_our_decoder
        settings=json.loads((ROOT/cfg['pid_args']).read_text())
        # Official loading changes global cuDNN/TF32 flags. Keep the trainer's
        # choices so validation does not silently change later optimizer updates.
        flags=(torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,
               torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32)
        try:
            self.model=load_our_decoder(SimpleNamespace(**settings),['model.config.ema.enabled=False'],True)
        finally:
            (torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,
             torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32)=flags
        self.model.eval().requires_grad_(False)
        self.activation_checkpoint=cfg['pid_activation_checkpoint']
        with torch.no_grad():
            captions,_=self.model._encode_text_raw([cfg['caption']])
        self.register_buffer('captions',captions.detach().to(**self.model.tensor_kwargs),persistent=False)
        self.register_buffer('schedule',self.model._get_t_list(torch.device('cuda'),num_steps=4),persistent=False)
        assert self.model.config.student_sample_steps==4
        # Generate the same noise sequence as official inference at B=1, then
        # broadcast it so noise is independent of batch/accumulation. BF16 kernels
        # may still produce different rounding for different batch geometries.
        g=torch.Generator(device='cuda').manual_seed(cfg['decoder_seed'])
        self.register_buffer('initial_noise',torch.randn(1,3,512,512,device='cuda',generator=g),persistent=False)
        self.register_buffer('step_noises',torch.stack([torch.randn(1,3,512,512,device='cuda',generator=g) for _ in range(3)]),persistent=False)

    @torch.no_grad()
    def reference(self,x):
        from pid._src.degradation import simple_downsample_image
        return self.model.encode_lq_latent(simple_downsample_image(x.float(),4)).float()

    def forward(self,z):
        b=z.shape[0]
        def network(x,t,c,**kwargs):
            if self.activation_checkpoint and torch.is_grad_enabled():
                return checkpoint(self.model.net,x,t,c,use_reentrant=False,**kwargs)
            return self.model.net(x,t,c,**kwargs)
        # _student_sample_loop is the official sampler and is NOT decorated
        # no_grad. The public generate_samples_from_batch entry point is.
        y=self.model._student_sample_loop(
            self.initial_noise.expand(b,-1,-1,-1),self.schedule,
            self.captions.expand(b,-1,-1),z.to(**self.model.tensor_kwargs),
            torch.zeros(b,device=z.device),net=network,step_noises=self.step_noises)
        # Match the official public inference result, including its clamp.
        return y.clamp(-1,1)


def load_models(cfg,stage,diagnostic_random=False):
    os.chdir(ROOT/'third_party/PiD')
    codec=CompactEncoder(ROOT/cfg['dcae_dir'],cfg['encoder_activation_checkpoint'],diagnostic_random).cuda()
    # Stage A should not keep a 1.4B decoder resident during encoder-only updates.
    if stage=='A': return codec,ReferenceVAE(),None
    decoder=FrozenPiD(cfg)
    return codec,decoder.reference,decoder
