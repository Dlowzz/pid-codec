"""Official DC-AE encoder and differentiable official PiD four-step sampling."""
import json
import hashlib
import contextlib
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

ROOT=Path(__file__).resolve().parents[2]


class CompactEncoder(nn.Module):
    def __init__(self, dcae_dir, activation_checkpoint=False, diagnostic_random=False, saved_encoder=None):
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
        if saved_encoder is not None:
            with torch.device('meta'):
                self.encoder=Encoder(**kwargs)
            state={k.removeprefix('encoder.'):v for k,v in saved_encoder.items() if k.startswith('encoder.')}
            self.encoder.load_state_dict(state,strict=True,assign=True)
            self.encoder.float()
        elif diagnostic_random:
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
        if saved_encoder is not None:
            self.projection.load_state_dict({k.removeprefix('projection.'):v for k,v in saved_encoder.items() if k.startswith('projection.')},strict=True)
        self.quantizer=None
        if activation_checkpoint:
            for block in self.encoder.down_blocks:
                original=block.forward
                def wrapped(x, fn=original):
                    if torch.is_grad_enabled():
                        return checkpoint(fn,x,use_reentrant=False)
                    return fn(x)
                block.forward=wrapped

    def encode(self,x):
        features=self.encoder(x)
        if features.shape[1:]!=(32,16,16):
            raise ValueError(f'Expected [B,32,16,16], got {features.shape}')
        return self.projection(features)

    def forward(self,x,rates=None,return_details=False):
        z=self.encode(x)
        if self.quantizer is None:return z
        zhat,symbols,rates=self.quantizer(z,rates,return_symbols=True)
        return (zhat,z,symbols,rates) if return_details else zhat


class ReferenceVAE:
    def __init__(self):
        from pid._src.tokenizers.qwenimage_vae import QwenImageVAEInterface
        with contextlib.chdir(ROOT/'third_party/PiD'):
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
            with contextlib.chdir(ROOT/'third_party/PiD'):
                self.model=load_our_decoder(SimpleNamespace(**settings),['model.config.ema.enabled=False'],True)
        finally:
            (torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,
             torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32)=flags
        self.model.eval().requires_grad_(False)
        self.lora_config = cfg.get('pid_lora')
        if self.lora_config:
            self._inject_lora()
        self.activation_checkpoint=cfg['pid_activation_checkpoint']
        self.side_branch=None
        self.side_cfg=cfg.get('side_branch')
        self.joint_finetune=cfg.get('stage_2c_joint_finetune',False)
        self.loss_weights={k:cfg.get(k) for k in ['lambda_mse','lambda_lpips','lambda_frequency','lambda_dists','lambda_vae']}
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

    def _inject_lora(self):
        # Load the unmodified checkpoint BEFORE PEFT renames Linear weights.
        from peft import LoraConfig, inject_adapter_in_model
        from pid._src.models.pid_model import PixelDiTSRLoraConfig
        official = PixelDiTSRLoraConfig()
        cfg = self.lora_config
        self.adapter_name = 'codec'
        targets = list(official.lora_target_modules)
        inject_adapter_in_model(LoraConfig(
            r=cfg['rank'], lora_alpha=cfg['alpha'], lora_dropout=0.,
            target_modules=targets, bias='none', init_lora_weights=True),
            self.model.net, adapter_name=self.adapter_name)
        for name, param in self.model.named_parameters():
            trainable = '.lora_A.' in name or '.lora_B.' in name
            param.requires_grad_(trainable)
            if trainable:
                param.data = param.data.float()
        self.model.eval()
        self.lora_metadata = dict(rank=cfg['rank'], alpha=cfg['alpha'], dropout=0.,
                                  targets=targets, adapter_name=self.adapter_name)
        if not self.lora_parameters():
            raise RuntimeError('No PiD LoRA parameters matched official targets')

    def train(self, mode=True):
        # Preserve deterministic inference behavior; LoRA dropout is zero.
        super().train(mode)
        self.model.eval()
        return self

    def lora_parameters(self):
        return [p for n,p in self.model.net.named_parameters()
                if p.requires_grad and ('.lora_A.' in n or '.lora_B.' in n)]

    def train_lora(self):
        """Enable the existing adapters without unfreezing original PiD/VAE weights."""
        for name,param in self.model.named_parameters():
            param.requires_grad_('.lora_A.' in name or '.lora_B.' in name)
        if not self.lora_parameters():raise RuntimeError('Joint finetuning requires existing PiD LoRA')

    def lora_state_dict(self):
        return {n: p.detach().cpu() for n, p in self.model.net.named_parameters()
                if '.lora_A.' in n or '.lora_B.' in n}

    def load_lora_state_dict(self, state):
        params = {n: p for n, p in self.model.net.named_parameters()
                  if '.lora_A.' in n or '.lora_B.' in n}
        if params.keys() != state.keys():
            raise ValueError('PiD LoRA checkpoint keys do not match configured adapters')
        with torch.no_grad():
            for name, param in params.items():
                if param.shape != state[name].shape:
                    raise ValueError(f'LoRA shape mismatch: {name}')
                param.copy_(state[name])

    @contextlib.contextmanager
    def disable_lora(self):
        from peft.tuners.tuners_utils import BaseTunerLayer
        layers = [m for m in self.model.net.modules() if isinstance(m, BaseTunerLayer)]
        previous = [m.disable_adapters for m in layers]
        # PEFT enable_adapters(True) also enables adapter parameter gradients.
        # Preserve the caller's freeze policy (Stage 2C keeps trained LoRA frozen).
        gradients = [(p,p.requires_grad) for m in layers for p in m.parameters()]
        try:
            for m in layers: m.enable_adapters(False)
            yield
        finally:
            for m, disabled in zip(layers, previous): m.enable_adapters(not disabled)
            for p,requires_grad in gradients:p.requires_grad_(requires_grad)

    @torch.no_grad()
    def reference(self,x):
        from pid._src.degradation import simple_downsample_image
        return self.model.encode_lq_latent(simple_downsample_image(x.float(),4)).float()

    def enable_side_branch(self):
        from .side import SideBranch
        self.model.requires_grad_(False).eval()
        self.side_branch=SideBranch(self.side_cfg,self.model.net).cuda()
        self.model.vae_encoder.model.model.requires_grad_(False).eval()

    def full_vae(self,x):
        """Full 512 encoder with official normalization; input gradients retained."""
        wrapper=self.model.vae_encoder.model
        def encode(images):
            # Exactly the no-AMP Qwen wrapper computation, without its no_grad.
            with torch.autocast('cuda',enabled=False):
                return wrapper.model.encode(images.to(wrapper.dtype),wrapper.scale).to(images.dtype)
        if self.side_cfg['vae_activation_checkpoint'] and torch.is_grad_enabled() and x.requires_grad:
            return checkpoint(encode,x,use_reentrant=False)
        return encode(x)

    def forward(self,z,side=None):
        b=z.shape[0]
        def conditioned(x,t,c,s,**kwargs):
            # Scope must be INSIDE checkpoint: backward recomputes after the
            # original forward context has exited. Pass s as an explicit input.
            with self.side_branch.adaln.condition(s):
                return self.model.net(x,t,c,**kwargs)
        def network(x,t,c,**kwargs):
            fn=self.model.net if side is None else conditioned
            inputs=(x,t,c) if side is None else (x,t,c,side)
            if self.activation_checkpoint and torch.is_grad_enabled():
                return checkpoint(fn,*inputs,use_reentrant=False,**kwargs)
            return fn(*inputs,**kwargs)
        # _student_sample_loop is the official sampler and is NOT decorated
        # no_grad. The public generate_samples_from_batch entry point is.
        y=self.model._student_sample_loop(
            self.initial_noise.expand(b,-1,-1,-1),self.schedule,
            self.captions.expand(b,-1,-1),z.to(**self.model.tensor_kwargs),
            torch.zeros(b,device=z.device),net=network,step_noises=self.step_noises)
        # Match the official public inference result, including its clamp.
        return y.clamp(-1,1)


def load_models(cfg,stage,diagnostic_random=False,checkpoint_state=None):
    """A complete resume/evaluation checkpoint does not need bootstrap weights."""
    saved=checkpoint_state
    codec=CompactEncoder(ROOT/cfg['dcae_dir'],cfg['encoder_activation_checkpoint'],diagnostic_random,
                         saved_encoder=None if saved is None else saved['encoder']).cuda()
    if saved is not None:
        assert saved['stage']==stage
        if 'quantizer.deltas' in saved['encoder']:
            from .quantization import MultiRateQuantizer
            codec.quantizer=MultiRateQuantizer(saved['encoder']['quantizer.deltas']).cuda()
        codec.load_state_dict(saved['encoder'],strict=True)
        if stage=='A':return codec,ReferenceVAE(),None
        decoder=FrozenPiD(cfg)
        if stage in ('2B','2C'):
            assert saved['pid_lora_config']==decoder.lora_metadata
            assert saved['pid_base_weights']==json.loads((ROOT/'configs/pid_weights.json').read_text())
            decoder.load_lora_state_dict(saved['pid_lora'])
        if stage=='2C':
            assert saved['side_config']==cfg['side_branch']
            decoder.enable_side_branch()
            if 'quantizer.deltas' in saved['side_branch']:
                from .quantization import MultiRateQuantizer
                decoder.side_branch.quantizer=MultiRateQuantizer(saved['side_branch']['quantizer.deltas'],channels=cfg['side_branch']['channels']).cuda()
            decoder.side_branch.load_state_dict(saved['side_branch'],strict=True)
            if decoder.joint_finetune:decoder.train_lora()
            else:codec.requires_grad_(False).eval()
        return codec,decoder.reference,decoder
    if stage=='2A':
        from .quantization import MultiRateQuantizer
        if cfg.get('quantization_trainable'):raise ValueError('Stage 2A uses fixed anchors without rate loss')
        source=ROOT/cfg['initial_checkpoint']
        saved=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
        assert saved['stage']=='B' and not saved['debug'],'Stage 2A requires a production Stage 1B checkpoint'
        for key in ['dcae_repo','dcae_revision','caption','decoder_seed']:
            assert saved['config'][key]==cfg[key],f'Initialization mismatch: {key}'
        anchors=json.loads((ROOT/cfg['quantization_anchors']).read_text())
        # Recalibrated Stage 2A anchors retain the original Stage B bootstrap provenance.
        bootstrap=anchors.get('bootstrap_source',anchors)
        assert bootstrap['source_identity']==saved['identity'] and bootstrap['source_stage_step']==saved['stage_step']
        assert bootstrap['data_fingerprint']==saved['data_fingerprint']
        codec.load_state_dict(saved['encoder'],strict=True)
        codec.initialization=dict(source_checkpoint=str(source),source_stage_step=saved['stage_step'],
                                  global_optimizer_step=saved['global_optimizer_step'],
                                  data_fingerprint=saved['data_fingerprint'],source_identity=saved['identity'])
        del saved
        codec.quantizer=MultiRateQuantizer(ROOT/cfg['quantization_anchors']).cuda()
    if stage in ('2B','2C'):
        from .quantization import MultiRateQuantizer
        if cfg.get('quantization_trainable'):
            raise ValueError('Stage 2B keeps calibrated anchors fixed')
        source=ROOT/cfg['initial_checkpoint']
        saved=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
        joint=stage=='2C' and cfg.get('stage_2c_joint_finetune',False)
        expected=('2C' if joint else '2B') if stage=='2C' else '2A'
        assert saved['stage']==expected and not saved['debug'], f'Stage {stage} requires production Stage {expected}'
        if stage=='2C':
            with source.open('rb') as f:
                assert hashlib.file_digest(f,'sha256').hexdigest()==cfg['initial_checkpoint_sha256'], 'Initialization checkpoint changed'
            source_lora=saved['pid_lora'];source_lora_config=saved['pid_lora_config']
            if joint:
                source_side_cfg=saved['side_config']
                assert {k:v for k,v in source_side_cfg.items() if k!='adaln_scope'}=={k:v for k,v in cfg['side_branch'].items() if k!='adaln_scope'}, 'Side architecture changed at joint transition'
                source_side=saved['side_branch']
        for key in ['dcae_repo','dcae_revision','caption','decoder_seed','rate_targets']:
            assert saved['config'][key]==cfg[key], f'Initialization mismatch: {key}'
        codec.quantizer=MultiRateQuantizer(saved['encoder']['quantizer.deltas']).cuda()
        codec.load_state_dict(saved['encoder'],strict=True)
        codec.initialization=dict(source_checkpoint=str(source),source_stage_step=saved['stage_step'],
            global_optimizer_step=saved['global_optimizer_step'],sample_cursor=saved['sample_cursor'],
            data_fingerprint=saved['data_fingerprint'],source_identity=saved['identity'])
        del saved
    # Stage A should not keep a 1.4B decoder resident during encoder-only updates.
    if stage=='A': return codec,ReferenceVAE(),None
    decoder=FrozenPiD(cfg)
    if stage=='2C':
        assert source_lora_config==decoder.lora_metadata
        decoder.load_lora_state_dict(source_lora)
        codec.requires_grad_(False).eval()
        decoder.enable_side_branch()
        if cfg.get('stage_2c_joint_finetune',False):
            if 'quantizer.deltas' in source_side:
                from .quantization import MultiRateQuantizer
                decoder.side_branch.quantizer=MultiRateQuantizer(source_side['quantizer.deltas'],channels=cfg['side_branch']['channels']).cuda()
            decoder.side_initialization=decoder.side_branch.load_source_state_dict(source_side,source_side_cfg)
            codec.initialization['adaln_expansion']={k:v for k,v in decoder.side_initialization.items() if k!='key_mapping'}
            codec.requires_grad_(True).train()
            decoder.train_lora()
        if cfg.get('side_quantization_anchors') and decoder.side_branch.quantizer is None:
            from .quantization import MultiRateQuantizer
            path=ROOT/cfg['side_quantization_anchors']
            anchors=json.loads(path.read_text())
            assert anchors['source_sha256']==cfg['initial_checkpoint_sha256'], 'Side anchors use a different source checkpoint'
            assert anchors['data_fingerprint']==codec.initialization['data_fingerprint']
            decoder.side_branch.quantizer=MultiRateQuantizer(path,channels=cfg['side_branch']['channels']).cuda()
            assert decoder.side_branch.quantizer.num_rates==codec.quantizer.num_rates
    return codec,decoder.reference,decoder


class JointCodec(nn.Module):
    """DDP sees encoder AND PiD LoRA in the same differentiable forward."""
    def __init__(self, codec, decoder):
        super().__init__()
        self.codec=codec
        self.decoder=decoder

    @property
    def quantizer(self):
        return self.codec.quantizer

    def train(self,mode=True):
        super().train(mode)
        if self.decoder.side_branch is not None and not self.decoder.joint_finetune:self.codec.eval()
        return self

    def decode_side(self,zhat,s):
        x_cond=self.decoder(zhat,side=s)
        residual=self.decoder.side_branch.rgb(s).float()
        return x_cond,residual,(x_cond.float()+residual).clamp(-1,1)

    def forward(self, x, rates=None, return_details=False, encoded=None, target_features=None):
        if self.decoder.side_branch is not None:
            # In joint mode the encoder/STE graph reaches the second PiD pass.
            # The baseline used for side feature extraction always stays SG.
            with contextlib.nullcontext() if self.decoder.joint_finetune else torch.no_grad():
                z=self.codec.encode(x) if encoded is None else encoded
                zhat,symbols,rates=self.quantizer(z,rates,return_symbols=True)
            with torch.no_grad():
                base=self.decoder(zhat)
                fx=self.decoder.full_vae(x.float()).detach() if target_features is None else target_features.detach()
                fb=self.decoder.full_vae(base.float()).detach()
            s_raw=self.decoder.side_branch.query(fx,fb)
            s=s_raw;side_symbols=None
            if self.decoder.side_branch.quantizer is not None:
                s,side_symbols,_=self.decoder.side_branch.quantizer(s_raw,rates,return_symbols=True)
            cond,residual,pred=self.decode_side(zhat,s)
            details=dict(zhat=zhat,z=z,symbols=symbols,rates=rates,base=base,
                         fx=fx,fbase=fb,side=s,side_raw=s_raw,side_symbols=side_symbols,
                         conditioned=cond,residual=residual,prediction=pred)
            return details if return_details else pred
        zhat,z,symbols,rates=self.codec(x,rates,return_details=True)
        prediction=self.decoder(zhat)
        return (zhat,z,symbols,rates,prediction) if return_details else prediction
