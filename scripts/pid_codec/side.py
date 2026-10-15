"""Stage 2C side information: spatial queries, additive AdaLN, signed RGB residual.

No upstream forward is copied. Hooks add to the existing image AdaLN output;
the scope is entered INSIDE each checkpointed network invocation, including
backward recomputation. Original time/text modulation and LoRA stay enabled.
"""
import contextlib
import math
import torch
from torch import nn
import torch.nn.functional as F


def position_2d(size,dim):
    assert dim%4==0
    # Both grids use the same coordinate system (query centers vs feature centers).
    p=(torch.arange(size,dtype=torch.float32)+.5)/size*2*math.pi
    y,x=torch.meshgrid(p,p,indexing='ij')
    f=10000**(-torch.arange(dim//4,dtype=torch.float32)/(dim//4))
    return torch.cat([(x[...,None]*f).sin(),(x[...,None]*f).cos(),
                      (y[...,None]*f).sin(),(y[...,None]*f).cos()],-1)


class QueryLayer(nn.Module):
    def __init__(self,dim,heads):
        super().__init__()
        self.q_norm=nn.LayerNorm(dim);self.kv_norm=nn.LayerNorm(dim)
        self.attn=nn.MultiheadAttention(dim,heads,batch_first=True,dropout=0.)
        self.ff_norm=nn.LayerNorm(dim)
        self.ff=nn.Sequential(nn.Linear(dim,4*dim),nn.GELU(),nn.Linear(4*dim,dim))
    def forward(self,q,kv):
        k=self.kv_norm(kv)
        q=q+self.attn(self.q_norm(q),k,k,need_weights=False)[0]
        return q+self.ff(self.ff_norm(q))


class SpatialQueries(nn.Module):
    def __init__(self,cfg):
        super().__init__();d=cfg['dim'];self.grid=cfg['grid'];self.window=cfg['local_window']
        self.input_projection=nn.Linear(48,d)
        self.queries=nn.Parameter(torch.randn(1,self.grid**2,d)*.02)
        self.register_buffer('query_position',position_2d(self.grid,d).reshape(1,-1,d))
        self.register_buffer('feature_position',position_2d(self.grid*self.window,d)[None])
        self.layers=nn.ModuleList([QueryLayer(d,cfg['heads']) for _ in range(cfg['depth'])])
        self.output_projection=nn.Linear(d,cfg['channels'])
    def forward(self,fx,fb):
        b,_,h,w=fx.shape;g=self.grid;k=self.window
        if (h,w)!=(g*k,g*k):raise ValueError(f'Expected full-resolution VAE grid {(g*k,g*k)}, got {(h,w)}')
        # 256 aligned spatial queries, each attending to its corresponding 4x4
        # neighborhood. The three VAE features are concatenated per location.
        v=torch.cat([fx,fb,fx-fb],1).permute(0,2,3,1)
        v=self.input_projection(v)+self.feature_position
        v=v.reshape(b,g,k,g,k,-1).permute(0,1,3,2,4,5).reshape(b*g*g,k*k,-1)
        q=(self.queries+self.query_position).expand(b,-1,-1).reshape(b*g*g,1,-1)
        for layer in self.layers:q=layer(q,v)
        return self.output_projection(q).reshape(b,g,g,-1).permute(0,3,1,2).contiguous()


class SpatialAdaLN(nn.Module):
    def __init__(self,cfg,net):
        super().__init__();self.grid=cfg['grid'];self.patch_grid=512//net.patch_size
        if self.patch_grid!=self.grid*2 or net.use_ed or getattr(net,'_cp_group',None) is not None:
            raise ValueError('Stage 2C expects the current 512-image, 32x32 patch grid without context parallelism/ED')
        self.scope=cfg.get('adaln_scope','last_half')
        if self.scope not in ('last_half','all'):raise ValueError('AdaLN scope must be last_half or all')
        self.block_indices=list(range(0 if self.scope=='all' else len(net.patch_blocks)//2,len(net.patch_blocks)))
        d=cfg['dim']
        self.upsample=nn.Sequential(nn.ConvTranspose2d(cfg['channels'],d,4,2,1),nn.SiLU(),nn.Conv2d(d,d,3,padding=1),nn.SiLU())
        self.heads=nn.ModuleList([nn.Linear(d,6*net.hidden_size) for _ in self.block_indices])
        for head in self.heads:nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self._features=None
        self._handles=[]
        mods=[net.patch_blocks[i].adaLN_modulation_img for i in self.block_indices]
        if len({id(m) for m in mods})!=len(mods):raise ValueError('Shared AdaLN requires explicit per-block hooks')
        for j,mod in enumerate(mods):
            def inject(module,args,output,index=j):
                if self._features is None:return output
                delta=self.heads[index](self._features).to(output.dtype)
                return output+delta
            self._handles.append(mod.register_forward_hook(inject))
    @contextlib.contextmanager
    def condition(self,s):
        previous=self._features
        try:
            self._features=self.upsample(s).flatten(2).transpose(1,2)
            yield
        finally:self._features=previous


class RGBResidualDecoder(nn.Module):
    def __init__(self,channels):
        super().__init__()
        widths=[128,128,64,32,16,8]
        self.input_projection=nn.Conv2d(channels,widths[0],3,padding=1)
        self.blocks=nn.ModuleList([nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(a,b,3,padding=1),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.SiLU()) for a,b in zip(widths,widths[1:])])
        self.head=nn.Conv2d(widths[-1],3,3,padding=1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,s):
        h=F.silu(self.input_projection(s))
        for block in self.blocks:h=block(h)
        return self.head(h)  # Signed residual; no sigmoid/tanh, added only in RGB.


class SideBranch(nn.Module):
    def __init__(self,cfg,net):
        super().__init__()
        self.query=SpatialQueries(cfg)
        self.adaln=SpatialAdaLN(cfg,net)
        self.rgb=RGBResidualDecoder(cfg['channels'])
        self.quantizer=None  # Optional fixed six-rate scalar anchors, loaded after source weights.

    def load_source_state_dict(self,state,source_cfg):
        """Preserve trained late heads when expanding last-half AdaLN to all blocks."""
        source_scope=source_cfg.get('adaln_scope','last_half')
        mapping={name:name for name in state}
        added=[]
        if source_scope!=self.adaln.scope:
            if (source_scope,self.adaln.scope)!=('last_half','all'):
                raise ValueError('Only last_half -> all AdaLN expansion is supported')
            offset=len(self.adaln.block_indices)//2
            added=list(range(offset))
            for name in state:
                if name.startswith('adaln.heads.'):
                    parts=name.split('.')
                    parts[2]=str(int(parts[2])+offset)
                    mapping[name]='.'.join(parts)
        result=self.load_state_dict({mapping[name]:value for name,value in state.items()},strict=False)
        expected={f'adaln.heads.{i}.{field}' for i in added for field in ['weight','bias']}
        if result.unexpected_keys or set(result.missing_keys)!=expected:
            raise ValueError(f'Unexpected side checkpoint mismatch: {result}')
        for i in added:
            assert all(torch.count_nonzero(p)==0 for p in self.adaln.heads[i].parameters())
        return dict(source_scope=source_scope,target_scope=self.adaln.scope,added_zero_blocks=added,key_mapping=mapping)
    def parameter_counts(self):
        return {name:sum(p.numel() for p in module.parameters()) for name,module in self.named_children()}
