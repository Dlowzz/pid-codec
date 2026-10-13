"""Frozen perceptual metrics with differentiable image inputs; no model rewrite."""
from pathlib import Path
import hashlib
import torch
from torch import nn
import lpips
from DISTS_pytorch import DISTS


class ReconstructionMetrics(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lpips_weight = cfg['lambda_lpips']
        self.dists_weight = cfg['lambda_dists']
        self.lpips = lpips.LPIPS(net='vgg', version='0.1', verbose=False)
        # Official standalone implementation, using the already cached learned weights.
        # Its default load path is sys.prefix/weights.pt; supply our pinned asset explicitly.
        self.dists = DISTS(load_weights=False)
        weight_path=Path(cfg['dists_weights'])
        if cfg.get('dists_weights_sha256'):
            with weight_path.open('rb') as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=cfg['dists_weights_sha256']:
                    raise ValueError('DISTS weights changed from the pinned asset')
        weights = torch.load(weight_path, map_location='cpu', weights_only=True)
        with torch.no_grad():
            self.dists.alpha.copy_(weights['alpha'].reshape_as(self.dists.alpha))
            self.dists.beta.copy_(weights['beta'].reshape_as(self.dists.beta))
        self.eval().requires_grad_(False)

    def forward(self, pred, target):
        return self.lpips(pred.float(), target.float())

    def dists_loss(self, pred, target):
        # DISTS defaults to no_grad even if its inputs require gradients.
        return self.dists((pred.float()+1)/2, (target.float()+1)/2,
                          require_grad=torch.is_grad_enabled(), batch_average=False)
