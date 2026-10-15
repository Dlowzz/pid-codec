"""Standard 70x70 PatchGAN + hinge loss, as used by DiT-IC via taming.

The discriminator is the BatchNorm (use_actnorm=False), 3-layer specialization
of CompVis/taming-transformers/taming/modules/discriminator/model.py.
See licenses/taming-transformers.txt. No additional pretrained model is needed.
"""
import torch
from torch import nn
import torch.nn.functional as F


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[nn.Conv2d(3,64,4,2,1),nn.LeakyReLU(.2,True)]
        for a,b,stride in [(64,128,2),(128,256,2),(256,512,1)]:
            layers += [nn.Conv2d(a,b,4,stride,1,bias=False),nn.BatchNorm2d(b),nn.LeakyReLU(.2,True)]
        layers.append(nn.Conv2d(512,1,4,1,1))
        self.main=nn.Sequential(*layers)
        for module in self.modules():
            if isinstance(module,nn.Conv2d):nn.init.normal_(module.weight,0.,.02)
            elif isinstance(module,nn.BatchNorm2d):
                nn.init.normal_(module.weight,1.,.02);nn.init.zeros_(module.bias)

    def forward(self,x):return self.main(x)


def generator_loss(discriminator,prediction):
    # Freeze both weights and BN running statistics for the generator update.
    net=discriminator.module if hasattr(discriminator,'module') else discriminator
    net.requires_grad_(False).eval()
    with torch.autocast(prediction.device.type,dtype=torch.bfloat16):
        return -net(prediction).float().mean()


def discriminator_loss(discriminator,real,fake):
    discriminator.requires_grad_(True).train()
    with torch.autocast(real.device.type,dtype=torch.bfloat16):
        real_logits,fake_logits=discriminator(torch.cat([real.detach(),fake.detach()])).float().chunk(2)
    loss=.5*(F.relu(1-real_logits).mean()+F.relu(1+fake_logits).mean())
    stats=dict(discriminator_loss=loss.detach(),real_score=real_logits.detach().mean(),fake_score=fake_logits.detach().mean(),
               real_accuracy=(real_logits.detach()>0).float().mean(),fake_accuracy=(fake_logits.detach()<0).float().mean())
    return loss,stats
