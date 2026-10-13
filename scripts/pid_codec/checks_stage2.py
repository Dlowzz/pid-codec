"""Stage 2A mathematical parity and real reconstruction-gradient checks."""
import argparse,ast,json,hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from .frequency import FrequencyLoss
from .quantization import MultiRateQuantizer,SymbolEntropy
from .train import load_config,resolved,setup,make_frequency
from .models import load_models


def basic(official):
    tree=ast.parse(Path(official).read_text())
    names={'_create_dct_matrix','_rgb2ycbcr','_dct','_build_freq_weight'}
    methods=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(methods)==4
    for n in methods:n.decorator_list=[]
    node=ast.ClassDef(name='Oracle',bases=[],keywords=[],body=methods,decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    namespace={'torch':torch,'F':F};exec(compile(module,official,'exec'),namespace)
    oracle=namespace['Oracle']();oracle.block_size=8;oracle.dct_mat=oracle._create_dct_matrix(8)
    loss=FrequencyLoss();torch.testing.assert_close(loss.dct_matrix,oracle.dct_mat,rtol=0,atol=0)
    for quality in [1,49,50,85,100]:
        for mode,gamma in [('inv',1.),('inv_gamma',1.),('inv_gamma',.7)]:
            actual=FrequencyLoss(quality,mode,gamma)
            torch.testing.assert_close(actual.frequency_weight,oracle._build_freq_weight(quality,mode,gamma),rtol=0,atol=0)
    max_error=0.
    for shape in [(2,3,512,512),(1,3,19,23)]:
        x=torch.randn(shape,requires_grad=True);y=torch.randn(shape)
        torch.testing.assert_close(loss.ycbcr(x),oracle._rgb2ycbcr(x),rtol=0,atol=0)
        a=loss(x,y);b=(oracle._build_freq_weight()*(oracle._dct(oracle._rgb2ycbcr(x))-oracle._dct(oracle._rgb2ycbcr(y))).square()).mean()
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        ga=torch.autograd.grad(a,x,retain_graph=True)[0];gb=torch.autograd.grad(b,x)[0]
        torch.testing.assert_close(ga,gb,rtol=0,atol=0);max_error=max(max_error,float((ga-gb).abs().max()))
        assert loss(x,x).item()==0
    torch.testing.assert_close(loss.dct_matrix@loss.dct_matrix.T,torch.eye(8),atol=1e-6,rtol=0)
    q=MultiRateQuantizer(torch.linspace(.1,3,6)[:,None].expand(6,16))
    z=torch.randn(6,16,16,16,requires_grad=True);rates=torch.arange(6)
    out,symbols,_=q(z,rates,True);delta=q.deltas[rates,:,None,None]
    torch.testing.assert_close(out,torch.round(z/delta)*delta,rtol=0,atol=0)
    grad=torch.autograd.grad(out.sum(),z)[0];torch.testing.assert_close(grad,torch.ones_like(z),rtol=1e-6,atol=1e-6)
    assert not list(q.parameters())
    stat=SymbolEntropy();binary=torch.arange(4096).reshape(1,16,16,16)%2;stat.update(binary)
    assert stat.result()['estimated_bpp']==1/64 and stat.result()['empirical_entropy']==1
    return dict(deco_forward_gradient_exact=True,maximum_gradient_error=max_error,hard_forward_exact=True,ste_gradient=True,
                fixed_anchors=True,entropy_known_binary_case=True,official_sha256=hashlib.sha256(Path(official).read_bytes()).hexdigest())


def gradient(cfg):
    import lpips
    from .data import Images
    setup(cfg['seed']);codec,reference,decoder=load_models(cfg,'2A')
    metric=lpips.LPIPS(net='vgg',verbose=False).cuda().eval().requires_grad_(False);frequency=make_frequency(cfg,'2A')
    saved=torch.load(resolved(cfg['initial_checkpoint']),map_location='cpu',weights_only=False,mmap=True)
    for k,v in saved['encoder'].items():assert torch.equal(v,codec.state_dict()[k].cpu()),k
    del saved
    x=Images(resolved(cfg['manifest_dir']),'val',cfg['seed'])[0][None].cuda();reports=[]
    for rate in [0,5]:
        codec.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):zhat,z,symbols,rates=codec(x,rate,True)
        assert z.shape==(1,16,16,16)
        pred=decoder(zhat);l1=F.l1_loss(pred.float(),x);lp=metric(pred.float(),x).mean();freq=frequency(pred,x)
        rec=l1+.5*lp
        gr=torch.autograd.grad(rec,zhat,retain_graph=True)[0]
        gf=torch.autograd.grad(freq,zhat,retain_graph=True)[0]
        # Deliberately exclude alignment: prove reconstruction gradients cross PiD + STE.
        (rec+freq).backward()
        encnorm=sum(p.grad.float().square().sum() for p in codec.encoder.parameters() if p.grad is not None).sqrt().item()
        projnorm=codec.projection.weight.grad.float().norm().item()
        assert encnorm>0 and projnorm>0
        assert all(not p.requires_grad and p.grad is None for p in decoder.parameters())
        assert all(not p.requires_grad and p.grad is None for p in metric.parameters())
        reports.append(dict(rate=rate,l1=float(l1.detach()),lpips=float(lp.detach()),frequency=float(freq.detach()),
                            reconstruction_latent_grad_norm=float(gr.norm()),frequency_latent_grad_norm=float(gf.norm()),
                            frequency_to_reconstruction_gradient_ratio=float(gf.norm()/gr.norm()),
                            encoder_grad_norm=encnorm,projection_grad_norm=projnorm,pid_frozen=True,lpips_frozen=True))
        del zhat,z,pred,l1,lp,freq,rec,gr,gf
    return dict(initialization_exact=True,initialization=codec.initialization,gradient_checks=reports)


def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['basic','gradient']);p.add_argument('--official');p.add_argument('--config',default='configs/pid_codec/stage2a.json')
    args=p.parse_args();cfg=load_config(args.config)
    report=basic(args.official) if args.kind=='basic' else gradient(cfg)
    dest=resolved(cfg['output_dir'])/'checks';dest.mkdir(parents=True,exist_ok=True)
    (dest/(args.kind+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
