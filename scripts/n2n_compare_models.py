"""splits=2 vs splits=4: photometry bins AND injection recovery, held-out m92."""
import sys, numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree
sys.path.insert(0,"/home/taylor/Documents/srt")
import sep, torch
from nn import denoiser
from nn.noise2noise_model import UNet
sep.set_extract_pixstack(3000000)
OUT="/tmp/claude-1000/-home-taylor-Documents-srt/da3595b7-58db-41eb-aa9d-5c4610a03787/scratchpad"

A=np.load(f"{OUT}/m92_halfA.npy"); B=np.load(f"{OUT}/m92_halfB.npy")
S=(A+B)/2.0
bk=sep.Background(S); rms=float(bk.globalrms); rs=S-bk
print(f"m92 full stack, rms {rms:.4f}", flush=True)

# --- real-source photometry setup ---
src=sep.extract(rs, thresh=8.0*rms, err=rms)
xy=np.column_stack([src["x"],src["y"]]); t=cKDTree(xy)
xy=xy[np.array([len(t.query_ball_point(q,24.0))==1 for q in xy])]
fa,_,fla=sep.sum_circle(rs,xy[:,0],xy[:,1],8.0,err=rms)
BINS=[(0,2,"brightest"),(2,4,"mid"),(4,6,"faint"),(6,99,"very faint")]

# --- injection setup ---
LEVELS=[1.0,2.0,3.0,5.0,8.0,20.0,40.0]; NPER=90; R=20
sigma=6.0/2.3548
yy,xx=np.mgrid[-R:R+1,-R:R+1]
psf=np.exp(-(xx**2+yy**2)/(2*sigma**2))
rng=np.random.default_rng(0)
tree2=cKDTree(np.column_stack([src["x"],src["y"]]))
inj=np.zeros_like(S); sites=[]
h,w=S.shape
for lvl in LEVELS:
    placed=guard=0
    while placed<NPER and guard<NPER*300:
        guard+=1
        y=int(rng.integers(400,h-400)); x=int(rng.integers(400,w-400))
        if tree2.query_ball_point([x,y],30.0): continue
        if sites and min((y-sy)**2+(x-sx)**2 for sy,sx,_ in sites)<60**2: continue
        inj[y-R:y+R+1,x-R:x+R+1]+=psf*(lvl*rms); sites.append((y,x,lvl)); placed+=1
iys=np.array([s[0] for s in sites],float); ixs=np.array([s[1] for s in sites],float)
ilv=np.array([s[2] for s in sites])
f_true,_,_=sep.sum_circle(inj,ixs,iys,8.0)
isnr=f_true/(rms*np.sqrt(np.pi*8.0**2))

def evaluate(tag, path):
    ck=torch.load(path,map_location="cpu",weights_only=True)
    m=UNet(); m.load_state_dict(ck["model_state"]); m.eval()
    d0=denoiser.denoise_frame(S.astype(np.float32),m).astype(np.float64)
    d1=denoiser.denoise_frame((S+inj).astype(np.float32),m).astype(np.float64)
    ds=d0-np.array(sep.Background(d0))
    fb,_,flb=sep.sum_circle(ds,xy[:,0],xy[:,1],8.0,err=rms)
    ok=(fla==0)&(flb==0)&(fa>0); Af,Bf=fa[ok],fb[ok]
    mag=-2.5*np.log10(Af/Af.max())
    print(f"\n=== {tag}  (checkpoint epoch {ck['epoch']}) ===", flush=True)
    print(f"  rms {rms:.4f} -> {float(sep.Background(d0).globalrms):.4f}", flush=True)
    ph={}
    for lo,hi,lab in BINS:
        mm=(mag>=lo)&(mag<hi)
        if mm.sum()>=3:
            ph[lab]=float(np.median(Bf[mm]/Af[mm]))
            print(f"  {lab:11s} n={mm.sum():5d}  {ph[lab]:.4f}", flush=True)
    fr,_,_=sep.sum_circle(d1-d0,ixs,iys,8.0)
    print(f"  {'injSNR':>8} {'recovered':>10}", flush=True)
    rec={}
    for lvl in LEVELS:
        mm=ilv==lvl
        rec[lvl]=float(np.median(fr[mm]/f_true[mm]))
        print(f"  {np.median(isnr[mm]):8.1f} {rec[lvl]:10.4f}", flush=True)
    return ph,rec

r2=evaluate("splits=2 (baseline)", "local/models/n2n_stack_R_300s_s2.pt")
r4=evaluate("splits=4 (quarters)", "local/models/n2n_stack_R_300s_s4.pt")

print("\n"+"="*66)
print(f"{'':14}{'splits=2':>12}{'splits=4':>12}   delta")
for _,_,lab in BINS:
    if lab in r2[0] and lab in r4[0]:
        a,b=r2[0][lab],r4[0][lab]
        print(f"{lab:14}{a:12.4f}{b:12.4f}   {b-a:+.4f}")
print("-"*66)
for lvl in LEVELS:
    a,b=r2[1][lvl],r4[1][lvl]
    print(f"{'inj '+str(lvl)+'x':14}{a:12.4f}{b:12.4f}   {b-a:+.4f}")
print("="*66)
