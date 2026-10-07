#!/usr/bin/env python3
"""Post images, all on black. Numbers: docs/RETROSPECTIVE.md, results/baseline-x9-2026-10-07.json.
Outputs: results/cover-black.png (TensorFold hero, backdrop darkened, + title), results/cards-1-black.png and
results/cards-2-black.png (four large result cards each), results/results-black.png (everything on one sheet)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont
import numpy as np, os

W,H=1600,900; DPI=100
BLACK="#000000"; CARD="#0e0e12"; EDGE="#2a2a33"; FG="#ffffff"; MUTED="#b9bcc8"; DIM="#7c8090"
CORAL="#ff9a7a"; BLUE="#7cc4ff"; LIME="#9af7c0"
URL="github.com/kachowtowmater/tensorfold-glm53-rtx-pro-6000"
CREDIT="Built on TensorFold (ashhart, Apache-2.0) · Aevonix 4× RTX PRO 6000 recipe · Mia-AiLab EXL3 4bpw pack · commdata2338 engine patch · incoai DFlash2 · turboderp exllamav3"
cards=[
 ("Image-turn TTFT (28k-token prefix)","2.87 s → 0.12 s","before: the whole prefix was re-prefilled every turn","vision-prefix resume patch (ours) · 24× faster",LIME),
 ("4 cold agents sharing a 40k prefix","12.4 s → 3.3 s","before: 6.3–12.4 s TTFT per agent","burst-prefix sharing · turn 13 s → 4.2 s",LIME),
 ("Agent turn at 8 concurrent, real traffic","5.8 s → 2.5 s","old SGLang serve → TensorFold, same clients","2.3× faster",LIME),
 ("top_k = -1 / top_p 1.0 decode","24 → 188 tok/s","before: CPU sampler collapse","commdata2338 engine patch · 7.8×",LIME),
 ("Cache evictions per 1,000 agent requests","40 → 0","CACHE_ENTRIES 64 → 128","4-agent warm turn 2.9 → 2.5 s · churn 5.8 → 3.1 s",LIME),
 ("Cold TTFT, 40-conversation churn","5.9 s → 5.2 s","HC_SPLIT_MIN_ROWS 512 → 2048","small fills skip the 4-rank split · every cell",LIME),
 ("Raw single-stream decode (prose / code)","223 / 433 tok/s","unchanged = the Aevonix recipe's number","expert weights are ~18% of a spec-decode step",CORAL),
 ("Quality on our own harness","gsm8k 96.5%","1273 / 1319, full test set · CJK clean","top-k 8→6 kept quality, gained 0% → requant parked",BLUE),
]

def hero_black():
    hero=Image.open("results/assets/tensorfold-hero.png").convert("RGB")
    arr=np.asarray(hero).astype(np.float32); bgc=np.array([11,22,99],dtype=np.float32)
    dist=np.sqrt(((arr-bgc)**2).sum(axis=2)); k=np.clip((dist-18)/60,0,1)[...,None]
    im=Image.fromarray((arr*k).clip(0,255).astype(np.uint8)); im.save("results/assets/tensorfold-hero-black.png"); return im

def font(sz,bold=False):
    for p in (["/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"]):
        if os.path.exists(p): return ImageFont.truetype(p,sz)
    return ImageFont.load_default()

def sheet(title, subset, out, footer=True):
    fig=plt.figure(figsize=(W/DPI,H/DPI),dpi=DPI); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off"); fig.patch.set_facecolor(BLACK)
    def text(x,y,s,size,color=FG,weight="normal",ha="left",va="center"): ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va=va,zorder=5)
    text(60,H-62,title,26,FG,weight="bold"); text(W-60,H-62,"before → after · same box · same prompts",14,MUTED,ha="right")
    cw,ch,gx,gy=730,320,20,20; x0=60; y_top=H-110
    for i,(lab,aft,bef,note,acc) in enumerate(subset):
        r,c=divmod(i,2); x=x0+c*(cw+gx); y=y_top-ch-r*(ch+gy)
        ax.add_patch(FancyBboxPatch((x,y),cw,ch,boxstyle="round,pad=0,rounding_size=18",fc=CARD,ec=EDGE,lw=1.5,zorder=2))
        ax.add_patch(FancyBboxPatch((x,y+22),7,ch-44,boxstyle="round,pad=0,rounding_size=3.5",fc=acc,ec="none",zorder=3))
        text(x+34,y+ch-44,lab,19,MUTED)
        text(x+34,y+ch/2+6,aft,54,acc,weight="bold")
        text(x+34,y+66,bef,16,DIM); text(x+34,y+34,note,16,MUTED)
    if footer:
        text(60,52,CREDIT,11.5,DIM); text(60,26,URL,14,FG,weight="bold")
    fig.savefig(out,dpi=DPI,facecolor=BLACK); plt.close(fig); print("wrote",out)

def cover(hero):
    hw,hh=hero.size; band=hero.resize((W,int(hh*W/hw)),Image.LANCZOS); bh=band.size[1]
    out=Image.new("RGB",(W,H),BLACK); out.paste(band,(0,0)); d=ImageDraw.Draw(out)
    y=bh+34
    d.text((60,y),"GLM-5.3-Flash on 4× RTX PRO 6000",font=font(58,True),fill=(255,255,255))
    d.text((60,y+78),"the agent-path overlay: two days of paired A/B, open patches, knobs, bench scripts, every result incl. the ones that lost",font=font(24),fill=(185,188,200))
    stats=[("2.87 s → 0.12 s","image-turn TTFT"),("12.4 s → 3.3 s","cold 4-agent burst TTFT"),("24 → 188 tok/s","top_k = -1 sampler fix"),("223 / 433 tok/s","raw decode, unchanged")]
    x=60
    for v,l in stats:
        col=(154,247,192) if "→" in v else (255,154,122)
        d.text((x,y+140),v,font=font(40,True),fill=col); d.text((x,y+192),l,font=font(18),fill=(124,128,144)); x+=385
    d.text((60,H-58),CREDIT,font=font(15),fill=(124,128,144)); d.text((60,H-34),URL,font=font(19,True),fill=(255,255,255))
    d.text((W-60,H-34),"TensorFold artwork: ashhart/TensorFold (Apache-2.0), backdrop darkened",font=font(14),fill=(124,128,144),anchor="ra")
    out.save("results/cover-black.png"); print("wrote results/cover-black.png")

hero=hero_black(); cover(hero)
sheet("What improved (1/2)",cards[:4],"results/cards-1-black.png")
sheet("What improved (2/2), and what did not",cards[4:],"results/cards-2-black.png")
# all-in-one sheet for the README
c1=Image.open("results/cover-black.png"); s1=Image.open("results/cards-1-black.png"); s2=Image.open("results/cards-2-black.png")
allin=Image.new("RGB",(W,H*3),BLACK); allin.paste(c1,(0,0)); allin.paste(s1,(0,H)); allin.paste(s2,(0,2*H)); allin.save("results/results-black.png"); print("wrote results/results-black.png")
for old in ("results/results-2026-10-07.png",):
    if os.path.exists(old): os.remove(old)
