#!/usr/bin/env python3
"""Post image: the official TensorFold hero (ashhart/TensorFold assets/, Apache-2.0) as the top band, our measured
results as cards below, in the same palette. Numbers: docs/RETROSPECTIVE.md, results/baseline-x9-2026-10-07.json."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np, os

W,H=1600,900; BAND=290; DPI=100
NAVY="#000000"; NAVY2="#000000"; CARD="#0e0e12"; EDGE="#2a2a33"; FG="#ffffff"; MUTED="#b9bcc8"; DIM="#7c8090"
CORAL="#ff9a7a"; VIOLET="#c08cff"; BLUE="#7cc4ff"; LIME="#9af7c0"

# ---- cards (matplotlib) ----
fig=plt.figure(figsize=(W/DPI,(H-BAND)/DPI),dpi=DPI)
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H-BAND); ax.axis("off")
g=np.linspace(0,1,256).reshape(-1,1)
ax.imshow(g,extent=[0,W,0,H-BAND],aspect="auto",cmap=matplotlib.colors.LinearSegmentedColormap.from_list("bg",[NAVY2,NAVY]),zorder=0)
def text(x,y,s,size,color=FG,weight="normal",ha="left",va="center"): ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va=va,zorder=5)
def card(x,y,w,h,label,after,before,note,accent):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=14",fc=CARD,ec=EDGE,lw=1.2,zorder=2))
    ax.add_patch(FancyBboxPatch((x,y+12),5,h-24,boxstyle="round,pad=0,rounding_size=2.5",fc=accent,ec="none",zorder=3))
    text(x+26,y+h-22,label,13,MUTED)
    text(x+26,y+h-57,after,26,accent,weight="bold")
    text(x+26,y+28,before,11.5,DIM)
    text(x+26,y+12,note,11.5,MUTED)
cw,ch,gx,gy=732,112,20,12; x0=58; y_top=(H-BAND)-64
text(58,(H-BAND)-32,"GLM-5.3-Flash on 4× RTX PRO 6000 — the agent-path overlay",21,FG,weight="bold")
text(W-58,(H-BAND)-32,"before → after, same box, same prompts",13,MUTED,ha="right")
cards=[
 ("Image-turn TTFT (28k-token prefix)","2.87 s → 0.12 s","before: whole prefix re-prefilled every turn","vision-prefix resume patch (ours) · 24× faster",LIME),
 ("4 cold agents, shared 40k prefix","12.4 s → 3.3 s","before: 6.3–12.4 s TTFT per agent","burst-prefix sharing · turn 13 s → 4.2 s",LIME),
 ("Agent turn @ 8 concurrent, real traffic","5.8 s → 2.5 s","old SGLang serve → TensorFold, same clients","2.3× faster",LIME),
 ("top_k = -1 / top_p 1.0 decode","24 → 188 tok/s","before: CPU sampler collapse","commdata2338 engine patch · 7.8×",LIME),
 ("Cache evictions per 1,000 agent requests","40 → 0","CACHE_ENTRIES 64 → 128","4-agent warm turn 2.9 → 2.5 s · churn 5.8 → 3.1 s",LIME),
 ("Cold TTFT, 40-conversation churn","5.9 s → 5.2 s","HC_SPLIT_MIN_ROWS 512 → 2048","small fills skip the 4-rank split · every cell",LIME),
 ("Raw single-stream decode (prose / code)","223 / 433 tok/s","unchanged = the Aevonix recipe's number","expert weights are ~18% of a spec-decode step",CORAL),
 ("Quality on our own harness","gsm8k 96.5%","1273 / 1319 full set · CJK clean","top-k 8→6 kept quality, gained 0% → requant parked",BLUE),
]
for i,(lab,aft,bef,note,acc) in enumerate(cards):
    r,c=divmod(i,2); card(x0+c*(cw+gx), y_top-ch-r*(ch+gy), cw,ch,lab,aft,bef,note,acc)
text(58,36,"Built on TensorFold (ashhart, Apache-2.0) · Aevonix 4× RTX PRO 6000 recipe · Mia-AiLab EXL3 4bpw pack · commdata2338 engine patch · incoai DFlash2 · turboderp exllamav3",11,DIM)
text(58,16,"github.com/kachowtowmater/tensorfold-glm53-rtx-pro-6000",12.5,FG,weight="bold")
fig.savefig("/tmp/_cards.png",dpi=DPI,facecolor=NAVY2); plt.close(fig)

# ---- compose with the official hero ----
hero=Image.open("results/assets/tensorfold-hero.png").convert("RGB")
arr=np.asarray(hero).astype(np.float32)
bgc=np.array([11,22,99],dtype=np.float32)           # the flat navy backdrop
dist=np.sqrt(((arr-bgc)**2).sum(axis=2))             # colour distance from the backdrop
k=np.clip((dist-18)/60,0,1)[...,None]                # 0 = backdrop, 1 = mesh
arr=arr*k                                            # backdrop → black, mesh untouched
hero=Image.fromarray(arr.clip(0,255).astype(np.uint8)); hero.save("results/assets/tensorfold-hero-black.png")
hw,hh=hero.size; scale=W/hw; hero=hero.resize((W,int(hh*scale)),Image.LANCZOS)
top=max(0,(hero.size[1]-BAND)//2); band=hero.crop((0,top,W,top+BAND))
out=Image.new("RGB",(W,H),NAVY2); out.paste(band,(0,0)); out.paste(Image.open("/tmp/_cards.png").convert("RGB"),(0,BAND))
out.save("results/results-2026-10-07.png"); print("wrote results/results-2026-10-07.png", out.size)
