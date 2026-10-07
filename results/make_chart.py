#!/usr/bin/env python3
"""Results graphic for the README / post. Numbers from docs/RETROSPECTIVE.md and results/baseline-x9-2026-10-07.json (measured on mcqueen, 4x RTX PRO 6000, Oct 5-7 2026)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG="#0b0f17"; FG="#e6edf3"; MUTED="#8b949e"; GOOD="#3fb950"; BEFORE="#484f58"; ACC="#58a6ff"
fig=plt.figure(figsize=(16,9),dpi=120,facecolor=BG)
fig.text(0.04,0.93,"GLM-5.3-Flash on TensorFold · 4× RTX PRO 6000 (Max-Q, 250 W)",color=FG,fontsize=26,weight="bold")
fig.text(0.04,0.885,"Agent-path gains from two days of paired A/B on one box, Oct 5–7 2026. Raw single-stream decode stayed at the Aevonix",color=MUTED,fontsize=13)
fig.text(0.04,0.855,"recipe's number (223 prose / 433 code tok/s): under speculative decoding, expert weights are only ~18% of a step.",color=MUTED,fontsize=13)

# left: before/after bars (latency, lower is better), log scale
ax=fig.add_axes([0.27,0.20,0.33,0.58]); ax.set_facecolor(BG)
rows=[("Image-turn TTFT\n(28k prefix, turns 2–3)",2.87,0.12,"s"),
      ("4 cold agents, shared 40k prefix\nTTFT (worst of 4)",12.4,3.3,"s"),
      ("Same cold burst\nturn time",13.0,4.2,"s"),
      ("Agent turn @ 8 concurrent\n(real traffic, SGLang → TF)",5.8,2.5,"s"),
      ("40-conversation churn, cold TTFT\n(HC_SPLIT_MIN_ROWS 512 → 2048)",5.9,5.2,"s"),
      ("4-agent warm turn\n(recipe defaults → tuned knobs)",2.9,2.5,"s")]
y=range(len(rows))[::-1]
for yi,(lab,b,a,u) in zip(y,rows):
    ax.barh(yi+0.18,b,height=0.34,color=BEFORE); ax.barh(yi-0.18,a,height=0.34,color=GOOD)
    ax.text(b*1.05,yi+0.18,f"{b:g} {u}",va="center",color=MUTED,fontsize=11)
    ax.text(a*1.05,yi-0.18,f"{a:g} {u}  ({b/a:.1f}× faster)",va="center",color=GOOD,fontsize=11,weight="bold")
ax.set_yticks(list(y)); ax.set_yticklabels([r[0] for r in rows],color=FG,fontsize=11.5)
ax.set_xscale("log"); ax.set_xlim(0.08,60); ax.set_ylim(-0.6,len(rows)-0.4); ax.set_xlabel("seconds (log scale, lower is better)",color=MUTED)
ax.tick_params(axis="x",colors=MUTED); [s.set_color("#30363d") for s in ax.spines.values()]
ax.barh([-5],[0],color=BEFORE,label="before"); ax.barh([-5],[0],color=GOOD,label="after")
ax.legend(loc="lower right",frameon=False,labelcolor=FG)

# right: throughput + facts
ax2=fig.add_axes([0.66,0.20,0.32,0.58]); ax2.set_facecolor(BG); ax2.axis("off")
facts=[("explicit top_k=-1 / top_p 1.0 decode","24 → 188 tok/s","sampler collapse fixed (commdata patch)"),
       ("cache evictions per ~1,000 agent requests","40 → 0","CACHE_ENTRIES 128"),
       ("prefill seconds lost to image re-prefill","80% → 0%","vision-prefix resume patch (ours)"),
       ("gsm8k, full test set, our harness","96.5%","unchanged; 1273/1319"),
       ("top-k 8→6 (−25% expert bytes)","0% speed","weights are ~18% of a spec-decode step"),
       ("mixed-width EXL3 requant","≈ +3% projected","parked: not worth a 12 h window")]
yy=0.97
for k,v,n in facts:
    ax2.text(0.0,yy,k,color=MUTED,fontsize=11.5,transform=ax2.transAxes)
    ax2.text(0.0,yy-0.055,v,color=ACC if "→" in v or "%" in v else FG,fontsize=20,weight="bold",transform=ax2.transAxes)
    ax2.text(0.0,yy-0.095,n,color=MUTED,fontsize=10,transform=ax2.transAxes)
    yy-=0.165

fig.text(0.04,0.07,"Built on: TensorFold (ashhart, Apache-2.0) · Aevonix 4× RTX PRO 6000 recipe (86 patches) · Mia-AiLab EXL3 4bpw pack · commdata2338 engine patch · incoai DFlash2 drafter · exllamav3 (turboderp)",color=MUTED,fontsize=10.5)
fig.text(0.04,0.04,"github.com/kachowtowmater/tensorfold-glm53-rtx-pro-6000 — patches, knobs, bench scripts, every result incl. the ones that lost",color=FG,fontsize=11)
fig.savefig("results/results-2026-10-07.png",facecolor=BG); print("wrote results/results-2026-10-07.png")
