#!/usr/bin/env python3
"""Agent-shaped load for an OpenAI-compatible chat endpoint. Deterministic content per stream/turn.
Phase A: S concurrent conversations, base ~BASE words, T turns each appending a tool result and asking for a short next step.
Phase B (cache): C conversations x 2 turns at concurrency 4, to see warm-turn TTFB under cache pressure.
Prints medians: TTFB, e2e tok/s per request, aggregate tok/s, wall per phase."""
import asyncio,json,random,statistics as st,sys,time,urllib.request,threading
URL="http://127.0.0.1:8090/v1/chat/completions"; MODEL="glm-5.3-flash-rtx"
S=int(sys.argv[1]) if len(sys.argv)>1 else 4; T=int(sys.argv[2]) if len(sys.argv)>2 else 6; BASE=int(sys.argv[3]) if len(sys.argv)>3 else 30000
C=int(sys.argv[4]) if len(sys.argv)>4 else 40
KW="def class return import for while if else try except async await self None True False print len range dict list str int open read write json yaml path file error warn info debug test assert fixture mock patch raise".split()
def code(rng,n):
    out=[]
    while len(out)<n:
        out.append(f"{rng.choice(KW)} {rng.choice(KW)}_{rng.randint(0,999)}({rng.choice(KW)}={rng.randint(0,99)}):  # L{rng.randint(1,9999)} {rng.choice(KW)} {rng.choice(KW)}")
    return "\n".join(out)
def req(messages,max_tokens=200):
    body={"model":MODEL,"messages":messages,"max_tokens":max_tokens,"temperature":0.7,"top_k":20,"stream":True}
    r=urllib.request.Request(URL,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t0=time.time(); ttfb=None; toks=0; text=""
    with urllib.request.urlopen(r,timeout=600) as resp:
        for line in resp:
            if not line.startswith(b"data:"): continue
            if ttfb is None: ttfb=time.time()-t0
            s=line[5:].strip()
            if s==b"[DONE]": break
            try: d=json.loads(s)
            except: continue
            delta=d["choices"][0].get("delta",{}) if d.get("choices") else {}
            c=delta.get("content") or delta.get("reasoning_content") or ""
            if c: toks+=1; text+=c
    dur=time.time()-t0
    return dict(ttfb=ttfb or dur,dur=dur,chunks=toks,text=text)
def conversation(sid,turns,base_words,results,stagger):
    time.sleep(stagger)
    rng=random.Random(1000+sid)
    msgs=[{"role":"system","content":"You are a coding agent. Reply with the single next shell command or file edit, under 150 words."},
          {"role":"user","content":"Repository snapshot:\n"+code(rng,base_words//9)+"\nTask: fix the failing test. Start."}]
    for t in range(turns):
        r=req(msgs); r.update(sid=sid,turn=t); results.append(r)
        msgs.append({"role":"assistant","content":r["text"][:1200] or "ok"})
        msgs.append({"role":"user","content":"Tool result:\n"+code(rng,170)+"\nContinue."})
def phase(name,n_conv,turns,base_words,conc):
    results=[]; threads=[]; t0=time.time(); sem=threading.Semaphore(conc)
    def run(sid,stag):
        with sem: conversation(sid,turns,base_words,results,stag)
    for i in range(n_conv):
        th=threading.Thread(target=run,args=(i,(i%conc)*0.7)); th.start(); threads.append(th)
    for th in threads: th.join()
    wall=time.time()-t0
    ttfb0=[r["ttfb"] for r in results if r["turn"]==0]; ttfbw=[r["ttfb"] for r in results if r["turn"]>0]
    e2e=[r["chunks"]/r["dur"] for r in results if r["dur"]>0]
    print(f"{name}: reqs={len(results)} wall={wall:.0f}s | TTFB cold med {st.median(ttfb0):.2f}s | TTFB warm med {st.median(ttfbw):.2f}s p90 {sorted(ttfbw)[int(0.9*len(ttfbw))-1]:.2f}s | e2e chunks/s per-req med {st.median(e2e):.0f} | aggregate chunks/s {sum(r['chunks'] for r in results)/wall:.0f}")
print(f"config: S={S} T={T} BASE_words={BASE} C={C}")
phase("A agents",S,T,BASE,S)
phase("B cache ",C,2,BASE//2,4)
