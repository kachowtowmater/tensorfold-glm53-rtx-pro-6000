#!/usr/bin/env python3
"""Two-turn image conversation: 40k-token text prefix + one image + question; turn 2 appends text. Reports TTFT/prefill per turn from the client (streaming first content) and the server's cache_reason."""
import json,random,struct,zlib,base64,time,urllib.request,sys
URL="http://127.0.0.1:8090/v1/chat/completions"; M="glm-5.3-flash-rtx"
def png(w=64,h=64):
    raw=b"".join(b"\x00"+bytes(v for x in range(w) for v in ((x*4)%256,(y*4)%256,128)) for y in range(h))
    def chunk(t,d): c=struct.pack(">I",len(d))+t+d; return c+struct.pack(">I",zlib.crc32(t+d)&0xffffffff)
    return b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",w,h,8,2,0,0,0))+chunk(b"IDAT",zlib.compress(raw))+chunk(b"IEND",b"")
img="data:image/png;base64,"+base64.b64encode(png()).decode()
rng=random.Random(99); W="repo file def class import return error test fix log json path read write".split()
prefix=" ".join(rng.choice(W) for _ in range(28000))
def turn(messages,label):
    body={"model":M,"messages":messages,"max_tokens":60,"temperature":0,"stream":True}
    r=urllib.request.Request(URL,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}); t0=time.time(); ttft=None; text=""
    with urllib.request.urlopen(r,timeout=600) as resp:
        for line in resp:
            if line.startswith(b"data:") and b'"content"' in line:
                if ttft is None: ttft=time.time()-t0
                try: text+=json.loads(line[5:])["choices"][0]["delta"].get("content") or ""
                except Exception: pass
    print(f"{label}: TTFT {ttft:.2f}s total {time.time()-t0:.2f}s reply={text[:40]!r}")
    return text
tag=sys.argv[1] if len(sys.argv)>1 else ""
m=[{"role":"system","content":"Tool manual:\n"+prefix},{"role":"user","content":[{"type":"text","text":"Describe the image in 5 words."},{"type":"image_url","image_url":{"url":img}}]}]
a=turn(m,f"{tag} turn1 (cold, image)")
m=m+[{"role":"assistant","content":a},{"role":"user","content":"Now list three colours you saw, comma separated."}]
turn(m,f"{tag} turn2 (same prefix+image, +text)")
m=m+[{"role":"assistant","content":"red, green, blue"},{"role":"user","content":"Reply with the word done."}]
turn(m,f"{tag} turn3 (+text)")
