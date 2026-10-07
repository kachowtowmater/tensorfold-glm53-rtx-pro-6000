import json,time,urllib.request
import os
base="http://127.0.0.1:%s/v1/chat/completions" % os.environ.get("PORT","8090")
def run(label, content, extra):
    body={"model":"glm-5.3-flash-rtx","messages":[{"role":"user","content":content}],"max_tokens":400,"temperature":0}
    body.update(extra)
    req=urllib.request.Request(base,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t=time.time(); d=json.load(urllib.request.urlopen(req,timeout=120)); el=time.time()-t
    u=d["usage"]; print("%-28s prompt=%d out=%d wall=%.1fs client=%.0f tok/s" % (label,u["prompt_tokens"],u["completion_tokens"],el,u["completion_tokens"]/el))
run("prose greedy", "Write a 500 word essay about the history of rivers in human civilization. Output only the essay.", {})
run("code greedy", "Write a Python module with a class LRUCache (get/put, O(1)) plus unit tests. Output only code.", {})
run("structured greedy", "Return a JSON array of 40 objects, each with fields id, city, country, population (integer), and a one-sentence note. Output only JSON.", {})
run("prose temp0.7 top_k20", "Write a 500 word essay about the history of rivers in human civilization. Output only the essay.", {"temperature":0.7,"top_k":20})
