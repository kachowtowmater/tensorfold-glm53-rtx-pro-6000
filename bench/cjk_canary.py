import json,urllib.request,sys
base="http://127.0.0.1:8090/v1/chat/completions"
prompts=["请用三句话介绍长江的历史意义。","写一首关于秋天的七言绝句。","解释什么是机器学习，用通俗的中文。","列出五种中国传统节日及其习俗。","把这句话翻译成英文：今天天气很好，我们去公园散步吧。","用中文写一段关于咖啡与茶的比较。","请描述北京故宫的建筑特点。","写一个关于程序员和猫的小故事，两百字。","解释一下量子计算的基本概念。","请给出一份简单的红烧肉做法。","什么是人工智能的对齐问题？","用中文总结《三国演义》的主要情节。","写一封给朋友的中文邮件，邀请他周末来家里吃饭。","解释一下中文里‘成语’的来源，并举三个例子。","请介绍一下杭州西湖的十景。","描述一下中国高铁的发展。","请用中文解释什么是区块链。","写一段关于春节回家的感想。","用中文列出学习编程的五个建议。","请解释‘道可道，非常道’的含义。"]
bad=0; total=0; outs=[]
for p in prompts:
    body={"model":"glm-5.3-flash-rtx","messages":[{"role":"user","content":p}],"max_tokens":200,"temperature":0}
    req=urllib.request.Request(base,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    d=json.load(urllib.request.urlopen(req,timeout=120)); t=d["choices"][0]["message"]["content"]
    n=t.count("�"); bad+=n; total+=len(t); outs.append({"prompt":p,"fffd":n,"len":len(t),"head":t[:60]})
print(json.dumps({"prompts":len(prompts),"fffd_total":bad,"chars_total":total,"samples":outs[:3]},ensure_ascii=False,indent=1))
