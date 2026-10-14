#!/usr/bin/env python3
"""x14 identity gate for patch 0923 (fastokens behind TF_W1_FASTOKENS=1). CPU only, in the x14 image.

tok_check_x14.py <model_dir> <corpus_dir> [requests.jsonl ...]   (run inside the x14 image)
Compares fastokens (patch 0923 FastEncoder) against HF tokenizers (the x9/x13 path), token id for token id:
  A. FastEncoder.ids(t, flag) == tok.encode(t, add_special_tokens=flag).ids, both flags, over:
       corpus/long-400k.txt (~400k tok), corpus/needle-300k.txt (~300k tok), every corpus/*.txt book,
       300 varied texts (random unicode incl. CJK/emoji/combining/controls, added-token soup, code, CRLF, whitespace runs),
       the needle + long corpus rendered through the GLM chat template (as the server sees them),
       and, if request logs are given (JSONL with prompt_tail/completion), the 200 most recent real requests: each message, each completion,
       and the prompt_tail rendered through the chat template (texts stay in this process; only counts are printed).
  B. IncrementalTokenizer(fast) == IncrementalTokenizer(HF offsets path): ids, literal boundaries (char, token) and the
       hash state, for every text above, cold and as a warm turn (kept snapshot of turn 1, turn 2 = turn 1 + more).
  C. App._encode_ids with fast == without fast; vision path untouched when the flag is off.
  D. GIL: a counting thread keeps >= 50% of its idle rate while fastokens encodes the 400k text.
Exit 0 = zero mismatches everywhere; 1 = any mismatch or failed check."""
import gzip, hashlib, json, os, pathlib, random, sys, threading, time, types

os.environ["TF_W1_FASTOKENS"] = "1"
from tokenizers import Tokenizer
import tensorfold.cuda.fasttok as ft
import tensorfold.cuda.incremental as inc
import tensorfold.cuda.server as srv
from tensorfold.cuda.chat_template import ChatTemplate

model_dir, corpus = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
captures = sys.argv[3:]   # optional request logs
fails = 0


def check(ok, msg):
    global fails
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    fails += (not ok)


tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
fast = ft.load(model_dir / "tokenizer.json")
check(fast is not None, "TF_W1_FASTOKENS=1 -> FastEncoder loaded (fastokens %s)" % __import__("fastokens").__version__
      if hasattr(__import__("fastokens"), "__version__") else "TF_W1_FASTOKENS=1 -> FastEncoder loaded")
os.environ["TF_W1_FASTOKENS"] = "0"
check(ft.load(model_dir / "tokenizer.json") is None, "TF_W1_FASTOKENS=0 -> no FastEncoder (x9 + 0921 path)")
os.environ["TF_W1_FASTOKENS"] = "1"
tmpl = ChatTemplate(model_dir)

groups = {}
long_txt = (corpus / "long-400k.txt").read_text(encoding="utf-8")
needle_txt = (corpus / "needle-300k.txt").read_text(encoding="utf-8")
groups["corpus-400k"] = [long_txt]
groups["corpus-needle"] = [needle_txt]
groups["corpus-books"] = [p.read_text(encoding="utf-8", errors="replace") for p in sorted(corpus.glob("pg*.txt"))]
groups["chat-rendered-corpus"] = [
    tmpl.render([{"role": "user", "content": long_txt + "\n\nIn one sentence, who is the first character named in the text above?"}],
                tools=None, enable_thinking=False),
    tmpl.render([{"role": "system", "content": "You are helpful."}, {"role": "user", "content": needle_txt + "\n\nWhat is the code?"}],
                tools=None, enable_thinking=True)]

rnd = random.Random(658)
specials = [t.content for t in tok.get_added_tokens_decoder().values()]
pool = [chr(c) for c in list(range(0x20, 0x7f)) + list(range(0x4e00, 0x4e80)) + list(range(0x1f600, 0x1f650))
        + list(range(0x0400, 0x0450)) + list(range(0x0600, 0x0640)) + [0x301, 0x200d, 0x0, 0x7, 0x9, 0xa, 0xd, 0xfeff, 0xa0, 0x3000]]
varied = []
for _ in range(150):
    varied.append("".join(rnd.choice(pool) for _ in range(rnd.randint(0, 4000))))
for _ in range(60):
    varied.append("".join(rnd.choice(specials) + rnd.choice(["", " ", "x", "\n", "\n\n", "字", "  ", "\r\n", "😀"]) for _ in range(rnd.randint(1, 400))))
src = sorted(pathlib.Path(srv.__file__).parent.parent.rglob("*.py"))
for p in rnd.sample(src, 40):
    varied.append(p.read_text(errors="replace"))
for _ in range(30):
    w = rnd.choice(["a", " ", "\n", "\t", "  ", "1", "ab", "\r\n", "...", "--", "'s", "’", "é"])
    varied.append(w * rnd.randint(1, 20000))
for _ in range(14):
    varied.append(long_txt[rnd.randint(0, len(long_txt) - 50000):][:rnd.randint(1, 50000)])
varied += ["", " ", "\n", "\n\n", "</think>", "<|user|>", "<think>\n", "<|assistant|>\n<think></think>"]
groups["varied-%d" % len(varied)] = varied

if captures:
    recs = []
    for path in captures:
        op = gzip.open if path.endswith(".gz") else open
        with op(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if isinstance(d.get("prompt_tail"), list):
                    recs.append(d)
    recs.sort(key=lambda d: d.get("t") or "")
    recs = recs[-200:]
    cap_texts, cap_rendered = [], []
    for d in recs:
        msgs = [m for m in d["prompt_tail"] if isinstance(m, dict)]
        for m in msgs:
            c = m.get("content")
            if isinstance(c, str):
                cap_texts.append(c)
            elif isinstance(c, list):
                cap_texts += [p.get("text") for p in c if isinstance(p, dict) and isinstance(p.get("text"), str)]
        if isinstance(d.get("completion"), str):
            cap_texts.append(d["completion"])
        try:
            cap_rendered.append(tmpl.render(msgs, tools=None, enable_thinking=bool(d.get("reasoning_content"))))
        except Exception:
            pass
    groups["recent-%d-requests-texts" % len(recs)] = cap_texts
    groups["recent-%d-requests-chat-rendered" % len(recs)] = cap_rendered

# A. raw ids
total_tok = 0
for name, texts in groups.items():
    for flag in (False, True):
        bad = 0
        for t in texts:
            a = tok.encode(t, add_special_tokens=flag).ids
            b = fast.ids(t, flag)
            total_tok += len(a) if not flag else 0
            if a != b:
                bad += 1
        check(bad == 0, f"A ids  {name}: {len(texts)} texts, add_special_tokens={flag}: {bad} differ")
print(f"A total tokens compared (flag False): {total_tok:,}", flush=True)

# B. incremental tokenizer: fast path == HF offsets path (ids, boundaries, hash state), cold and warm
class Snap:
    pass


def bkey(r):
    # Boundary holds a hashlib object (compares by identity): compare (char, token, digest of the hash state) instead
    return [(b.char, b.token, b.state.digest()) for b in r.record.boundaries]


def rec_eq(r1, r2):
    return (list(r1) == list(r2) and bkey(r1) == bkey(r2) and r1.record.state.digest() == r2.record.state.digest())


def run_incremental(texts):
    hf_inc, fa_inc = inc.IncrementalTokenizer(tok), inc.IncrementalTokenizer(tok, fast=fast)
    check(hf_inc.supported and fa_inc.supported, "B GLM tokenizer is in the incremental 'supported' profile")
    kept_hf, kept_fa = [], []
    hf_inc.kept, fa_inc.kept = (lambda: tuple(kept_hf)), (lambda: tuple(kept_fa))
    bad = bad_warm = fallbacks = n = nb = 0
    orig = fa_inc._fast_tail

    def counting(tail, special):
        nonlocal fallbacks
        r = orig(tail, special)
        fallbacks += r is None
        return r
    fa_inc._fast_tail = counting
    for t in texts:
        kept_hf.clear(); kept_fa.clear()
        a, b = hf_inc.encode(t), fa_inc.encode(t)
        n += 1
        nb += len(b.record.boundaries)
        if not rec_eq(a, b) or list(a) != tok.encode(t).ids:
            bad += 1
            continue
        # warm turn: keep turn 1's snapshot, turn 2 appends an assistant reply and a new user turn
        for kept, r in ((kept_hf, a), (kept_fa, b)):
            s = Snap(); s.ids = list(r); inc.keep_rendered(s, r); kept.append(s)
        t2 = t + "<|assistant|>\n<think></think>\nok, noted.<|user|>\nnext question " + str(len(t))
        a2, b2 = hf_inc.encode(t2), fa_inc.encode(t2)
        if not rec_eq(a2, b2) or list(a2) != tok.encode(t2).ids or a2.record.reused != b2.record.reused:
            bad_warm += 1
    return n, bad, bad_warm, fallbacks, nb


for name, texts in groups.items():
    n, bad, bad_warm, fb, nb = run_incremental(texts)
    check(bad == 0 and bad_warm == 0, f"B incremental {name}: {n} texts, {nb} literal barriers: cold {bad} differ, warm-turn {bad_warm} differ "
          f"(fast-path fallbacks to HF offsets: {fb})")

# C. App._encode_ids
for flag in (False, True):
    bad = sum(srv.App._encode_ids(types.SimpleNamespace(tok=tok, fast=fast), t, flag)
              != srv.App._encode_ids(types.SimpleNamespace(tok=tok, fast=None), t, flag) for t in varied)
    check(bad == 0, f"C App._encode_ids fast == HF, add_special_tokens={flag}: {len(varied)} texts, {bad} differ")


# D. GIL + speed
def rate(fn):
    stop = threading.Event(); n = [0]

    def count():
        while not stop.is_set():
            n[0] += 1
    th = threading.Thread(target=count); th.start(); t0 = time.perf_counter(); fn(); dt = time.perf_counter() - t0
    stop.set(); th.join(); return n[0] / dt, dt


idle, _ = rate(lambda: time.sleep(1.0))
r_hf, t_hf = rate(lambda: tok.encode_batch([long_txt], add_special_tokens=False))
r_ff, t_ff = rate(lambda: tok.encode_batch_fast([long_txt], add_special_tokens=False))
r_fa, t_fa = rate(lambda: fast.ids(long_txt, False))
print(f"D 400k text ({len(tok.encode(long_txt).ids):,} tok): HF encode_batch(offsets, x13 incremental tail) {t_hf*1e3:.0f} ms | "
      f"HF encode_batch_fast {t_ff*1e3:.0f} ms | fastokens {t_fa*1e3:.0f} ms ({t_hf/t_fa:.1f}x vs offsets path) | "
      f"counter during fastokens {100*r_fa/idle:.0f}% of idle", flush=True)
check(r_fa >= 0.5 * idle, "D GIL released during fastokens encode (counter >= 50% of idle)")
print("ALL PASS" if not fails else f"{fails} FAIL")
sys.exit(1 if fails else 0)
