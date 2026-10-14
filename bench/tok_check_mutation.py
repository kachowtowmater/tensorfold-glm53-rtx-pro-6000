"""Mutation proof for tok_check_x14.py: break patch 0923 two ways; the gate must go red on the A and B lines.
tok_check_mutation.py A|B <model_dir> <corpus_dir>   (A: no added-token extraction on long inputs; B: literal barrier one token late)"""
import os, runpy, sys
import tensorfold.cuda.incremental as inc, tensorfold.cuda.fasttok as ft
mode = sys.argv.pop(1)
if mode == "B":   # literal barrier one token late
    o = inc.IncrementalTokenizer._fast_tail
    def m(self, t, s):
        r = o(self, t, s)
        return None if r is None or not r[1] else (r[0], [(c, i + 1) for c, i in r[1]])
    inc.IncrementalTokenizer._fast_tail = m
elif mode == "A":  # fastokens without added-token extraction on long inputs
    o = ft.FastEncoder.ids
    def m(self, text, special=False):
        return o(self, text, special) if len(text) < 100000 else self.tok.encode_ordinary(text).ids
    ft.FastEncoder.ids = m
sys.argv = ["tok_check_x14.py"] + sys.argv[1:]
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tok_check_x14.py"), run_name="__main__")
