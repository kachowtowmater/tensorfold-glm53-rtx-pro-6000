# Rejected experiments (kept because a reject with a number is a result)

All measured on the same box with the same scripts (`bench/`), paired against the serving control of the day. Patches are ours unless noted; they apply on the x1 tree (Aevonix 86 + commdata 0900) and are kept as-is.

| patch / change | idea | measured | why rejected |
|---|---|---|---|
| `0903-decode-wait-budget.patch` (`TF_GLM_DECODE_WAIT_MS`) | yield a decode round when decoders waited past a budget during fills | greedy output identical; churn warm turn 3.0 → 5.5–5.7 s, decode/request 47 → 30 (100 ms and 300 ms) | forced decode rounds slow every other stream's small fills |
| `0904-exl3-dec-expert-loads.patch` (PDL on the EXL3 decode kernel, port of jayleaton 0580) | programmatic dependent launch on `dec_kernel` | no gain on the kernel microbench; Aevonix 0047 already has the `nc` load width | no measurable effect |
| `0905-mixed-chunk-v1.patch` (MFILLR, `TF_GLM_MIXED_CHUNK`) | run a prefill chunk and decode rows in one iteration so decoders never stall behind fills | identity PASS; 4-agent and churn latency gates FAIL | the stall was not the cost; see the retrospective |
| `0908-mfillr-e.p0.patch` (MFILLR-E, `TF_GLM_MIXED_CHUNK=eager`) | co-schedule chunk + decode expert passes with L2 prefetch of the decode window's experts | identity PASS; gates FAIL | same; needs a CUDA mixed-row expert kernel to pay off |
| Mia `0062` sliced fill (`FILL_BUDGET_MS 200`, not ours, not shipped) | trade new-prompt TTFT for decoder continuity | TTFT +30–50% worse, churn turn no gain | fills are already fast on 96 GB cards |
| `TF_GLM_COMM=ipc` | CUDA IPC exchange instead of NCCL | prose 223 → 134, code 428 → 319 tok/s | 40% slower |
| `TF_GLM_DFLASH_POLICY f10` / `fcost7:noisy` | deeper fixed drafts / cost-measured depth | acceptance 54% → 26% (f10); JSON 381 → 300 (fcost) | fewer accepted tokens per step |
| KV `bf16`, `PREFILL_ROWS 16384`, `MULTI_WINDOW 32/2`, `FILL_ROWS 8192`, `PARALLEL 12`, L2 prefetch off, priority off | sweep | noise or worse (KV bf16 −2–4%) | nothing to adopt |
| top-k 8 → 6 (`num_experts_per_tok`, config only) | −25% expert reads per token | gsm8k-300 96.0% vs 95.7% (quality held); speed 0% (228/412/397/233 vs 219/428/377/229 tok/s) | expert bytes are ~18% of a speculative decode step |
| mixed-width EXL3 requant (3.65 bpw experts) | −17% expert bytes | projected ≈ +3% from the top-k result; pipeline proven (resume, FP8 load, 3–3.5 h convert) but not run | not worth a 12 h window on this box |

Numbers and context: `docs/RETROSPECTIVE.md`.
