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

## Week of 2026-10-13 (x9 → x14), knobs and rebases, no patch files

Same box and scripts, paired against x9 in the same window. The co-run is a 400k-token cached session sending a request every 1 s (85 tokens out) while one short request decodes; its number is the short request's decode tok/s.

| change | idea | measured | why rejected |
|---|---|---|---|
| Aevonix recipe 1.2.0 (TF v0.6.6) with our x9 additions rebased onto it | track upstream | same as x9 at our knobs; its new defaults `TF_GLM_MULTI_WINDOW 128` / `TF_GLM_MULTI_GRAPH_STEP 8` cost the long-session co-run **−24%** | nothing gained over x9; we stay on the 1.1.0 pin |
| `TF_GLM_MULTI_WINDOW 32` | smaller decode window | C32 aggregate **−14%** | throughput loss at high concurrency |
| `TF_GLM_FILL_ROWS` / `TF_GLM_PREFILL_ROWS` 2048 / 4096 variants | smaller fill and prefill chunks | within noise; FILL 2048 + PREFILL 4096 gave a small first-token gain during a cold 400k prefill | too small to justify a knob change |
| `COPY_MAX 8` | more copy drafts | within noise | nothing to adopt |
| indexer (seg-score) grid 512 (part of Mia 0107) + TensorFold PR #421 port (`_take_over` without a clone) | less indexer and copy work per step | no measurable gain on a quiet serve | x14 carries the tokenizer change only |

**Methodology lesson.** Measuring with other users' traffic on the box produced fake 1.2–1.9× "wins" this week. Block outside traffic for the whole A/B window and run both arms inside it.

## Week of 2026-10-14/16 (x15 → x25), what lost while shipping the gains

Same box and scripts, paired against the control of the day. These are the candidates that did not make the release; the shipped ones are in the main README.

| change | idea | measured | why rejected |
|---|---|---|---|
| x16 A1a select one-pass (`0930`, `TF_GLM_SEL_ONEPASS=1`) | read a prompt chunk's top-512 pools in one pass over the scores instead of two | no gain on the measured paths | the two-pass path was already fast enough; patch ships flag-off |
| x16 B verbatim conversation history | keep every past turn's tokens verbatim instead of the trimmed kept state | no gain on TensorFold's kept-state cache | memory for nothing |
| x16 C stream rejection on 429 | reject overflow requests at the stream level | converts queueing delay into client errors | worse failure mode for the same load |
| x16 cache-share count (`0932`, `TF_GLM_CACHE_SHARE_PCT`) | size the kept-state count from a share of the pool's free memory | no measurable latency change at the tested shares | ships flag-off as a safety valve, not an adopt |
| `FILL_ROWS` 2048 / 1024 (after x15) | smaller fill chunks for smoother interleaving | long-prompt TTFT **+15% / +85%** | chunks are already the right size under time-credit fairness |
| GVR top-k port | GPU radix top-k from another stack | C1 **−9%**, 400k decode **−11%** | slower than the recipe's selection; superseded by the x24 fused kernel |
| IndexShare on Flash | share indexer state across ranks | lossy on GLM-5.3-Flash (output differs) | exactness is a hard requirement here |
| ForceP2P (`NCCL_P2P_LEVEL` overrides) | force peer-to-peer for every NCCL collective | no gain — the cards already do 54 GB/s P2P where it matters; the recipe disables P2P only where shared memory wins twice | nothing to adopt |
| Mixed decode+prefill forward pass | one forward for both classes | NO-GO: exactness forces dual kernel routes | see `0905`/`0908` above; the scheduler-side fixes (x21/x23) got the win without it |

Numbers and context: `docs/RETROSPECTIVE.md`.
