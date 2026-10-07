---
title: TensorFold × GLM-5.3-Flash on mcqueen — program retrospective (2026-10-05 → 10-07)
summary: Everything we tried to make GLM-5.3-Flash faster on 4× RTX PRO 6000 under TensorFold, with the measured result of each attempt (adopted, rejected, parked), the finding that closed the program (expert weight bytes are ~18% of a speculative decode step, so quant and top-k cannot move raw speed), the artifacts left on the shelf, and how to resume. Written at the operator's request so the negative results stay usable, like the InfLLM record.
type: research
status: closed (program stopped 2026-10-07 09:40 PT at the operator's call)
updated: 2026-10-07
cards: default#1177–#1240 (board `default`)
---

# 1. One-paragraph verdict

Raw single-stream decode of GLM-5.3-Flash on this box did **not** move: 223 tok/s prose / 433 code before and after two days of work, which is the Aevonix recipe's own number and the best published figure for this card class. What moved is the **agent path**: turn times roughly halved under 4–8 agents, image-turn TTFT 2.87 s → 0.12 s, the 55%-of-prefill cache-miss class fixed, zero evictions. The closing finding is the reason raw speed is stuck: under speculative decoding each MoE layer touches ~60 distinct experts per step whatever we do to the weights, so expert bytes are ~18% of a ~20 ms step; cutting them 25% (top-k 8→6) changed speed 0%, and the planned mixed-width quant (−17%) projects to ≈+3%. The other ~82% of the step was not profiled (Window P parked). Everything below is measured, with the raw file named.

# 2. Setup (what every number below was measured on)

| item | value |
|---|---|
| box | mcqueen: 4× RTX PRO 6000 Blackwell Max-Q (96 GB each, PCIe, TP4), 250 W cap (settled), GPU 0 runs 88–92 °C under load (slot airflow, #1181) |
| engine | TensorFold v0.6.5 @609ca419 (ashhart; Python + torch-JIT CUDA kernels; **Python engine frozen upstream, dev moved to a Zig engine, issue #286**) |
| recipe | Aevonix `GLM-5.3-Flash-EXL3-4x-RTX-PRO-6000-TensorFold` @a190785: 86 patches, base `nvcr.io/nvidia/pytorch` sha256:2140e69… |
| weights | Mia-AiLab GLM-5.3-Flash-EXL3-4bpw-TensorFold (164 GB; EXL3 4-bit mcg routed experts, dense quantized at load, FP8 KV) |
| drafter | incoai DFlash2 (block drafter, ~9 tokens verified/step), policy `fnc7:0.3` |
| our image lineage | `tensorfold-glm53:1.1.0` → x1 (+commdata 0900) → x9 (+0910 vision-prefix) = **serving** · x10 (+0909 mixed-bit load, built, not serving) · x11 (profiling, in build) |
| measurement | `/tmp/tfprobe2.py` 4 probes (prose greedy, code greedy, JSON greedy, prose t0.7/top_k20; server-side tok/s + accepted/drafted from rank0.log, ±1–2%) · `/tmp/agentsim.py 4 6 30000 40` (4 agents × 6 turns on 58k-token repos + 40-conversation churn, ±10–15% per cell) · `llm_decode_bench.py --test-profile gsm8k` · gate log (`docker logs model-gate`) for real traffic |
| rules kept | fangchen (100.80.249.86) firewalled during measured runs; Eric's public endpoint smoke-tested after every serve change; liveness off while testing |

# 3. Baselines

| | SGLang kpool-0909 (Sept) | Aevonix recipe as shipped (10-05 trial) | **x9 (ours, 2026-10-07 01:50, idle)** |
|---|---|---|---|
| single-stream prose / code / JSON tok/s | 193 C1 (prose) | 223 / 441 / 383 | **222.6 / 432.9 / 383.1** |
| acceptance prose / code | MTP ≈2.58 tok/round | 2.3 / 5.7 tok/round | 222/330 (67%) / 329/392 (84%) |
| real agent traffic e2e tok/s @ conc 1/2/3/4 | 66 / 51 / 47 / 45 | 137 / 90 / 84 / 68 | (same engine) |
| agent turn duration @ conc 1/2/4/6/8 (gate log) | 1.5 / 2.0 / 3.2 / 4.4 / 5.8 s | 1.1 / 0.7 / 1.6 / 2.4 / 2.5 s | — |
| agentsim 4-agent warm turn / p90 | — | 2.9 / 3.8 s (R0 defaults) | **2.9 / 3.8 s** (before.json); best paired run 2.5 / 3.5 (R6) |
| agentsim churn warm turn / p90 | — | 5.8 / 10.0 s | 5.5 / 9.8 s; best paired 3.1 / 8.8 (R6) |
| gsm8k | — | model card 98.8 (Mia's harness, not comparable) | **96.51% full 1319 (Wilson 95.4–97.4)**; 300-item paired subset 95.67% |
| KV pool | 3.28× concurrency @524k | 5.1M tokens FP8 @1M window | same |
| image-turn TTFT (28k prefix) | — | 2.87 s every turn | **0.12 s** (turns 2–3) |

Full baseline with every command: `baseline-x9-2026-10-07.json` (card #1223, verified DONE).

# 4. Everything tried, in order, with the number that decided it

## 4.1 Trial and tuning rounds (10-05, cards #1177 #1178)
| what | result | verdict |
|---|---|---|
| TF trial vs SGLang (2 windows) | server-side C1 code 441 vs SGLang 193 C1; real-traffic e2e 137 vs 66 @conc 1; TTFB 0.09 s vs 1.37 s | TF adopted (Charles: "TF is on the table… make it better") |
| R4: MULTI_WINDOW 64 / GRAPH_STEP 4 / PREFILL_ROWS 8192 (+CE_ARENA 320) / FILL_ROWS 4096 | 4-agent warm turn 2.9→2.7 s, TTFT p90 0.98→0.52 s, churn cold 12.2→8.6 s | adopted |
| R5: DFLASH_POLICY f10 (deeper fixed drafts) | acceptance 54%→26%, every cell worse | rejected |
| R6: + CACHE_ENTRIES 64 | 4-agent 2.5/3.5 s, churn warm 3.1 s, agg 71 | adopted (later 128: 0 evictions / 1,077 requests) |
| T1 autotune prefill buckets to 8192 rows | 11 q4_prefill shapes +8.6…+23.5%, exl3_prompt +4.1% (kernel-level; e2e inside noise) | kept |
| T2 TF_GLM_COMM=ipc | prose 223→134, code 428→319 | rejected |
| Round-3 sweep: FILL_ROWS 8192, MULTI_WINDOW 32/2, PARALLEL 12, L2PF off, PRIORITY off, **KV bf16**, PREFILL_ROWS 16384, `fcost7:noisy` | all noise or worse (KV bf16 −2–4%; fcost JSON 300 vs 381) | rejected |
| K1 TF_GLM_CLEAR_THINKING=0 | untestable without thinking traffic | open (#1182) |

## 4.2 Patch stack experiments (10-06)
| image | patch | result | verdict |
|---|---|---|---|
| **x1** | 0900 commdata2338 engine patch (GPU_SAMPLE_FULL, SAMPLE_SKIP_FUTILE, BURST_PREFIX, CACHE_REASONS) | explicit top_k=-1 / top_p 1.0: **24 → 188 tok/s** (CPU-sampler collapse fixed); 4 cold agents sharing a 40k prefix: TTFT 6.3–12.4 s → 3.3 s ×4; probes +2–5% | **adopted** |
| x2 | 0901 Mia sliced fill (FILL_BUDGET_MS 200) | TTFT +30–50% worse, churn turn no gain (fills already fast on 96 GB) | rejected (P1 window 1) |
| x3 | 0902 PR #387 port | worker ported into the wrong family (qwen4_exp); flags-off tree 35–45% slower | void |
| x4 | **0903 TF_GLM_DECODE_WAIT_MS** (our first custom scheduling patch) | identity holds; churn warm turn 3.0→5.5–5.7 s, decode/request 47→30 | rejected; **P1 line stopped (3 windows)** |
| x5 | 0904 PDL on EXL3 decode kernel (jayleaton 0580 port) | no gain on microbench; Aevonix 0047 already had the load width | rejected |
| v2 | **HC_SPLIT_MIN_ROWS 512→2048** (config) | churn cold TTFT 5.9→5.2 s, consistent on every cell, probes identical | **adopted** |
| x6 | 0905 MFILLR v1 (mixed chunk+decode, design #1200) | G1 identity PASS; G2 latency gates FAIL | rejected |
| x8 | 0908 MFILLR-E (co-scheduled expert passes + L2 prefetch, design #1207) | identity PASS; gates FAIL | rejected; D2 shelved (needs a CUDA mixed-row expert kernel) |
| — | 0907 kept-views | refuted (artifact) | dropped |
| **x9** | **0910 vision-prefix resume** (our first custom TF edit in production) | image prompts bypassed the kept-state cache (632 req/day, 80% of prefill seconds): turn-2/3 TTFT 2.87 s → **0.12 s**, cached 27,968/28,058, replies identical; mutation proof = x1 red | **adopted** |
| x10 | 0909 mixed-bit EXL3 load (bits per tensor from trellis shape) + autotune K enumeration | loads the 4-bit pack with identical probes and tok/s (Window A step 3) | built, verified, on the shelf |

## 4.3 Analysis work that fed the above
- Knob inventory: 369-line matrix of TF / Aevonix / Mia / commdata knobs, plus what llama.cpp / SGLang / vLLM expose that TF lacks (`deliverables/1177-knob-matrix.md`, `1184-forks-and-knobs.md`, `1185-tf-gaps.md`, `1199-port-candidates.md`).
- Layer checks H2/H3 and the L1 (memory) report via the `compare-semantic-path` tool (codemap paths + static counts + Jev score): `deliverables/1202-l1-report.md`, `1203-/1204-layer-check.md`, `../../tools/compare-semantic-path.md`.
- Jev as a change-direction predictor: 10/15 direction accuracy vs 6/15 for always-"noise", Brier 0.097 vs 0.089 for always-"noise" → **WEAK**, not a gate (#1198).
- `cache_reason` log analysis found the vision bypass (4,365 requests/day: vision 632 requests = 4,424 of ~5,550 prefill seconds) → 0910 and `models.yml` GLM entries text-only.
- Drafter-data capture at model-gate (v5.17 → v5.18: tool_calls + reasoning captured, Eric excluded by construction; ~10.5k records by 10-07 09:30, 74 unique tool-call records). Known gaps: lss-collector probe records (#1224, fixed by exclusion), 502-on-complete-streams (#1229), omp traffic bypasses the gate via llm-lb (#1233).

## 4.4 Window A dry run (10-07 00:00–01:50 PT, serve down 113 min) — `windowA-2026-10-07.md`
| step | result |
|---|---|
| convert dry run + kill/resume | exllamav3 16a4979 quantizes the **FP8** base directly (no 640 GB pre-pass); `-r` resumes (`Resuming at: layers.1`); dense layers 0–2 in 3.5 min, a MoE layer in ~5 min → **full convert ≈ 3–3.5 h** (plan said 6–12); uniform-4 reference errors saved |
| sc_measure → optimize → validate | `datasets` module missing (fixed); sc_measure streams 306 GB at 0.4 GB/s ≈ 13 min/pass × several passes; killed at 22 min, chain unproven |
| x10 on the 4-bit pack | probes identical to x9 (222/330 329/392 310/367 242/424), tok/s 224/433/383/235 |
| **top-k 8→6** (Goddard's free knob) | quality held: gsm8k-300 **96.0% vs 95.67%**; speed **unchanged** 228/412/397/233 vs 219/428/377/229; acceptance −2…−4 pts |
| drafter smoke | not run (time) |
| restore | x9 up in 150 s, probes identical, public smoke PASS |

**The arithmetic that closed the quant line.** Per step ~9 tokens verified × top-8 ≈ 60 distinct experts per MoE layer × ~9.4 MB (4-bit) × 42 layers ≈ 24 GB read across 4 GPUs ≈ 3.7 ms of a ~20 ms step (~18%). −25% bytes (top-k 6) → −4% step, cancelled by the acceptance drop; −17% bytes (3.65 bpw mixed quant) → ≈ +3% tok/s, inside noise. Dettmers' quant advice (target 3.6–3.7 bpw, router 8-bit, lm_head ≥6, pin under-calibrated experts to 4, abort rules) is in the quant runbook for the day weights do matter (no drafter, or high concurrency).

## 4.5 What was NOT measured (and would be next)
Where the other ~16 ms per decode step go. Panel expectation (Dan, Zheng, He): TP4 all-reduce over PCIe (~90 collectives/step, 2–9 ms) and ungraphed 9-token verify steps (~1000 eager launches ≈ 10 ms) are the prime suspects; DSA indexer/KV growth with context and sampler syncs next. Stop signal for any engine rewrite: CUDA graphs cover >85% of the step and GPU busy >85% (then only TP layout/collective fusion remains). Window P (#1240) would produce the (2k/30k/90k) × (rows 1/4/8) × (drafter on/off) table in 2 h; parked by Charles (odds of a ≥20% software lever ≈ 30%).

# 5. Lessons (the ones that generalize)
1. **Measure the bottleneck class before buying bytes.** Two nights went to weight-size levers before a 30-minute top-k test showed weights are ~18% of the step under speculative decoding. The first window should have been the step-time breakdown.
2. **Speculative decoding changes which optimizations work.** Expert-bandwidth tricks (quant, pruning, top-k) assume one token per step; with a 9-token block they fade. Drafter acceptance (tokens per step) is the lever that survives.
3. **Model-card numbers are not our numbers.** "98.8 gsm8k / 162 of 164 humaneval" were Mia's harness; ours is 96.51 full / 95.67 subset, and humaneval has no harness here (#1237). Gates must cite our own baseline.
4. **The agent path pays; raw decode did not.** Cache misses (vision bypass, evictions), rank-split thresholds, sampler collapse, cold-prefix bursts: each was a real, reproducible 2×–20× on a real cell. Single-stream decode on a competent recipe is at its ceiling.
5. **Dry-run windows catch the "whoopsie".** A missing `datasets` module and a 13-min-per-pass measurement would have surfaced at hour 10 of a long window; the 2 h dry run cost 113 min and also returned the top-k result and a 3× faster convert estimate.
6. **Verifiers were worth every send-back.** Six send-backs on one baseline card caught a dropped 55% cache-miss class, a wrong p95, a missing third run, an unexecuted CJK canary and a scratch clone left behind. The mutation tests also caught a validator that accepted 16-bit experts and a window script whose hard clock never fired.
7. **GLM workers + a hard host fence need a door.** Rule N (no worker ssh to mcqueen after the fork-loop incident) blocked 6 of 7 workers; `mcq-run`/`mcq-serve` (worker writes `mcq/run.sh`, lead runs it under a deny list, posts the log on the card) made them productive at ~1 round trip per scripting slip (shell function recursion, unexpanded `~`, relative paths after `cd`, loop variable `path` clobbering PATH in zsh).
8. **TensorFold's Python engine is frozen upstream.** Custom patches (0903–0910) are ours to maintain; the maintainer moved to Zig. Treat the recipe as a fork.

# 6. On the shelf (how to resume)
| artifact | where | state |
|---|---|---|
| serving recipe x9 | mcqueen `~/tf/recipe-x9` (d1ff3a9), image `tensorfold-glm53:1.1.0-x9`, `scripts/local.sh` = recipe v3 + `TF_GLM_VISION_PREFIX=1` | **live on :8090**; rollback `~/tf/recipe-x1` |
| mixed-bit load build x10 | `~/tf/recipe-x10`, image `…-x10`, patch `patches/extra/0909-mixed-bits.patch` | built, probes identical to x9 |
| profiling build x11 + `--no-drafts` passthrough | `~/tf/recipe-x11` (card #1238, in progress), patch 0911 | prep |
| exllamav3 pipeline | `~/exl3/{venv,exllamav3@16a4979,tools/validate_recipe.py,tools/convert_monitor.py,cal/cal.safetensors}`; runbook `quant-runbook-2026-10-06.md` + Corrections | proven in Window A; convert ≈ 3–3.5 h |
| drafter pipeline | `~/specforge/{venv,src@53398a8,tools/template_diff.py,split_check.py}`, harness `deliverables/1210-README-training.md`, capture `~/model-gate/capture/` | plumbing proven; real training waits for ≥10k post-v5.18 records |
| step-time runner | `~/exl3/bench/steptime.py` (#1239) | handed in, unverified |
| window runbook | `~/tf/window/window.sh` (#1222; hard-clock fix in progress) | DRY-tested |
| pre-flight gate | chick `(internal task dir) 1225/preflight.sh` (13/14 PASS) | reusable |
| baseline | `baseline-x9-2026-10-07.json` (this dir) | the BEFORE half of any future table |
| measurement scripts | mcqueen `/tmp/{tfprobe2.py,quietprobe.sh,agentsim.py,burstprobe.py,visionprobe.py}` (copy into a repo before they vanish) | working |

Resume path if raw speed is wanted again: run Window P (2 h, #1240) with x11 + steptime.py → read the table → pick drafter retrain (flat step time), by-load draft depth (concurrency), kernel swap (named kernel), or collective fusion (PCIe floor). Resume path for the 170HX box: same recipe family; run the Morrowmake adaptive-depth fork natively there (#1234).

# 7. Evidence index
- Daily record with every table: `2026-10-05.md` (trial, real-load, tuning rounds, x1–x9, vision bypass, Jev, capture).
- Window A: `windowA-2026-10-07.md`, `windowA-logs/` (probe server lines x9/x10/top-k6, gsm8k-300 summaries, convert resume excerpt, uniform-4 error references).
- Baseline: `baseline-x9-2026-10-07.json`.
- Plans: `../../plans/tf-glm53-speed-program.md` (goal, invariants, the operator's hierarchy), `../../plans/tf-glm53-gpu-windows.md` (approved window plan, execution log, close-out).
- Runbooks: `quant-runbook-2026-10-06.md`, `drafter-runbook-2026-10-06.md`.
- Worker research: `deliverables/` (knob matrix, forks and knobs, TF gaps, port candidates, mixed-chunk and mixed-expert designs, L1 report, layer checks, drafter plan, training README).
- Ledger rows: `../../infrastructure/loadout-ledger.md` 2026-10-05/06 (APPLIED ×4, TRIED ×1).
- Cards: `tb default show <id>` for #1177–#1240; raw worker outputs under chick `(internal task dir) <id>/`.
