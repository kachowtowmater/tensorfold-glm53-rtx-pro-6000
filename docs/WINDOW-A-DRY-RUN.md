# Window A — dry run, 2026-10-06 23:57 → 2026-10-07 01:50 PT (113 min, hard stop 150), run by orchestrator on the operator's go
Blocked: all non-loopback clients on 8090/8096/8097 (+fangchen) for the window; removed at the end. Serve restored on x9, probes identical, public smoke 200/chat/stream/tool_call PASS.

| step | verdict | evidence (windowA-logs/) |
|---|---|---|
| 0 baseline-lite on live x9 | DONE | probes 222/330 329/392 310/367 242/424 @ 218.6/427.5/377.4/228.6 tok/s (x9-probe-server.txt); gsm8k-300 95.67% 287/300 (x9-gsm8k300.txt). NOTE: 98.8 / 162-164 in the plan were Mia's model-card numbers, never ours. |
| 1 convert dry run + kill/resume | PASS | FP8 base loads and quantizes directly (no dequant pre-pass); layers 0-2 (dense) in 3.5 min; ckpt/job.json written; after kill, `-r` printed "Resuming existing job / Resuming at: layers.1" (fell back to ckpt_old; lost 2 fast modules). MoE layer 3: 624 expert tensors in ~5 min → full convert ≈ 3-3.5 h, not 6-12. Uniform-4 reference errors saved (A1-uniform4-dense-err.txt, A1-layer3-expert-err.txt: median 0.00172, p90 0.0034). |
| 2 sc_measure → optimize → validate → allocation | NOT-YET-DETERMINED | first catch: `datasets` module missing (installed). Then sc_measure streams the 306 GB base at ~0.4 GB/s ≈ 13 min per pass and needs several passes; killed after 22 min with no output yet. Chain unproven. Window B must give B1' its own 1-3 h slot or find a faster load mode. |
| 3 x10 (patch 0909) on the 4-bit pack | PASS | autotune 8 min (new PATCHES_HASH); probes identical to x9: 222/330 329/392 310/367 242/424 @ 223.8/433.4/382.8/235.1 tok/s (x10-probe-server.txt). |
| 4 K=3 kernel path | SKIPPED | no mixed pack produced (step 2 unproven). |
| 5 drafter smoke | NOT RUN | out of time (steps 2 and 6 took the budget). |
| 6 top-k 8→6 | DONE, FINDING | quality holds: gsm8k-300 96.0% 288/300 (vs 95.67); probes coherent. Speed unchanged: 228.3/411.6/397.1/233.2 tok/s (vs 218.6/427.5/377.4/228.6), acceptance −2…−4 pts (217/335 316/396 306/352 232/451). −25% expert bytes/token bought ~0%. |
| 7 serve up x9 + public smoke | PASS | x9 up in 150 s; probes identical; https models 200, chat OK, 5 stream chunks, tool_call true. |

## What step 6 means (first-principles check)
Per decode step with the DFlash2 block (~9 tokens verified), each MoE layer touches ~60 distinct experts whatever k is; ≈24 GB of expert weights per step across 4 GPUs ≈ 3.7 ms of a ~20 ms step (~18%). So −25% expert bytes ≈ −4% step time, cancelled by the −3 pt acceptance drop. The same arithmetic puts the mixed-width quant (−17% expert bytes) at ≈ +3% tok/s, inside measurement noise, not the planned +10–15%. The expected lever is wrong at speculative shapes; the ~16 ms of non-expert step time (attention/KV, DSA indexer, drafter block pass, launch overhead, dense layers) is where the time goes. Not yet measured: the same k=8 vs k=6 comparison WITHOUT the drafter (needs a 2-line recipe passthrough for `--no-drafts`), which would confirm the bytes model at M=1.

## Exit rule
Window B as designed (quant) is NOT scheduled: steps 2 and 5 are not PASS, and the finding above removes the expected gain. Re-plan: measure the step-time breakdown first (D kernels + per-stage timing), then decide quant vs drafter vs scheduling work.
