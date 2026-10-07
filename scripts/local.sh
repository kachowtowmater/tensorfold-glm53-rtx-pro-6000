# mcqueen recipe on top of Aevonix defaults — default#1178, 2026-10-05 (orchestrator). Paired A/B vs defaults: 4-agent warm turn 2.9->2.5 s,
# TTFT p90 0.98->0.51 s, churn warm turn 5.8->3.1 s, decode/req +10-13%, acceptance unchanged. Rejected: DFLASH_POLICY f10 (accept 55->26%).
export SERVED_NAME="${SERVED_NAME:-glm-5.3-flash-rtx}" PORT="${PORT:-8090}"
export TF_GLM_MULTI_WINDOW="${TF_GLM_MULTI_WINDOW:-64}" TF_GLM_MULTI_GRAPH_STEP="${TF_GLM_MULTI_GRAPH_STEP:-4}"
export TF_GLM_PREFILL_ROWS="${TF_GLM_PREFILL_ROWS:-8192}" TF_GLM_CE_ARENA_MIB="${TF_GLM_CE_ARENA_MIB:-320}" TF_GLM_FILL_ROWS="${TF_GLM_FILL_ROWS:-4096}"
export TF_GLM_CACHE_ENTRIES="${TF_GLM_CACHE_ENTRIES:-128}"   # 2026-10-06: 0 evictions in 1,077 agent requests vs 40 before (cache_reason log)
# power cap: 250 W is the settled cap for this box (resets to 300 W on reboot): sudo nvidia-smi -pl 250
# x1 (default#1180, 2026-10-06): commdata2338 engine patch (patches/extra/0900). Measured vs recipe: explicit top_k=-1 24->188 tok/s;
# 4-agent shared-40k-prefix burst TTFT 6-12 s -> 3.3 s; probes/agentsim unchanged. Mia 0901 held in patches/extra-held (rebase pending, x2).
export TENSORFOLD_GPU_SAMPLE_FULL="${TENSORFOLD_GPU_SAMPLE_FULL:-1}" TENSORFOLD_SAMPLE_SKIP_FUTILE="${TENSORFOLD_SAMPLE_SKIP_FUTILE:-1}"
export TF_GLM_BURST_PREFIX="${TF_GLM_BURST_PREFIX:-1}" TF_GLM_CACHE_REASONS="${TF_GLM_CACHE_REASONS:-1}"
# 2026-10-06 (default#1178 line, fills-cost hypothesis): small warm-turn fills (~1k rows) no longer pay the rank split.
# Paired vs recipe: churn cold TTFT 5.9->5.2 s (run1) / 5.9 vs 6.4-6.9 (run2 vs 2 controls), 4-agent agg 186->195, churn agg 67->71; probes identical.
export TF_GLM_HC_SPLIT_MIN_ROWS="${TF_GLM_HC_SPLIT_MIN_ROWS:-2048}"
# 2026-10-06 x9 (default#1213): patch 0910 vision-prefix resume — image prompts keep/resume their text prefix at the grid point before the first image.
# Measured: 3-turn image conversation turn2/3 TTFT 2.87 s -> 0.12 s (cached 27,968/28,058), replies identical, probes identical. cache_reason=vision was 80% of prefill seconds on 10-05/06.
export TF_GLM_VISION_PREFIX="${TF_GLM_VISION_PREFIX:-1}"
