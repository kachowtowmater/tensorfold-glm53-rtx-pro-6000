# our knobs on top of Aevonix defaults, 2026-10-05. Paired A/B vs defaults: 4-agent warm turn 2.9->2.5 s,
# TTFT p90 0.98->0.51 s, churn warm turn 5.8->3.1 s, decode/req +10-13%, acceptance unchanged. Rejected: DFLASH_POLICY f10 (accept 55->26%).
export SERVED_NAME="${SERVED_NAME:-glm-5.3-flash-rtx}" PORT="${PORT:-8090}"
export TF_GLM_MULTI_WINDOW="${TF_GLM_MULTI_WINDOW:-64}" TF_GLM_MULTI_GRAPH_STEP="${TF_GLM_MULTI_GRAPH_STEP:-4}"
export TF_GLM_PREFILL_ROWS="${TF_GLM_PREFILL_ROWS:-8192}" TF_GLM_CE_ARENA_MIB="${TF_GLM_CE_ARENA_MIB:-320}" TF_GLM_FILL_ROWS="${TF_GLM_FILL_ROWS:-4096}"
export TF_GLM_CACHE_ENTRIES="${TF_GLM_CACHE_ENTRIES:-128}"   # 2026-10-06: 0 evictions in 1,077 agent requests vs 40 before (cache_reason log)
# power cap: 250 W is the settled cap for this box (resets to 300 W on reboot): sudo nvidia-smi -pl 250
# x1 (2026-10-06): commdata2338 engine patch (patches/extra/0900). Measured vs recipe: explicit top_k=-1 24->188 tok/s;
# 4-agent shared-40k-prefix burst TTFT 6-12 s -> 3.3 s; probes/agentsim unchanged. Mia 0901 held in patches/extra-held (rebase pending, x2).
export TENSORFOLD_GPU_SAMPLE_FULL="${TENSORFOLD_GPU_SAMPLE_FULL:-1}" TENSORFOLD_SAMPLE_SKIP_FUTILE="${TENSORFOLD_SAMPLE_SKIP_FUTILE:-1}"
export TF_GLM_BURST_PREFIX="${TF_GLM_BURST_PREFIX:-1}" TF_GLM_CACHE_REASONS="${TF_GLM_CACHE_REASONS:-1}"
# 2026-10-06 (fills-cost hypothesis): small warm-turn fills (~1k rows) no longer pay the rank split.
# Paired vs recipe: churn cold TTFT 5.9->5.2 s (run1) / 5.9 vs 6.4-6.9 (run2 vs 2 controls), 4-agent agg 186->195, churn agg 67->71; probes identical.
export TF_GLM_HC_SPLIT_MIN_ROWS="${TF_GLM_HC_SPLIT_MIN_ROWS:-2048}"
# 2026-10-06 x9: patch 0910 vision-prefix resume — image prompts keep/resume their text prefix at the grid point before the first image.
# Measured: 3-turn image conversation turn2/3 TTFT 2.87 s -> 0.12 s (cached 27,968/28,058), replies identical, probes identical. cache_reason=vision was 80% of prefill seconds on 10-05/06.
export TF_GLM_VISION_PREFIX="${TF_GLM_VISION_PREFIX:-1}"
# 2026-10-13 x14: patches 0921 (tokenize without the GIL) + 0923 (fastokens 0.3.2; IMAGE_EXTRAS needs fastokens==0.3.2). 0 = x9 + 0921.
export TF_W1_FASTOKENS="${TF_W1_FASTOKENS:-1}"
# x15 (2026-10-14): prefill fairness = the 0900 time-credit arbiter. Decode keeps >= 0.5 of wall time while a prompt
# fills (x14 gave it ~0.15, counted by iterations). Paired A/B: other requests' decode during a 32K/128K/400K
# prefill 6.6x/15x/13x faster; the long prompt's own fill 1.8-2.2x slower.
export TF_GLM_FILL_CREDIT="${TF_GLM_FILL_CREDIT:-time}" TF_GLM_FILL_SHARE="${TF_GLM_FILL_SHARE:-0.5}"
# x17 (patch 0933): route a heavy client's id to the background class (--name-priority "id=class"). Foreground TTFT
# behind that loop: median 86 -> 46 ms, max 177 -> 53 ms (foreground decode -5%). Unset = every id normal class.
export TENSORFOLD_NAME_PRIORITY="${TENSORFOLD_NAME_PRIORITY:-}"
# x18 (patches 0940 + 0941): prompt-cache tiers — evicted prompts park to host RAM, then NVMe; a returning session
# restores its parked state instead of prefilling cold. Evicted-session restore: 400K 45 s -> 2.6 s, 1M 142 s -> 9.2 s,
# identical text, survives a restart. Budgets are per rank.
export TF_GLM_MULTI_PARK="${TF_GLM_MULTI_PARK:-1}"
export TF_GLM_DISK_CACHE="${TF_GLM_DISK_CACHE:-/cache/pcache}"   # local NVMe cache dir for parked prompts
export TF_GLM_DISK_CACHE_GIB="${TF_GLM_DISK_CACHE_GIB:-96}"
export TF_GLM_HOST_CACHE_GIB="${TF_GLM_HOST_CACHE_GIB:-40}"      # RAM tier; TF_GLM_PARK_STAGING_GIB adds 8 pinned
# x24 (patch 0980): the segmented decode selection through one fused CUDA kernel (TF_GLM_TOPK_V2=0 = x23 path).
# Selection per decode round 11-37x faster; 400K-context decode 195 -> 227 tok/s (+16%), bit-exact (92/92 GPU checks).
export TF_GLM_TOPK_V2="${TF_GLM_TOPK_V2:-1}"
# x25 (patch 0990): write-behind park — park copies into a preallocated pinned staging extent and hands the NVMe
# write to a background thread. On KV-pool overflow (9 x 400K parked prompts, two runs): worst decode gap
# 2.05-2.19 s -> 0.86-1.18 s, gap p99 0.43 -> 0.09 s, restores identical. Unset = the synchronous park.
export TF_GLM_PARK_ASYNC="${TF_GLM_PARK_ASYNC:-1}"
