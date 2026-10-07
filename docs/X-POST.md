# X post (copy as is; attach results/cover-black.png, results/cards-1-black.png, results/cards-2-black.png in that order; link goes last)

Handles, from the GitHub profiles: TensorFold author ashhart = @ashxhart · Mia-AiLab = @MiaAI_lab · turboderp = @turboderp_ · Aevonix, commdata2338, incoai list no X account (named in text).

## Full version (long post)

Two days of paired A/B on GLM-5.3-Flash with TensorFold on 4× RTX PRO 6000. Everything is now public: patches, knobs, bench scripts, every result including the ones that lost.

What improved (same box, same prompts, before → after):

• Image turns: a 28k-token conversation with an image re-prefilled the whole prefix every turn. Our vision-prefix patch keeps it. TTFT 2.87 s → 0.12 s, replies identical. That was 80% of all prefill seconds on our agent traffic.
• 4 cold agents sharing a 40k prefix: TTFT 6.3–12.4 s → 3.3 s each, turn 13 s → 4.2 s (burst-prefix sharing from commdata2338's engine patch).
• Sampler collapse on top_k=-1 / top_p 1.0: 24 → 188 tok/s (same patch).
• Cache evictions on agent traffic: 40 → 0 per ~1,000 requests (CACHE_ENTRIES 128).
• Small warm-turn fills no longer pay the 4-rank split (HC_SPLIT_MIN_ROWS 2048): churn cold TTFT 5.9 → 5.2 s.
• Tuned windows/graph step/prefill rows: 4-agent warm turn 2.9 → 2.5 s, TTFT p90 0.98 → 0.51 s, churn warm turn 5.8 → 3.1 s.
• Real agent traffic vs our old SGLang serve: turn time at 8 concurrent 5.8 → 2.5 s.
• Mixed-width EXL3 loading patch for TensorFold (per-tensor 3/4/5-bit experts), verified a no-op on normal packs.

What did not improve: raw single-stream decode. 223 prose / 433 code tok/s before and after, which is the Aevonix recipe's number. We measured why: cutting expert reads 25% (top-k 8 → 6) kept quality (gsm8k 96.0 vs 95.7) and changed speed 0%. Under speculative decoding the weights are ~18% of a decode step. So we parked the requant instead of spending a 12-hour window on a projected +3%.

Also in the repo: four of our own patches that lost (decode-wait, PDL, two mixed-chunk schedulers) with their numbers, the measured baseline (gsm8k full set 96.5%, CJK clean), and a dry-run log showing exllamav3 resumes a killed convert and quantizes the FP8 base directly (3–3.5 h for the full model, not the 6–12 we planned for).

Built on TensorFold by @ashxhart (Apache-2.0), the Aevonix 4× PRO 6000 recipe, @MiaAI_lab's EXL3 pack, commdata2338's engine patch, incoai's DFlash2 drafter, @turboderp_'s exllamav3. Thank you all; none of this exists without that work.

Repo: https://github.com/kachowtowmater/tensorfold-glm53-rtx-pro-6000

## Short version (one tweet + the three images)

GLM-5.3-Flash on TensorFold, 4× RTX PRO 6000: our agent-path overlay is public. Image-turn TTFT 2.87 → 0.12 s, cold 4-agent bursts 12.4 → 3.3 s, sampler collapse 24 → 188 tok/s, 0 cache evictions, 8-concurrent turns 5.8 → 2.5 s vs SGLang. Raw decode unchanged (223/433 tok/s), and we measured why: weights are ~18% of a spec-decode step. Patches, knobs, bench scripts, every result incl. the losers. Built on @ashxhart's TensorFold + the Aevonix recipe + @MiaAI_lab's pack.
https://github.com/kachowtowmater/tensorfold-glm53-rtx-pro-6000
