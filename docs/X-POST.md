# X post — copy as is. Attach, in order: results/cover-black.png, results/cards-1-black.png, results/cards-2-black.png. Link last.

Handles were read from each GitHub profile. Accounts with no X handle are linked by URL instead.

---

We put GLM-5.3-Flash on TensorFold across 4× RTX PRO 6000 and spent two days of paired A/B making the agent path faster. Everything is public today: patches, knobs, bench scripts, every result, including the ones that lost.

Same box, same prompts, before → after:

• Image turns: a 28k-token conversation with a picture re-prefilled the whole prefix every turn. Our vision-prefix patch keeps it. TTFT 2.87 s → 0.12 s, replies identical. That was 80% of all prefill seconds on our agent traffic.
• 4 cold agents sharing a 40k prefix: TTFT 6.3–12.4 s → 3.3 s each, turn 13 s → 4.2 s.
• Sampler collapse on top_k=-1 / top_p 1.0: 24 → 188 tok/s.
• Cache evictions on agent traffic: 40 → 0 per ~1,000 requests.
• 4-agent warm turn 2.9 → 2.5 s, TTFT p90 0.98 → 0.51 s, churn warm turn 5.8 → 3.1 s.
• Real agent traffic, 8 concurrent, vs our old SGLang serve: turn time 5.8 → 2.5 s.
• Plus a TensorFold patch that loads mixed-width EXL3 packs (per-tensor 3/4/5-bit experts), verified a no-op on normal packs.

What did not change: raw single-stream decode, 223 prose / 433 code tok/s, the Aevonix recipe's number. We measured why instead of guessing: cutting expert reads 25% (top-k 8 → 6) kept quality (gsm8k 96.0 vs 95.7) and moved speed 0%. Under speculative decoding the weights are ~18% of a step. So the requant is parked, with the math in the repo.

None of this exists without the people whose work it sits on:

• TensorFold, the engine, by @ashxhart — https://github.com/ashhart/TensorFold
• The 4× RTX PRO 6000 recipe by Aevonix — https://github.com/Aevonix/GLM-5.3-Flash-EXL3-4x-RTX-PRO-6000-TensorFold
• The EXL3 4-bpw pack and the original patch series by @MiaAI_lab — https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold
• The engine patch that fixes the sampler and shares burst prefixes, by commdata2338 — https://github.com/commdata2338/glm-5.3-flash-4x-rtx-pro-6000-benchmark
• The DFlash2 drafter by incoai — https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2
• exllamav3 and the EXL3 format by @turboderp_ — https://github.com/turboderp-org/exllamav3

Thank you all.

Repo: https://github.com/kachowtowmater/tensorfold-glm53-rtx-pro-6000

---

## Short version (one tweet + the three images)

GLM-5.3-Flash on TensorFold, 4× RTX PRO 6000: our agent-path overlay is public. Image-turn TTFT 2.87 → 0.12 s, cold 4-agent bursts 12.4 → 3.3 s, sampler collapse 24 → 188 tok/s, 0 cache evictions, 8-concurrent turns 5.8 → 2.5 s vs SGLang. Raw decode unchanged (223/433 tok/s), and we measured why: weights are ~18% of a spec-decode step. Patches, knobs, bench scripts, every result incl. the losers. Built on @ashxhart's TensorFold, the Aevonix recipe, @MiaAI_lab's pack, commdata2338's patch, incoai's DFlash2, @turboderp_'s exllamav3.
https://github.com/kachowtowmater/tensorfold-glm53-rtx-pro-6000
