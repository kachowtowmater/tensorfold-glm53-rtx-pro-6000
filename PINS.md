# Pins — what this overlay sits on (fetch these yourself; none of it is vendored here)

| component | pin | licence | why we depend on it |
|---|---|---|---|
| TensorFold (ashhart) | v0.6.5 @ `609ca419abecebdc5a059498a613680bd3aa847f` — https://github.com/ashhart/TensorFold | Apache-2.0 | the engine. Note: its Python engine is frozen upstream (issue #286; development moved to a Zig engine), so patches like ours are the only way to change it |
| Aevonix recipe | `a19078566298f8c51d9dbb317f20efd51fe564df` — https://github.com/Aevonix/GLM-5.3-Flash-EXL3-4x-RTX-PRO-6000-TensorFold | see repo | 86 patches (Mia-AiLab + Aevonix) that make TF run TP4 on 4× RTX PRO 6000, plus prepare/start/stop scripts. This overlay installs onto it. Upstream `main` was since replaced by release 1.2.0 (TF v0.6.6, one commit); the pinned commit is no longer on a branch but still fetches by sha (`git fetch origin <sha>`, see README). We measured 1.2.0 and stayed on this pin (`rejected/README.md`) |
| Mia-AiLab EXL3 pack | `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` @ `78353f1f6eb2c96fa6c62de57b44345d1f497c38` (164 GB) | Apache-2.0 | the weights: EXL3 4-bit routed experts, dense quantized at load |
| DFlash2 drafter | `incoai/GLM-5.3-Flash-DFlash2` @ `bf582e4eacc1810f76656d1811693ff6c6737d2a` | CC BY-NC-ND 4.0 (non-commercial) | speculative decoding; `DRAFTER=mtp` is the alternative if the licence does not fit you |
| base image | `nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c` | NVIDIA | what the recipe builds on |
| fastokens (crusoecloud) | `fastokens==0.3.2` from PyPI (source https://github.com/crusoecloud/fastokens @ `c1e193ea7754306c394863e09d0def3aea211d4b`) | Apache-2.0 | patch 0923: a Rust re-implementation of the tokenizer.json BPE pipeline; prompts encode with the GIL released, 400k tokens in 41–44 ms. The installer adds it to the recipe's `IMAGE_EXTRAS` |
| MiaAI-Lab patch 0099 | https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold `patches/0099` (PR #108, merged in PR #131, 2026-10-10) | see repo | patch 0921 is our port of it to the 4× PRO 6000 series (adds the incremental-tokenizer sites) |
| exllamav3 (turboderp) | `16a49792a3c93d8432d72e6c4bce800841566577` (v1.5.4) — https://github.com/turboderp-org/exllamav3 | MIT | only if you requantize (mixed-width packs); not needed to serve |
| commdata2338 engine patch (optional) | https://github.com/commdata2338/glm-5.3-flash-4x-rtx-pro-6000-benchmark `patches/tensorfold-modified-engine.patch` @ `0734ef4e4` (blob `77f97a429ef62290f449aa002599a0f316fb0286`) | not stated | fixes the sampler collapse on `top_k=-1 / top_p 1.0` (24 → 188 tok/s) and adds burst-prefix sharing + cache_reason logging. **Not shipped here**; our production tree carries a rebased copy. If you use it, apply it as `patches/extra/0900-…` and use `patches/variants/0910-…x1.patch` |

Model: `zai-org/GLM-5.3-Flash` (the FP8 base, 306 GB) is only needed for requantization.

## What the overlay builds

Image tag `tensorfold-glm53:1.1.0-x14`. Patch series: 86 release + 4 from `patches/` (0909, 0910, 0921, 0923) = **90**; with the optional commdata2338 engine patch (0900) and the `0910 x1` variant, 91. Our production x14 tree is 86 + 0900 + 0910 x1 + 0921 + 0923 = 90 (it does not carry 0909, which only matters for mixed-width packs). Diff of our production scripts against the Aevonix pin: `scripts/diff-vs-aevonix-a190785.txt`.
