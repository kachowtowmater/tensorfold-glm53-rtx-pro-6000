# Pins — what this overlay sits on (fetch these yourself; none of it is vendored here)

| component | pin | licence | why we depend on it |
|---|---|---|---|
| TensorFold (ashhart) | v0.6.5 @ `609ca419abecebdc5a059498a613680bd3aa847f` — https://github.com/ashhart/TensorFold | Apache-2.0 | the engine. Note: its Python engine is frozen upstream (issue #286; development moved to a Zig engine), so patches like ours are the only way to change it |
| Aevonix recipe | `a19078566298f8c51d9dbb317f20efd51fe564df` — https://github.com/Aevonix/GLM-5.3-Flash-EXL3-4x-RTX-PRO-6000-TensorFold | see repo | 86 patches (Mia-AiLab + Aevonix) that make TF run TP4 on 4× RTX PRO 6000, plus prepare/start/stop scripts. This overlay installs onto it |
| Mia-AiLab EXL3 pack | `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` @ `78353f1f6eb2c96fa6c62de57b44345d1f497c38` (164 GB) | Apache-2.0 | the weights: EXL3 4-bit routed experts, dense quantized at load |
| DFlash2 drafter | `incoai/GLM-5.3-Flash-DFlash2` @ `bf582e4eacc1810f76656d1811693ff6c6737d2a` | CC BY-NC-ND 4.0 (non-commercial) | speculative decoding; `DRAFTER=mtp` is the alternative if the licence does not fit you |
| base image | `nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c` | NVIDIA | what the recipe builds on |
| exllamav3 (turboderp) | `16a49792a3c93d8432d72e6c4bce800841566577` (v1.5.4) — https://github.com/turboderp-org/exllamav3 | MIT | only if you requantize (mixed-width packs); not needed to serve |
| commdata2338 engine patch (optional) | https://github.com/commdata2338/glm-5.3-flash-4x-rtx-pro-6000-benchmark `patches/tensorfold-modified-engine.patch` @ `0734ef4e4` (blob `77f97a429ef62290f449aa002599a0f316fb0286`) | not stated | fixes the sampler collapse on `top_k=-1 / top_p 1.0` (24 → 188 tok/s) and adds burst-prefix sharing + cache_reason logging. **Not shipped here**; our production tree carries a rebased copy. If you use it, apply it as `patches/extra/0900-…` and use `patches/variants/0910-…x1.patch` |

Model: `zai-org/GLM-5.3-Flash` (the FP8 base, 306 GB) is only needed for requantization.
