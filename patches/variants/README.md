# 0910 variants — vision-prefix resume

Two variants of patch 0910 exist. Only ONE ships in `patches/`: the vanilla one
(`patches/0910-vision-prefix-resume.patch`). The other is kept here for the x1 tree.

## `0910-vision-prefix-resume.patch` (vanilla — the shipped one)

- Applies on: **clean Aevonix a190785 + TensorFold v0.6.5 (`609ca419`) + 0909** via
  `scripts/apply-patches.sh` (the overlay's 88-patch series).
- What it does: `TF_GLM_VISION_PREFIX=1` lets an image prompt resume its kept **text**
  prefix, capped at its first image row. Both ranks decide alike:
  - `engine.py`: `vision_prefix()` env reader; `self.vision_prefix`; `_resume(..., cap)` —
    a snapshot longer than `cap` tokens never admits; the shared-prefix stop block keeps
    `stops = [positions[0]]` when the first image row sits on the grid;
  - `multi.py`: `self.vision_prefix`; `admit()` computes `cap = positions[0]` and calls
    `_resume(prompt, cap)`; `pack_prompt` packs the capped hit; a feed lane's fill stops
    once at `stop0` (the grid point at or before its first image row); `_resume(..., cap)`
    mirrors the engine cap;
  - `cli.py`: the serve banner prints `vision_prefix: on|off`.
- 12 hunks: cli 1 · engine 5 · multi 6.

## `0910-vision-prefix-resume.x1.patch` (variant — x1 tree only)

- The original 0910, written against the **x1 tree** = Aevonix a190785 + TF v0.6.5
  (`609ca419`) + **0900-commdata-engine** (commdata2338's engine patch, base per
  SOURCES.md: local sha256 `fa055e12c602f1390b5c37f7cd327d015c35eef07dbc41f5abf7f9c6006a19c3`)
  + 0901-mia-sliced-fill.
- It has 16 hunks: the same cli/engine content **plus three multi.py hunks that guard the
  burst machinery 0900 added** (`_resume_lane` feed-resume guard, `_resume_lane` keep-state
  guard behind `feed`, and the burst-resume `_resume(..., cap=...)` in the burst_wait loop).
- Those three hunks **do not apply on vanilla**: vanilla multi.py has no `_resume_lane`,
  no `TF_GLM_BURST_PREFIX`/`burst_wait`, no `cache_reasons`/`_cache_miss`. Dropping them is
  logically faithful — there is no burst path to guard.
- Its `admit()` hunk also rewrites the `cache_reasons` block 0900 introduced (vision /
  vision_prefix / no_room reasons); vanilla has no `cache_reasons`, so that rewrite drops
  out entirely.
- Do NOT put this file in `patches/`: the overlay copies everything in `patches/*.patch`
  into `patches/extra/` and the series would apply it against the wrong tree and fail.
