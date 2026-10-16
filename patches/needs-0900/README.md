# Patches that need the optional 0900 engine patch

Three of our patches do not apply on the shipped series: they edit code that
**0900-commdata-engine** (commdata2338's engine patch, optional — see PINS.md) adds, so
they only apply on trees that carry it. They live here, outside `patches/`, because the
overlay copies everything in `patches/*.patch` into the recipe's `patches/extra/` and the
series would fail without 0900's code present.

`scripts/install-overlay.sh` applies them for you when it finds
`recipe/patches/extra/0900-commdata-engine.patch`: it copies this directory's patches in
after 0900 (filename order: 0900 → 0941 → 0970 → 0990) and recounts. Without 0900 the
installer skips this directory and says so.

## Why each one needs 0900

- `0941-glm-pcache-park-restore.patch` — wires the prompt-cache tiers (0940) into the
  engine; its `multi.py` hunks edit the burst machinery 0900 adds
  (`cache_reasons`/`_kept_by_id`, the `burst_wait` consumer loop) and 0900's
  cache-miss accounting.
- `0970-glm-cross-class-time-credit.patch` — reworks 0900's `TimeCredit` in
  `src/tensorfold/cuda/priority.py`: with `TF_GLM_FILL_CREDIT=time` (0900's flag) the
  credit is keyed by (fill class, round class) and `contend()` takes the iteration's
  share (a one-arg `contend` — what the 0900 tests call — keeps the share it had).
- `0990-glm-park-async.patch` — write-behind parking; it edits 0941's output (park/
  restore plumbing in `pcache.py` and `multi.py`), so apply order is 0900 → 0941 → 0990.

`patches/0972-glm-group-no-later-class.patch` mentions 0970 but does **not** need 0900:
it borrows the credit's position in `MultiDecoder._group` only for context and stays in
the shipped series. The shipped 0940 (the tiers' CPU part) has no 0900 dependence either.
