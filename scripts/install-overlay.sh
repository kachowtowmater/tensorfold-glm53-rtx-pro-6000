#!/usr/bin/env bash
# install-overlay.sh — put this repo's patches and knobs onto a clean Aevonix recipe checkout.
# Usage: ./scripts/install-overlay.sh /path/to/GLM-5.3-Flash-EXL3-4x-RTX-PRO-6000-TensorFold
# What it does (idempotent):
#   1. copies patches/*.patch into <recipe>/patches/extra/ (0909 mixed-bit load, 0910 vision-prefix resume,
#      0921 tokenize without the GIL, 0923 fastokens prompt encoding, 0930 select-onepass, 0932 kept-entries
#      share, 0933 name-priority, 0940 prompt-cache tiers' CPU part, 0972 group no-later-class, 0980 topk_v2);
#      when the recipe already carries 0900-commdata-engine.patch in patches/extra/ (you dropped it in — it is
#      NOT fetched or shipped here, licence unknown; see PINS.md and README "Optional"), the three
#      patches/needs-0900/ patches (0941 park/restore wiring, 0970 cross-class time credit, 0990 write-behind
#      park) are copied in as well, right after 0900 in filename order, and counted;
#   2. edits <recipe>/scripts/apply-patches.sh and prepare.sh: the strict patch count becomes 86 release + the
#      number of files in patches/extra; pure line offsets are allowed for extra/ patches (fuzz never); no .orig files;
#   3. installs scripts/local.sh (the tuned knobs) as <recipe>/scripts/local.sh if none exists; an existing one gets the
#      TF_W1_FASTOKENS=1 line appended if it lacks it (without it patch 0923 stays off);
#   4. sets the image tag to tensorfold-glm53:1.1.0-x25 and adds fastokens==0.3.2 and xxhash==3.5.0 to IMAGE_EXTRAS
#      (0923 imports fastokens; the prompt-cache tiers checksum with xxh3_64).
set -euo pipefail
recipe=${1:?path to the Aevonix recipe checkout}
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$recipe/scripts/apply-patches.sh" ] && [ -d "$recipe/patches/aevonix" ] || { echo "not an Aevonix recipe checkout: $recipe"; exit 2; }
mkdir -p "$recipe/patches/extra"
if [ -e "$recipe/patches/extra/0910-vision-prefix-resume.x1.patch" ]; then
  # the x1 variant replaces the vanilla 0910: copy everything except it
  for f in "$here"/patches/*.patch; do [ "$(basename "$f")" = 0910-vision-prefix-resume.patch ] || cp "$f" "$recipe/patches/extra/"; done
else
  cp "$here"/patches/*.patch "$recipe/patches/extra/"
fi
needs0900=0
if [ -f "$recipe/patches/extra/0900-commdata-engine.patch" ]; then
  cp "$here"/patches/needs-0900/*.patch "$recipe/patches/extra/"
  needs0900=$(ls "$here"/patches/needs-0900/*.patch | wc -l | tr -d ' ')
  echo "0900 found: also applying the $needs0900 patches from patches/needs-0900/ (they edit 0900's code)"
else
  echo "0900-commdata-engine.patch not in $recipe/patches/extra: skipping patches/needs-0900/ (0941, 0970, 0990 need 0900's code; see patches/needs-0900/README.md)"
fi
n_extra=$(ls "$recipe"/patches/extra/*.patch | wc -l | tr -d ' ')
python3 - "$recipe" "$n_extra" <<'PY'
import sys, re
recipe, n = sys.argv[1], int(sys.argv[2])
def edit(path, pairs):
    s = open(path).read(); orig = s
    for old, new, required in pairs:
        if old in s: s = s.replace(old, new)
        elif required and new not in s: raise SystemExit(f"{path}: expected text not found:\n{old}")
    if s != orig: open(path, "w").write(s)
ap = f"{recipe}/scripts/apply-patches.sh"
# an earlier run may have set a different count: normalise BEFORE the required-text checks
s = open(ap).read()
s = re.sub(r"== \d+ \]\] \|\| fail 'expected exactly 86 release( \+ \d+ extra)? patches'", f"== {86+n} ]] || fail 'expected exactly 86 release + {n} extra patches'", s)
open(ap, "w").write(s)
edit(ap, [
  ("printf '%s\\n' patches/{miaai-lab,aevonix}/*.patch", "printf '%s\\n' patches/{miaai-lab,aevonix,extra}/*.patch", True),
  ("[[ ${#patches[@]} == 86 ]] || fail 'expected exactly 86 release patches'",
   f"[[ ${{#patches[@]}} == {86+n} ]] || fail 'expected exactly 86 release + {n} extra patches'", True),
  ("patch -p0 --forward --batch --fuzz=0 \"${options[@]}\"", "patch -p0 --forward --batch --fuzz=0 --no-backup-if-mismatch \"${options[@]}\"", True),
  ("    if grep -Eiq '\\b(fuzz|offset)\\b' \"$log\"; then",
   "    # extra/ patches: a pure line offset is accepted (content matched, --fuzz=0 still enforced); fuzz never is\n"
   "    if [[ \"$p\" == patches/extra/* ]]; then pat='\\b(fuzz)\\b'; else pat='\\b(fuzz|offset)\\b'; fi\n"
   "    if grep -Eiq \"$pat\" \"$log\"; then", True),
  ("cd -- \"$patch_dir\"\nfor p in \"${patches[@]}\"; do\n  # A dry run catches offsets before this patch can modify any files.\n  for phase in check apply; do\n    options=()\n    [[ \"$phase\" != check ]] || options+=(--dry-run)\n    if ! patch -p0 --forward --batch --fuzz=0 --no-backup-if-mismatch \"${options[@]}\" --input=\"$repo_dir/$p\" >\"$log\" 2>&1; then",
   "cd -- \"$patch_dir\"\nfor p in \"${patches[@]}\"; do\n  # A patch may carry a \"# root-relative\" marker: it is applied from the target root instead of\n  # patch_dir (used for series additions that also touch tests/, which live beside src/).\n  if head -1 \"$repo_dir/$p\" | grep -q '^# root-relative'; then work_dir=$target; else work_dir=$patch_dir; fi\n  # A dry run catches offsets before this patch can modify any files.\n  for phase in check apply; do\n    options=()\n    [[ \"$phase\" != check ]] || options+=(--dry-run)\n    if ! (cd -- \"$work_dir\" && patch -p0 --forward --batch --fuzz=0 --no-backup-if-mismatch \"${options[@]}\" --input=\"$repo_dir/$p\") >\"$log\" 2>&1; then", True),
])
# an earlier run may have set a different count: normalise
s = open(ap).read()
s = re.sub(r"== \d+ \]\] \|\| fail 'expected exactly 86 release \+ \d+ extra patches'", f"== {86+n} ]] || fail 'expected exactly 86 release + {n} extra patches'", s)
open(ap, "w").write(s)
pp = f"{recipe}/scripts/prepare.sh"
s = open(pp).read()
s = re.sub(r"== \d+ \)\) \|\| fail E_PATCHES 'This build requires exactly 86 release( \+ \d+ extra)? patches\.'", f"== {86+n} )) || fail E_PATCHES 'This build requires exactly 86 release + {n} extra patches.'", s)
open(pp, "w").write(s)
edit(pp, [
  ("(( ${#PATCHES[@]} == 86 )) || fail E_PATCHES 'The release requires exactly 86 patches.'",
   f"(( ${{#PATCHES[@]}} == {86+n} )) || fail E_PATCHES 'This build requires exactly 86 release + {n} extra patches.'", True),
  ("patches/{miaai-lab,aevonix}/*.patch", "patches/{miaai-lab,aevonix,extra}/*.patch", False),
])
s = open(pp).read()
s = re.sub(r"== \d+ \)\) \|\| fail E_PATCHES 'This build requires exactly 86 release \+ \d+ extra patches.'", f"== {86+n} )) || fail E_PATCHES 'This build requires exactly 86 release + {n} extra patches.'", s)
open(pp, "w").write(s)
cs = f"{recipe}/scripts/config.sh"
s = open(cs).read()
s = re.sub(r'IMAGE="\$\{IMAGE:-tensorfold-glm53:1\.1\.0(-[a-z0-9]+)?\}"', 'IMAGE="${IMAGE:-tensorfold-glm53:1.1.0-x25}"', s)
m = re.search(r"^IMAGE_EXTRAS='([^']*)'$", s, re.M)
if not m: raise SystemExit(f"{cs}: IMAGE_EXTRAS line not found")
extras = m.group(1).split()
if "fastokens==" not in m.group(1): extras.append("fastokens==0.3.2")
if "xxhash==" not in m.group(1): extras.append("xxhash==3.5.0")
s = s[:m.start()] + f"IMAGE_EXTRAS='{' '.join(extras)}'" + s[m.end():]
open(cs, "w").write(s)
print(f"edited apply-patches.sh / prepare.sh for 86 + {n} patches; image tag tensorfold-glm53:1.1.0-x25 (+ fastokens==0.3.2 xxhash==3.5.0)")
PY
if [ -f "$recipe/scripts/local.sh" ]; then
  grep -q TF_W1_FASTOKENS "$recipe/scripts/local.sh" || grep -A0 -B1 '^export TF_W1_FASTOKENS' "$here/scripts/local.sh" >> "$recipe/scripts/local.sh"
  # an older install (x14 or earlier) lacks the x15/x17/x18/x25 knobs: append any line present here but missing there
  grep -vE '^\s*#|^\s*$' "$here/scripts/local.sh" | while IFS= read -r line; do
    key=$(echo "$line" | sed -E 's/^export ([A-Za-z_0-9]+)=.*/\1/')
    [ -n "$key" ] && ! grep -q "export $key=" "$recipe/scripts/local.sh" && echo "$line" >> "$recipe/scripts/local.sh"
  done
else
  cp "$here/scripts/local.sh" "$recipe/scripts/local.sh"
fi
if [ "$needs0900" -gt 0 ]; then
  echo "overlay installed: $n_extra extra patch(es) in $recipe/patches/extra (base set $((n_extra - needs0900)) + the $needs0900 needs-0900 patches, applied after 0900); knobs at scripts/local.sh"
else
  echo "overlay installed: $n_extra extra patch(es) in $recipe/patches/extra (base set; without 0900 the needs-0900 set is not applied); knobs at scripts/local.sh"
fi
echo "next: cd $recipe && ./scripts/prepare.sh   (builds the image: clones TensorFold v0.6.5, applies 86 + $n_extra patches)"
