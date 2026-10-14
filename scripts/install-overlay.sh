#!/usr/bin/env bash
# install-overlay.sh — put this repo's patches and knobs onto a clean Aevonix recipe checkout.
# Usage: ./scripts/install-overlay.sh /path/to/GLM-5.3-Flash-EXL3-4x-RTX-PRO-6000-TensorFold
# What it does (idempotent):
#   1. copies patches/*.patch into <recipe>/patches/extra/ (0909 mixed-bit load, 0910 vision-prefix resume,
#      0921 tokenize without the GIL, 0923 fastokens prompt encoding);
#      (commdata2338's engine patch is NOT fetched or shipped — licence unknown; see PINS.md and README "Optional")
#   2. edits <recipe>/scripts/apply-patches.sh and prepare.sh: the strict patch count becomes 86 release + the
#      number of files in patches/extra; pure line offsets are allowed for extra/ patches (fuzz never); no .orig files;
#   3. installs scripts/local.sh (the tuned knobs) as <recipe>/scripts/local.sh if none exists; an existing one gets the
#      TF_W1_FASTOKENS=1 line appended if it lacks it (without it patch 0923 stays off);
#   4. sets the image tag to tensorfold-glm53:1.1.0-x14 and adds fastokens==0.3.2 to IMAGE_EXTRAS (patch 0923 imports it).
set -euo pipefail
recipe=${1:?path to the Aevonix recipe checkout}
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$recipe/scripts/apply-patches.sh" ] && [ -d "$recipe/patches/aevonix" ] || { echo "not an Aevonix recipe checkout: $recipe"; exit 2; }
mkdir -p "$recipe/patches/extra"
cp "$here"/patches/*.patch "$recipe/patches/extra/"
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
edit(ap, [
  ("printf '%s\\n' patches/{miaai-lab,aevonix}/*.patch", "printf '%s\\n' patches/{miaai-lab,aevonix,extra}/*.patch", True),
  ("[[ ${#patches[@]} == 86 ]] || fail 'expected exactly 86 release patches'",
   f"[[ ${{#patches[@]}} == {86+n} ]] || fail 'expected exactly 86 release + {n} extra patches'", True),
  ("patch -p0 --forward --batch --fuzz=0 \"${options[@]}\"", "patch -p0 --forward --batch --fuzz=0 --no-backup-if-mismatch \"${options[@]}\"", True),
  ("    if grep -Eiq '\\b(fuzz|offset)\\b' \"$log\"; then",
   "    # extra/ patches: a pure line offset is accepted (content matched, --fuzz=0 still enforced); fuzz never is\n"
   "    if [[ \"$p\" == patches/extra/* ]]; then pat='\\b(fuzz)\\b'; else pat='\\b(fuzz|offset)\\b'; fi\n"
   "    if grep -Eiq \"$pat\" \"$log\"; then", True),
])
# an earlier run may have set a different count: normalise
s = open(ap).read()
s = re.sub(r"== \d+ \]\] \|\| fail 'expected exactly 86 release \+ \d+ extra patches'", f"== {86+n} ]] || fail 'expected exactly 86 release + {n} extra patches'", s)
open(ap, "w").write(s)
pp = f"{recipe}/scripts/prepare.sh"
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
s = re.sub(r'IMAGE="\$\{IMAGE:-tensorfold-glm53:1\.1\.0(-[a-z0-9]+)?\}"', 'IMAGE="${IMAGE:-tensorfold-glm53:1.1.0-x14}"', s)
m = re.search(r"^IMAGE_EXTRAS='([^']*)'$", s, re.M)
if not m: raise SystemExit(f"{cs}: IMAGE_EXTRAS line not found")
if "fastokens==" not in m.group(1):
    s = s[:m.start()] + f"IMAGE_EXTRAS='{m.group(1)} fastokens==0.3.2'" + s[m.end():]
open(cs, "w").write(s)
print(f"edited apply-patches.sh / prepare.sh for 86 + {n} patches; image tag tensorfold-glm53:1.1.0-x14 (+ fastokens==0.3.2)")
PY
if [ -f "$recipe/scripts/local.sh" ]; then
  grep -q TF_W1_FASTOKENS "$recipe/scripts/local.sh" || grep -A0 -B1 '^export TF_W1_FASTOKENS' "$here/scripts/local.sh" >> "$recipe/scripts/local.sh"
else
  cp "$here/scripts/local.sh" "$recipe/scripts/local.sh"
fi
echo "overlay installed: $n_extra extra patch(es) in $recipe/patches/extra; knobs at scripts/local.sh"
echo "next: cd $recipe && ./scripts/prepare.sh   (builds the image: clones TensorFold v0.6.5, applies 86 + $n_extra patches)"
