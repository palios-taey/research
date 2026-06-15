#!/usr/bin/env bash
# Sanitizer build template for a libFuzzer harness against a native ML library.
#
# Target-agnostic scaffolding. Fill in TARGET_SRC / TARGET_INCLUDE / the link
# line for the library under test. The important, non-obvious flags are noted.
set -euo pipefail

# --- configure ---------------------------------------------------------------
HARNESS_SRC="${HARNESS_SRC:-fuzz_target_skeleton.cc}"
TARGET_INCLUDE="${TARGET_INCLUDE:-/path/to/library/include}"
TARGET_SRC="${TARGET_SRC:-}"     # extra .cc to compile in, or link a prebuilt lib
TARGET_LIB="${TARGET_LIB:-}"     # e.g. -L/path/to/build -lmylib
OUT="${OUT:-fuzz_target}"
CXX="${CXX:-clang++}"

# --- flags -------------------------------------------------------------------
# -fsanitize=fuzzer,address,undefined : libFuzzer + ASAN + UBSan.
# -fno-omit-frame-pointer -g          : usable stack frames for triage.
# -O1                                 : keep enough optimisation that inlining
#                                       roughly matches a release build's shape,
#                                       but not so much that UBSan loses precision.
# IMPORTANT: build the library under test WITHOUT LTO and with any internal
# "debug check / assert" toggles turned OFF — you want to find the bugs the
# library ships with in release, not assertion failures that only fire in debug.
SAN_FLAGS=(-std=c++17 -O1 -g -fno-omit-frame-pointer
           -fsanitize=fuzzer,address,undefined)

set -x
"$CXX" "${SAN_FLAGS[@]}" \
  -I"$TARGET_INCLUDE" \
  "$HARNESS_SRC" ${TARGET_SRC:+$TARGET_SRC} \
  ${TARGET_LIB:+$TARGET_LIB} \
  -o "$OUT"
set +x

echo "built $OUT"
echo "run:   ./$OUT -max_len=4096 corpus_dir/   # fuzz"
echo "repro: ./$OUT path/to/crash               # single input"
echo
echo "Pair with a SEPARATE no-sanitizer release build for significance triage"
echo "(see ../release_significance_triage.py) — an ASAN abort here is not yet a"
echo "production DoS until the release build also crashes."
