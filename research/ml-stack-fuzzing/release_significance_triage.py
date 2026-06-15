#!/usr/bin/env python3
"""Release-significance crash triage for fuzzing the ML inference stack.

The point of this script is one filter that most fuzzing write-ups skip:

    A sanitizer (ASAN/UBSan) abort is NOT, by itself, production impact.

A libFuzzer + ASAN build aborts on the *first* out-of-bounds byte. But a
release build compiled without sanitizers will frequently absorb that same
read (it lands in adjacent valid heap, or is a benign over-read of a few
bytes) and keep running. So an ASAN crash proves "undefined behaviour exists
here"; it does NOT prove "an attacker who sends this input crashes your
server." Only a crash on a *no-sanitizer release build* is a real
denial-of-service candidate.

This tool takes a pile of fuzzer-found crashing inputs and produces a triaged
list where each unique signature is labelled `RELEASE-SEGV` (production DoS
candidate) or `release-survives` (UB only — sanitizer artefact). That label is
what should gate whether a finding is worth a maintainer's time.

Pipeline:
  1. dedup crashing inputs by content hash
  2. classify each by ASAN signature (error class + top in-project stack frame)
  3. group by signature
  4. drop signatures already known/characterised (KNOWN_FRAMES)
  5. for each NEW signature: re-run the representative input on a *release*
     (no-sanitizer) build and record whether it SIGSEGVs
  6. emit JSON + a human summary; the production-significant set is the
     NEW signatures that SIGSEGV a release build

This is deliberately target-agnostic. Wire the two callbacks at the top
(`asan_cmd` and `release_run`) to your harness and your release build and the
rest is reusable across any native parser/loader/compiler that takes
untrusted input (model-file loaders, tokenizers, grammar/structured-output
compilers, image/audio preprocessors, etc.).
"""
from __future__ import annotations
import os
import sys
import glob
import json
import hashlib
import argparse
import subprocess
import collections

# ----------------------------------------------------------------------------
# Configure these two for your target. Both are intentionally simple shell-outs
# so the same triage logic works for any libFuzzer/AFL++ harness.
# ----------------------------------------------------------------------------

# ASAN error classes worth distinguishing, in priority order.
ASAN_CLASSES = [
    "heap-buffer-overflow", "heap-use-after-free", "stack-buffer-overflow",
    "global-buffer-overflow", "stack-overflow", "SEGV", "FPE",
    "negative-size-param", "allocation-size-too-big", "out-of-memory",
    "runtime error",  # UBSan
]

# Substrings of stack frames you've already characterised; signatures whose top
# frame matches one of these are reported but skipped for release-triage.
KNOWN_FRAMES: list[str] = []  # e.g. ["SomeKnownLowSevGetter"]


def asan_signature(asan_binary: str, crash_path: str, project_marker: str):
    """Run a crashing input under the ASAN build; return (error_class, top_frame).

    `project_marker` is a substring identifying *your* stack frames (e.g. a
    namespace like ' in mylib::') so the signature is the first meaningful
    in-project frame rather than a libc/allocator frame.
    """
    env = {**os.environ,
           "ASAN_OPTIONS": "detect_leaks=0:abort_on_error=1",
           "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1"}
    try:
        p = subprocess.run([asan_binary, crash_path], capture_output=True,
                           text=True, timeout=30, env=env)
        err = p.stderr + p.stdout
    except subprocess.TimeoutExpired:
        return ("timeout", "(timeout)")
    error_class = "?"
    for k in ASAN_CLASSES:
        if k in err:
            error_class = k
            break
    frames = [l for l in err.splitlines() if project_marker in l]
    top = "(no-project-frame)"
    for l in frames:
        # frame text after " in " up to the arg list / const qualifier
        f = l.split(" in ", 1)[-1].split(" const")[0].split("(")[0].strip()
        top = f[:90]
        if not any(k in f for k in KNOWN_FRAMES):
            break
    return (error_class, top)


def release_segv(release_cmd: list[str], crash_path: str) -> bool:
    """Run a crashing input through a NO-SANITIZER release build.

    Returns True iff the process dies with SIGSEGV (rc -11 / 139) or SIGABRT
    (rc -6 / 134) — i.e. a real crash an unsanitized deployment would also hit.
    A clean exit (even rc != 0 from a caught exception) means the release build
    absorbed the UB → sanitizer artefact, not production impact.
    """
    try:
        p = subprocess.run(release_cmd + [crash_path], capture_output=True,
                           timeout=30)
        return p.returncode in (-11, 139, -6, 134)
    except subprocess.TimeoutExpired:
        return True  # a hang on the release build is itself a DoS


def triage(crash_glob, asan_binary, release_cmd, project_marker):
    # 1. dedup by content
    seen, items = set(), []
    for f in glob.glob(crash_glob):
        if not os.path.isfile(f):
            continue
        try:
            data = open(f, "rb").read()
        except OSError:
            continue
        if not data:
            continue
        h = hashlib.sha256(data).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        items.append(f)
    # 2 + 3. classify + group by ASAN signature
    groups = collections.defaultdict(list)
    for f in items:
        groups[asan_signature(asan_binary, f, project_marker)].append(f)
    # 4 + 5. report; release-triage NEW signatures
    report = []
    for (error_class, top), members in sorted(groups.items(),
                                              key=lambda kv: -len(kv[1])):
        known = any(k in top for k in KNOWN_FRAMES)
        rel = None
        if not known and release_cmd:
            rel = "RELEASE-SEGV" if release_segv(release_cmd, members[0]) \
                else "release-survives"
        report.append({
            "error_class": error_class, "frame": top, "count": len(members),
            "known": known, "release": rel,
            "example": os.path.basename(members[0]),
        })
    return len(items), report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crashes", required=True,
                    help="glob of crashing inputs, e.g. './allcrashes/*'")
    ap.add_argument("--asan-binary", required=True,
                    help="the libFuzzer/ASAN binary that takes a crash file as argv[1]")
    ap.add_argument("--release-cmd", default="",
                    help="space-separated no-sanitizer release runner that takes "
                         "a crash file as its last arg, e.g. "
                         "'/path/to/release-venv/python run_one.py'")
    ap.add_argument("--project-marker", required=True,
                    help="substring identifying your stack frames, e.g. ' in mylib::'")
    args = ap.parse_args()

    release_cmd = args.release_cmd.split() if args.release_cmd else []
    n, report = triage(args.crashes, args.asan_binary, release_cmd,
                       args.project_marker)

    print(f"# unique crash inputs (by content): {n}")
    print(json.dumps(report, indent=2))
    print("\n=== SUMMARY ===")
    for r in report:
        tag = "KNOWN" if r["known"] else "*** NEW ***"
        rel = f" | release={r['release']}" if r["release"] else ""
        print(f"[{tag}] {r['error_class']} @ {r['frame']}  x{r['count']}{rel}")
    sig = [r for r in report if not r["known"] and r["release"] == "RELEASE-SEGV"]
    print(f"\nProduction-significant (NEW + release SIGSEGV): {len(sig)}")
    for r in sig:
        print(f"  -> {r['error_class']} @ {r['frame']} (example {r['example']})")
    # exit non-zero if there is a production-significant finding (CI-friendly)
    sys.exit(1 if sig else 0)


if __name__ == "__main__":
    main()
