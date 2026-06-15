# Fuzzing the ML Inference Stack — Harness Design + Release-Significance Triage

A small, reusable methodology for finding **real** memory-safety and
denial-of-service bugs in the native code that takes untrusted input across the
modern LLM-serving stack: model-file loaders, tokenizers, and the
grammar / structured-output compilers that sit in front of constrained
decoding (the parsers that turn a user-supplied EBNF / JSON-schema / regex into
a compiled automaton inside engines like vLLM and SGLang).

The contribution here is **not** a single bug. It is one discipline that
separates a noisy fuzzing run from a finding a maintainer will actually act on:

> **A sanitizer crash is not production impact.** An ASAN/UBSan build aborts on
> the first out-of-bounds byte; a release build compiled without sanitizers
> usually absorbs that same read and keeps running. So a sanitizer abort proves
> *undefined behaviour exists* — it does **not** prove *an attacker who sends
> this input crashes your server*. Only a crash on a **no-sanitizer release
> build** is a real DoS candidate.

Most public fuzzing write-ups report the ASAN crash count. That number is an
upper bound on real findings, often a loose one. The tooling here applies a
**release-significance filter** to every unique crash signature, so what gets
escalated to a maintainer is the subset that actually crashes production-shaped
binaries — not the much larger pile of benign sanitizer artefacts.

## Why this surface

Constrained decoding and model loading are now on the **untrusted-input path**
of hosted inference: a request can carry a grammar, a JSON schema, or a regex
that the server compiles in-process; a model artifact can carry attacker-shaped
metadata. The hot loops are native C/C++/Rust for speed, and the input grammars
are adversarially expressive. That combination — untrusted input × native
parser × performance-tuned hot path — is exactly where memory-safety and
algorithmic-complexity bugs live, and it is comparatively under-fuzzed relative
to, say, image codecs.

## The methodology (four habits)

1. **Multiplex every untrusted entry point behind one harness.** Select the API
   under test from the first fuzzer byte (compile-from-EBNF, from-JSON-schema,
   from-regex, post-compile matcher state, …). One campaign then exercises the
   whole surface instead of one entry point. Distinct entry points fail in
   distinct ways; a single-entry harness leaves most of the attack surface dark.
   (`harness/fuzz_target_skeleton.cc`.)

2. **Catch the library's *expected* validation failures.** The harness must only
   terminate on a sanitizer fault, never on a normal "invalid input" exception —
   otherwise every malformed byte string reads as a crash and the corpus never
   deepens past the parser's front door. (Same skeleton.)

3. **Build for findability, then re-triage for significance.** The ASAN/UBSan
   build (`harness/build.sh`) is for *discovery*: build the library under test
   without LTO and with internal debug-asserts **off**, so you find the bugs the
   library ships in release, not debug-only assertions. Then run every unique
   crash through a separate release build to decide what is real
   (`release_significance_triage.py`).

4. **Sustain, distribute, and re-triage — don't spot-check.** Real findings in
   this class come from sustained campaigns (hours, all cores, fork mode,
   multiple nodes), with a periodic re-triage loop over accumulated crashes, not
   from a five-minute single-harness run. The triage tool is written to be the
   reduce step over a fleet's worth of accumulated crash inputs.

## The triage tool

`release_significance_triage.py` is target-agnostic. Point it at a pile of
fuzzer-found crashing inputs, the ASAN binary, an optional no-sanitizer release
runner, and a string identifying your stack frames. It:

1. dedups crashing inputs by content hash,
2. classifies each by ASAN signature (error class + first in-project frame),
3. groups by signature and drops already-characterised ones,
4. **re-runs each new signature on the release build and labels it
   `RELEASE-SEGV` (production DoS candidate) or `release-survives` (UB only),**
5. emits JSON + a human summary and exits non-zero when a production-significant
   finding exists (CI-friendly).

```bash
python3 release_significance_triage.py \
  --crashes './allcrashes/*' \
  --asan-binary ./fuzz_target \
  --release-cmd '/path/to/release-venv/python run_one.py' \
  --project-marker ' in yourlib::'
```

The `RELEASE-SEGV` set is the only set worth a maintainer's attention.

## Contents

| File | Purpose |
|---|---|
| `release_significance_triage.py` | The reusable triage tool — dedup → ASAN-classify → group → **release-significance filter** → escalate. Runnable, target-agnostic. |
| `harness/fuzz_target_skeleton.cc` | Multiplexed libFuzzer target skeleton (the entry-point-multiplexing + expected-failure-catching pattern). |
| `harness/build.sh` | Sanitizer build template with the non-obvious flags annotated (no-LTO, asserts-off, `-O1` for release-shaped inlining). |

These are scaffolding + the reusable triage logic, not a finished harness for
any one library — wiring a specific target is a few lines at the marked spots.

## Worked example (in coordinated disclosure)

This methodology was applied to a widely-used native structured-output /
grammar-compilation library on the inference path. The release-significance
filter separated several sanitizer-only artefacts from a smaller set of crashes
that reproduce on a production-shaped (no-sanitizer) build — denial-of-service
from untrusted compile-time input.

That finding has been reported through the project's **GitHub coordinated
disclosure** channel and is, as of this writing (June 2026), **in triage and
not yet public**. Per responsible-disclosure practice, the affected library,
the vulnerable entry points, and the reproducing inputs are intentionally
withheld here until the advisory is published.

> **Pending verifiable artifact.** When the advisory publishes, this section
> will be updated with the public advisory link (and CVE, if assigned) so the
> claim is externally verifiable rather than asserted. Until then, treat the
> worked example as *Inferred from our own run*, not independently confirmed.

## Epistemic status (three-register)

- **Observed:** the harness design and the triage tool are real and run as
  shown; the release-significance filter materially reduced the escalation set
  on our own campaign (sanitizer-crash count ≫ release-SIGSEGV count).
- **Inferred:** that the surviving release-SIGSEGV finding constitutes an
  exploitable DoS in the affected library — reported, but not yet
  maintainer-confirmed (see above).
- **Unknown / out of scope:** exploitability beyond denial-of-service; whether
  the same signatures reproduce on every build configuration / platform; any
  claim about other libraries not covered by our own runs.

## License

Apache-2.0 (repository default). The scaffolding and triage tool are meant to be
copied into your own fuzzing setup.

## Provenance

Ported from internal fuzzing working dirs once the methodology was defensible as
a standalone contribution and the embargoed specifics could be cleanly excluded.
Initial port: 2026-06-15.
