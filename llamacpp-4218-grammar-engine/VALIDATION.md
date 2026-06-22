# Validation evidence

Validation date: 2026-06-22.

Branch tip:

```text
3e4aa92bed972d60cbf9a02795d40bed10a60338
grammar: reject EOG while UTF-8 is incomplete
```

## Commands

Focused test build:

```bash
cmake -S . -B build-compactor \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DGGML_NATIVE=OFF

cmake --build build-compactor \
  --target test-grammar-compactor test-grammar-differential \
           test-grammar-integration test-llama-grammar test-gbnf-validator \
  -j$(nproc)
```

Focused harnesses:

```bash
./build-compactor/bin/test-grammar-compactor
/usr/bin/time -f "long_elapsed=%E maxrss_kb=%M" \
  ./build-compactor/bin/test-grammar-compactor --long
./build-compactor/bin/test-grammar-differential
ctest --test-dir build-compactor -R "(grammar|gbnf)" --output-on-failure
```

Sanitizer run:

```bash
cmake -S . -B build-asan \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DGGML_NATIVE=OFF \
  -DLLAMA_SANITIZE_ADDRESS=ON \
  -DLLAMA_SANITIZE_UNDEFINED=ON

cmake --build build-asan --target test-grammar-compactor -j$(nproc)
ASAN_OPTIONS=detect_leaks=0:abort_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
  ./build-asan/bin/test-grammar-compactor
```

Full server-enabled build:

```bash
cmake -S . -B build-full-server \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DGGML_NATIVE=OFF

cmake --build build-full-server -j$(nproc)
```

## Focused Compactor Output

Short run:

```text
astar n=1000 origins=2 sealed=1 current_items=5 stored_items=5 resume_entries=1
balanced-prefix n=64 origins=65 sealed=64 current_items=4 stored_items=67 resume_entries=63
token-not-empty-candidate excluded=0 allowed=1
nullable-utf8-eog partial=0 completed=1 wrong=0
completion-reallocation-stress callers=8191 advanced=8191 final_items=16384
```

Long run:

```text
astar n=100000 origins=2 sealed=1 current_items=5 stored_items=5 resume_entries=1
balanced-prefix n=512 origins=513 sealed=512 current_items=4 stored_items=515 resume_entries=511
token-not-empty-candidate excluded=0 allowed=1
nullable-utf8-eog partial=0 completed=1 wrong=0
completion-reallocation-stress callers=8191 advanced=8191 final_items=16384
long_elapsed=0:18.96 maxrss_kb=8252
```

Interpretation:

- Observed: `a*` plateaus at two origins and five stored items through `n=100000`.
- Observed: balanced recursive prefixes retain linear origin depth, which is expected because distinct unmatched input positions remain semantically relevant.
- Observed: token, nullable UTF-8 EOG, and completion reallocation regressions pass in the same focused harness.

## Differential Output

```text
grammar-differential decisions=2328 fnv64=b77904254b842fea
```

Trace equivalence against the precompactor baseline with the expanded harness:

```text
precompactor trace lines: 2329
current trace lines: 2329
diff output lines: 0
```

Interpretation:

- Observed: clone-vs-commit decisions are stable for the expanded differential corpus.
- Observed: the current branch and the precompactor baseline produce identical expanded traces for this harness.
- Unknown: this is not a formal language-equivalence proof.

## CTest Gate

```text
test-grammar-parser: passed
test-grammar-compactor: passed
test-grammar-differential: passed
test-grammar-integration: passed
test-llama-grammar: passed
test-json-schema-to-grammar: passed

100% tests passed, 0 tests failed out of 6
```

## Sanitizer Gate

Focused ASAN/UBSAN compactor output:

```text
astar n=1000 origins=2 sealed=1 current_items=5 stored_items=5 resume_entries=1
balanced-prefix n=64 origins=65 sealed=64 current_items=4 stored_items=67 resume_entries=63
token-not-empty-candidate excluded=0 allowed=1
nullable-utf8-eog partial=0 completed=1 wrong=0
completion-reallocation-stress callers=8191 advanced=8191 final_items=16384
asan_stderr_bytes=0
```

Interpretation:

- Observed: no ASAN or UBSAN report was emitted by the focused compactor regression harness.

## Full Build Gate

```text
completed to 100%, including llama-server and llama-app
```

Non-blocking warnings observed during build configuration or dependency tooling:

- OpenSSL was unavailable in one configuration, so HTTPS support was disabled.
- UI dependency tooling reported audit warnings.
- Compiler warnings outside the grammar-engine diff were present.

These warnings did not block the grammar tests or the full server-enabled build.

## API Surface Check

Tracked-source search:

```text
get_stacks|stacks_cur|stacks_org: no matches
```

Interpretation:

- Observed: the current branch does not retain the old prototype stack-access shim names in tracked source files.
