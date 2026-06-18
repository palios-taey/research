# Harnesses

These harnesses are intended to be run from a patched llama.cpp checkout.

## Growth validation

```bash
python3 path/to/llamacpp-4218-grammar-engine/harnesses/run_llamacpp_4218_validation.py build --llama-dir . --out-dir 4218-baseline
LLAMA_GRAMMAR_GSS=1 python3 path/to/llamacpp-4218-grammar-engine/harnesses/run_llamacpp_4218_validation.py build --llama-dir . --out-dir 4218-gss
```

The script compiles `helper_sources/llamacpp_4218_probe.cpp` if a probe binary is not already present.

## Reject-path benchmark

```bash
c++ -std=c++17 -O2 path/to/llamacpp-4218-grammar-engine/harnesses/reject_apply_bench.cpp \
  -I. -Iinclude -Isrc -Iggml/include -Lbuild/bin -lllama -Wl,-rpath,$PWD/build/bin \
  -o build/reject_apply_bench

./build/reject_apply_bench path/to/vocab.gguf path/to/grammar.gbnf 128000 30
LLAMA_GRAMMAR_GSS=1 ./build/reject_apply_bench path/to/vocab.gguf path/to/grammar.gbnf 128000 30
```

## Differential equivalence

```bash
c++ -std=c++17 -O2 path/to/llamacpp-4218-grammar-engine/harnesses/equivalence_harness.cpp \
  -I. -Iinclude -Isrc -Iggml/include -Lbuild/bin -lllama -Wl,-rpath,$PWD/build/bin \
  -o build/equivalence_harness

./build/equivalence_harness path/to/vocab.gguf equivalence.md \
  path/to/llamacpp-4218-grammar-engine/harnesses/fixtures/workflow_trailing_strategy/grammar.gbnf
```

## Independent falsification

```bash
c++ -std=c++17 -O2 path/to/llamacpp-4218-grammar-engine/harnesses/independent_falsify.cpp \
  -I. -Iinclude -Isrc -Iggml/include -Lbuild/bin -lllama -Wl,-rpath,$PWD/build/bin \
  -o build/independent_falsify

./build/independent_falsify path/to/vocab.gguf independent-falsification.md
```

Use a vocab-only GGUF or any model file loadable by llama.cpp in vocab-only mode.
