# Harnesses

The active current-branch harnesses are copied under `harnesses/current/`:

- `test-grammar-compactor.cpp`
- `test-grammar-differential.cpp`

They are intended to match the files added to the public branch at commit `3e4aa92bed972d60cbf9a02795d40bed10a60338`.

## Current Branch Usage

From a checkout of the public branch:

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

./build-compactor/bin/test-grammar-compactor
./build-compactor/bin/test-grammar-compactor --long
./build-compactor/bin/test-grammar-differential
ctest --test-dir build-compactor -R "(grammar|gbnf)" --output-on-failure
```

For sanitizer coverage:

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

## Historical Harnesses

The other harness files in this directory are retained for historical reproduction of the first-pass prototype and early root-cause measurements. They are not the active current-branch validation gate.
