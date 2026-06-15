// libFuzzer target skeleton for a native ML library that ingests untrusted
// input (a model file, a tokenizer blob, a grammar/schema, a regex, etc.).
//
// Target-agnostic. Replace the body of LLVMFuzzerTestOneInput with calls into
// the API you want to harden. The structure below encodes three habits that
// matter for fuzzing this class of code:
//
//   1. Multiplex several entry points behind one harness, selected by the
//      first fuzzer byte, so one campaign exercises every untrusted-input
//      surface instead of just one. (Distinct entry points fail in distinct
//      ways; a single-entry harness leaves most of the attack surface unfuzzed.)
//
//   2. Catch the library's *expected* parse/validation failures. The fuzzer
//      should only terminate the process on a sanitizer fault or an unexpected
//      abort, never on a normal "this input is invalid" exception — otherwise
//      every malformed input looks like a crash and the corpus never deepens.
//
//   3. Keep the harness allocation-light and deterministic so the same crash
//      input reproduces identically under the release-significance triage.
//
// This is scaffolding, not a finished harness for any specific library.

#include <cstdint>
#include <cstddef>
#include <string>

// #include "your_library.h"

namespace {

// Stand-ins for the real API. Replace with the actual calls.
void parse_and_compile(const std::string& input);   // entry point A
void parse_alternate(const std::string& input);      // entry point B
void run_after_compile(const std::string& input);    // entry point C

void dispatch(uint8_t mode, const std::string& input) {
  switch (mode % 3) {
    case 0: parse_and_compile(input); break;
    case 1: parse_alternate(input);   break;
    default: run_after_compile(input); break;
  }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 1) return 0;
  const uint8_t mode = data[0];
  std::string input(reinterpret_cast<const char*>(data + 1), size - 1);

  try {
    dispatch(mode, input);
  } catch (const std::exception&) {
    // Expected: malformed input rejected by the library. Not a finding.
  } catch (...) {
    // Expected: non-std exception from the library's own validation.
  }
  // Sanitizer faults (ASAN/UBSan) abort the process before reaching here and
  // are surfaced by libFuzzer as a crash, which is exactly what we want.
  return 0;
}

// --- replace these stubs with real calls -----------------------------------
namespace {
void parse_and_compile(const std::string&) {}
void parse_alternate(const std::string&) {}
void run_after_compile(const std::string&) {}
}  // namespace
