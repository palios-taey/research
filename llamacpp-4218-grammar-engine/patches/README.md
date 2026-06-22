# Patch status

`llamacpp-4218-gss-recognizer.patch` is a historical first-pass prototype diff. It is retained only as audit background for the original stack-sharing mechanism.

Do not treat this patch as the current fix. The current audited implementation is the public branch linked from the top-level `README.md`:

```text
https://github.com/palios-taey/llama.cpp/tree/codex/4218-rootcause-earley
```

The prototype patch predates later audit findings around multi-range character handling, raw-byte/token handling, clone safety, completion reallocation, and nullable UTF-8 EOG behavior.
