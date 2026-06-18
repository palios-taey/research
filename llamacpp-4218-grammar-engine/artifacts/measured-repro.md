# Measured reproduction

## Minimal grammar

```gbnf
root ::= nest
nest ::= "a" nest "b" | "a" nest "c" |
```

For accepted input `a^n b^n`, each `a` branches into two recursive alternatives and the empty alternative. The concrete continuation-stack implementation keeps those alternatives as separate stack vectors even when they later reconverge. The observed maximum stack count follows `3 * 2^n`.

| n | Input length | Expected max stacks | Observed baseline max stacks |
|---:|---:|---:|---:|
| 8 | 16 | 768 | 768 |
| 12 | 24 | 12,288 | 12,288 |
| 16 | 32 | 196,608 | 196,608 |

Representative measured baseline timings for `n = 16`:

| Run | Max stacks | Total accept time | Slowest accept step | Max RSS |
|---|---:|---:|---:|---:|
| Initial standalone repro | 196,608 | 32,799.244 ms | 25,187.073 ms | 53,072 KB |
| Validation harness baseline | 196,608 | 42,900.235 ms | not reported here | not reported here |

The validation harness reached timeout at deeper settings on the baseline path. The candidate GSS path reduced the `n = 16` case to 8 active stacks and `0.034 ms` total accept time.
