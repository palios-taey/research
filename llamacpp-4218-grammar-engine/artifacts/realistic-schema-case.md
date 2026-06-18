# Realistic recursive schema case

The realistic fixture is a Pydantic-style recursive workflow schema compiled to GBNF. The problematic shape is a recursive union where the distinguishing `strategy` discriminator appears late in each object:

```json
{
  "root": {
    "title": "group-13",
    "children": [
      { "title": "group-12", "children": [ ... ], "strategy": "sequential" }
    ],
    "strategy": "sequential"
  }
}
```

Because the discriminator is late, the grammar must carry multiple possible object types across nested child lists before it can reject the wrong branch. That reproduces the same reconvergent-stack growth seen in the synthetic family.

| Fixture | Depth | Baseline max stacks | Baseline total accept time | Prototype max stacks | Prototype total accept time |
|---|---:|---:|---:|---:|---:|
| Recursive workflow, trailing discriminator | 13 | 81,920 | 57,554.288 ms | 10 | 0.371 ms |

Controls:

| Control | Max stacks |
|---|---:|
| Recursive workflow with strategy-first object layout | 6 |
| Flat tool-call union without recursive object union | 9 |

The result supports that the issue is not JSON syntax itself; it is reconvergent ambiguity in recursive schemas. The prototype measurement demonstrates the mechanism, not final patch correctness.
