# Public inclusion and exclusion matrix

| Candidate material | Public MVP decision | Reason |
| --- | --- | --- |
| Provider-neutral contracts and validators | Include | Small, testable, standard-library implementation |
| Synthetic coding/data/visual/CAD fixtures | Include | Public-safe and reproducible |
| Private implementation history or repository metadata | Exclude | Clean-room boundary |
| Unpublished manuscripts, figures, raw data, and simulator outputs | Exclude | Research and privacy protection |
| Provider credentials, quotas, private URLs, and account records | Exclude | Security and terms boundary |
| Model weights, blobs, telemetry databases, and caches | Exclude | Size, licensing, and provenance boundary |
| Actual-model benchmark results | Exclude from this build | `not-run`; no provider call authorized |
| Promotion, applications, merge, push, release, or publication action | Exclude | Separate approval gate |

The machine-readable file [`../publication-manifest.json`](../publication-manifest.json)
records the candidate files, hashes, classifications, license status, reviewer
status, deterministic decision, and reason. Its own hash is explicitly
excluded to avoid recursion.
