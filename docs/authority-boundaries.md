# Authority boundaries

The five authority flags are always true and always frontier-owned:

| Authority | Worker permission | Final owner |
| --- | --- | --- |
| Scientific interpretation | Propose mechanics only | Frontier/human reviewer |
| Citation decisions | No citation acceptance | Frontier/human reviewer |
| Security severity | Report deterministic checks only | Frontier/human reviewer |
| Canonical-source acceptance | No automatic source update | Frontier/human reviewer |
| Release/publication | No publish, merge, or release action | Frontier/human reviewer |

`automatic_apply=false` is enforced by the contracts. A successful dry run,
unit test, or synthetic benchmark is not a scientific, security, or release
decision.
