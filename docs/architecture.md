# Architecture

The MVP has five explicit layers:

1. A versioned `WorkOrder` describes a bounded public or public-synthetic task.
2. The router selects one of the four exact baseline architectures and a
   deterministic route.
3. Worker routes are proposal-only; the `DryRunAdapter` makes no provider call.
4. Repository-owned validators check schemas, path scope, protected literals,
   and allowlisted validation IDs.
5. A sanitized result or review bundle is handed to a human/frontier reviewer.

```mermaid
sequenceDiagram
    participant M as Manager
    participant R as Router
    participant W as Worker proposal lane
    participant D as Deterministic validators
    participant H as Human/frontier reviewer
    M->>R: versioned work order
    R-->>W: frontier, hosted_worker, local_worker, or blocked
    W-->>D: proposal receipt only
    D-->>H: sanitized result and metrics
    H-->>M: accept, repair, reject, or keep blocked
```

The repository does not contain a provider SDK, model runtime, automatic patch
application, or release action.
