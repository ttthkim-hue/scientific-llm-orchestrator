# Privacy model

The MVP uses two safe input classes: `public` and `public-synthetic`. The
fixtures in this tree are synthetic. A future adapter must reject private or
local-only material unless a separate, documented authorization layer exists.

The public contracts intentionally omit raw prompts, raw responses, telemetry
databases, credentials, local absolute paths, usernames, hostnames, model
blobs, and provider quota details. Metrics use explicit nulls for values that
were not measured. The publication scanner checks common secret, path, URL,
personal-identifier, binary, and unsupported-claim boundaries.

Passing the scanner is only deterministic acceptance. A human semantic review
is still required before publication.
