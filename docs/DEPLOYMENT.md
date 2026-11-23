# Deployment

The application is one container. Everything it needs at runtime - the parsed
corpus, both indexes and the embedding model - is baked into the image, so a
cold start is a process start, not a download.

## Build and run locally

```bash
make docker-build
make docker-run          # http://127.0.0.1:8000
```

Or without Docker:

```bash
make venv
make data                # ~1 min   downloads ~24 MB from CMS
make index               # ~26 min  embeds 98,186 descriptions on CPU
make serve
```

`make data` and `make index` are only needed if you want to rebuild from
source. The repository already contains the parsed corpus and the built
indexes, so a fresh clone can run `make serve` directly.

## Deploying to Render (free tier)

The repository contains a `render.yaml` blueprint, so the service can be
created from the repo rather than from dashboard settings.

1. Push the repository to GitHub.
2. In Render: **New → Blueprint**, point it at the repository, apply.
3. Render builds the Dockerfile and serves it. The health check path is
   `/api/health`.

Alternatively, without the blueprint: **New → Web Service**, connect the repo,
choose **Docker** as the runtime, leave the build and start commands empty
(the Dockerfile defines both) and set the health check path to `/api/health`.

### What to expect on the free tier

* The service sleeps after a period of inactivity and takes some seconds to
  wake. The front end detects the 503 that a waking container returns and shows
  "this is a free-tier container waking up" rather than a raw error.
* Memory is the binding constraint, not CPU. The image is built for a single
  worker with `OMP_NUM_THREADS=1`; raising either will push resident memory up.
* `PORT` is injected by the platform and the container honours it.

The same image runs unchanged on Fly.io, Railway, Cloud Run or any host that
can run a container and route a port.

## Configuration

Every setting has a working default; these are the ones worth changing.

| Variable | Default | Effect |
|---|---|---|
| `PORT` | `8000` | Port the server binds |
| `ICD10_RATE_LIMIT_REQUESTS` | `60` | Requests allowed per window, per client IP |
| `ICD10_RATE_LIMIT_WINDOW` | `60` | Window length in seconds |
| `ICD10_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Must match the model the index was built with, or startup fails loudly |
| `ICD10_RRF_K` | `10` | Fusion damping constant |
| `ICD10_RRF_WEIGHT_DENSE` | `1.0` | Weight of the semantic arm |
| `ICD10_RRF_WEIGHT_LEXICAL` | `1.0` | Weight of the keyword arm |
| `ICD10_DOC_TEXT_VARIANT` | `desc_incl_parent` | Which text is embedded; changing it requires a rebuild |
| `OMP_NUM_THREADS` | `1` | ONNX Runtime thread count |

## Operational notes

* `/api/health` reports liveness and index readiness separately, so a platform
  health check will not kill the container while the model is still loading.
* The rate limiter is in-process. It is correct for a single instance, which is
  what the free tier runs; behind more than one instance it would need to move
  to shared storage. This is a known limitation, not an oversight.
* CORS is open because the API is a public, read-only demo.
* There is no database, no cache and no background worker. Nothing to back up,
  nothing to migrate, nothing that can drift out of sync with the index.
