# Render and Vercel Deployment Proof

> **NON-PRODUCTION PORTFOLIO DEMO**
>
> A live, dated Render smoke test passed on 2026-08-23 UTC. This is deployment
> proof for a time-limited portfolio demo, not a production availability or
> reliability commitment.
>
> Final Issue 19 browser-to-backend validation was recorded on 2026-08-26 UTC
> against the Vercel frontend and Render API documented below.

## Purpose and target

Issue 19 adds a Vercel-hosted Next.js product UI to the existing Render
deployment without changing the FastAPI or RAG boundaries:

```text
Browser
  -> Vercel Next.js
  -> Render FastAPI public contract
  -> CivicLens orchestration
  -> approved analytics or hybrid RAG
  -> externally managed Neon PostgreSQL + pgvector database
```

Streamlit remains the separate engineering and debugging UI. Next.js contains
no API proxy, provider client, database client, retrieval logic, or citation
reconstruction.

Render was selected for Docker-based web services, declarative Blueprints, and
straightforward teardown. The current hosted database is Neon PostgreSQL with
pgvector. The deployment uses Render Free services for dated portfolio proof.
Their sufficiency for this bounded smoke test was validated; ongoing
availability remains subject to Free-tier cold starts and platform limits.

## Externally managed database assumption

The current hosted database is an externally managed Neon Free PostgreSQL
instance in AWS US West 2 (Oregon) with pgvector enabled. The Blueprint
deliberately contains no `databases` block and does not create, replace,
resize, or otherwise manage that database. The previous Render PostgreSQL
database remains outside the Blueprint and is retained as a rollback target.

`civiclens-api` declares `DATABASE_URL` with `sync: false`; its Neon Direct
connection string is managed only in the Render Dashboard. Blueprint updates
preserve the existing Dashboard value and do not prompt for it again. Never
copy the URL or its credentials into Git, documentation, screenshots, or logs.

The two web services are top-level Blueprint resources. They do not take
ownership of the manually created Render Project or its `Demo` environment.
After deployment, they can be moved into that project/environment manually if
desired.

## Blueprint services

### Source branch behavior

During the original live Blueprint creation, the Blueprint definition was selected from
the `issue-15-cloud-deployment` branch. Although neither service sets a
`branch` in `render.yaml`, Render linked both created services to that selected
Blueprint branch. This was verified from the live service configuration and
deploy logs. That records the original deployment state; the hosted services
are now aligned with `main`.

`render.yaml` creates only:

- `civiclens-api`: Free Docker web service built with `Dockerfile.api`, with
  `/health` as its Render health check.
- `civiclens-ui`: Free Docker web service built with `Dockerfile.ui`, with
  `/_stcore/health` as its Render health check.

Automatic deploys are disabled. The checked-in Blueprint aligns with the
verified hosted runtime: deterministic `local-deterministic-1536` embeddings,
the pgvector dense store, hybrid retrieval, and `ANSWER_PROVIDER=openai` for
grounded RAG answer generation. OpenAI generation
is constrained to retrieved evidence, and CivicLens validates citations before
returning the public response. `ANSWER_PROVIDER` is the single source of truth;
the removed `USE_OPENAI_ANSWERS` flag is not a runtime input.
`CIVICLENS_CORS_ALLOWED_ORIGINS` remains a Blueprint `sync: false` value so the
exact stable Vercel production origin is configured server-side rather than
committed.

The UI receives the generated HTTPS API URL through the API service's
`RENDER_EXTERNAL_URL`; no `onrender.com` hostname is hardcoded. Its request
timeout is 60 seconds to accommodate Free-service cold starts.

## Bootstrap strategy

The API service uses exactly one initialization mechanism and gates API
startup on its success:

```text
dockerCommand: /bin/sh scripts/render_start.sh
```

The small startup script runs `python -m scripts.bootstrap` and only then
replaces itself with Uvicorn. Keeping the sequencing in a checked-in script
avoids command-quoting ambiguity in Render's Docker command override.

The first live Blueprint deploy did not execute its configured
`initialDeployHook`: the service reached Live and started Uvicorn without any
bootstrap, migration, ingestion, chunking, or embedding log entries. The
startup command is therefore the reliable Free-service fallback. Bootstrap
must complete before Uvicorn starts; a bootstrap failure keeps the API from
claiming liveness or readiness. No paid `preDeployCommand` is used.

The existing bootstrap remains unchanged and runs:

```text
ordered migrations -> manifest ingestion -> section-aware chunking
                   -> PostgreSQL metadata + dense-vector synchronization
```

It uses stable IDs and upserts, does not reset the database, and never performs
an automatic destructive reindex. An incompatible stored embedding profile
fails clearly and requires explicit operator review. Historical migrations
`0001` and `0002` remain unchanged; the current migration set also includes
`0003_bounded_orchestration_metadata.sql`.

The command can run again on a deploy, restart, or Free-service cold start.
That tradeoff is acceptable for this time-limited demo because migrations are
version/checksum tracked and ingestion, chunking, and compatible embeddings use
stable IDs and upserts. The command never requests destructive reindexing. An
incompatible stored embedding profile still fails and requires explicit
operator review.

## Health and readiness

- During bootstrap, the FastAPI process may not yet be available. After
  bootstrap succeeds, `scripts/render_start.sh` starts Uvicorn.
- `GET /health` then checks only that the FastAPI process is alive.
- `GET /ready` checks the PostgreSQL schema, current manifest documents and
  chunks, embedding profile, and vector availability.

After startup bootstrap succeeds and Uvicorn is running, `/ready` must return
`200` to verify that the prepared RAG/PostgreSQL backend is ready before
deployment proof is accepted.

## Dated deployment evidence

The following secret-free checks were completed against the existing
`civiclens-rag-demo` Blueprint on 2026-08-23 UTC (2026-08-22 EDT):

- Validated application commit: `5596d90b68428b7a55133036830786c98b7be614`.
- API: <https://civiclens-api-o8ap.onrender.com>.
- Streamlit: <https://civiclens-ui.onrender.com>.
- API `GET /health`: HTTP `200`, `{"status":"ok"}`.
- API `GET /ready`: HTTP `200`, status `ready`.
- Bootstrap: migrations `0001` and `0002` tracked as already applied on the
  successful deployment; 7 documents, 87 current chunks, and 87 compatible
  deterministic embeddings were stored in the existing PostgreSQL database.
- Cited-answer smoke: `What does complaint_type mean?` returned
  `route=rag`, `status=answered`, and two validated sources with stable chunk
  IDs and source paths.
- Safe-no-answer smoke: `Explain the orbital pineapple parking treaty.`
  returned `route=rag`, `status=abstained`, the normal safe answer text, and
  zero sources.
- Browser-driven Streamlit smoke: the hosted UI submitted the supported
  question through FastAPI and rendered the answered route/status plus the
  validated NYC 311 field-guide and RAG-design source metadata.

The public response fields were limited to the provider-neutral answer
contract. No retrieved chunk text, provider diagnostics, database URL,
credentials, or raw provider payload appeared. OpenAI was disabled throughout.

## Issue 19 hosted deployment proof

- Frontend: <https://civiclens-rag-nyc311.vercel.app>
- API: <https://civiclens-api-o8ap.onrender.com>
- API `GET /health`: HTTP `200`, status `ok`.
- API `GET /ready`: HTTP `200`, status `ready`.
- Database: externally managed Neon PostgreSQL + pgvector with 7 documents,
  77 chunks, and 77 deterministic 1536-dimensional embeddings.
- CORS: the explicit Render allowlist granted the stable Vercel origin
  `https://civiclens-rag-nyc311.vercel.app`; no wildcard origin was used.
- Grounded RAG: a supported documentation question reached the configured
  OpenAI answer provider through hybrid retrieval and rendered the grounded
  answer with CivicLens-validated citations and provenance.
- Approved analytics: the hosted browser rendered an answered analytics result
  from the existing deterministic allowlisted tools over checked-in sample CSV
  outputs. No internal tool rows or unrestricted SQL interface were exposed.
- Safe abstention: an unsupported question returned the normal abstained status,
  safe answer text, and zero sources rather than a system error or fabricated
  citation.

Secret-free hosted evidence is preserved in the grounded RAG, analytics, and
safe-abstention screenshots under `docs/screenshots/`. The browser request path
remains Vercel Next.js directly to Render FastAPI, then CivicLens orchestration
and Neon PostgreSQL + pgvector for hybrid RAG. Render Free instances may
cold-start, so the first request can take longer than subsequent requests. This
is a non-production portfolio deployment and carries no availability,
reliability, or live NYC 311 service claim.

## Manual Blueprint deployment

1. Review the current `render.yaml` on the deployment branch or `main` before
   syncing the Blueprint.
2. In the Render Dashboard, open **Blueprints** and choose **New Blueprint
   Instance**.
3. Select the `civiclens-rag-nyc311` repository and the branch containing the
   reviewed `render.yaml`.
4. Confirm Render detects the root `render.yaml`.
5. Confirm the plan contains exactly two new Free web services,
   `civiclens-api` and `civiclens-ui`.
6. Confirm `DATABASE_URL` is declared with `sync: false`, the existing
   Dashboard-managed Neon value is preserved, and no database resource appears
   in the plan.
7. Confirm both services are in Oregon and automatic deploys are disabled.
8. Sync/deploy the Blueprint and inspect build, bootstrap, and runtime logs.
9. Wait for `python -m scripts.bootstrap` to report successful migrations,
   ingestion, chunking, and embedding storage.
10. Open the generated API URL and verify `/health`, then `/ready`.
11. Open the generated Streamlit URL and verify that it can call the API.
12. Only after readiness succeeds, run one cited-answer test and one safe
    no-answer test and verify returned provenance.
13. Save dated, secret-free screenshots or logs and the live URLs if the demo
    will remain available.

Do not delete the previous Render PostgreSQL database during the rollback
retention period.

## Reproduction validation checklist

Live proof requires all of the following:

- API `/health` returns `200`.
- API `/ready` returns `200` after bootstrap.
- The Vercel origin receives approved CORS access from Render FastAPI.
- The Vercel frontend calls the public FastAPI answer contract directly.
- Streamlit can submit a question through the generated API HTTPS URL.
- A documentation question returns a grounded answer with validated source
  identifiers and provenance.
- An approved analytics question returns through its deterministic allowlisted
  tool path.
- An unsupported question returns the safe no-answer behavior.
- No raw database URL, password, provider payload, or stack trace appears in
  public responses or captured evidence.

The dated Issue 15 and Issue 19 checks above satisfy this checklist for the
recorded non-production deployment.

## Free-tier and cost assumptions

- Both web services use the Free plan. Under Render's current policy they spin
  down after 15 minutes without inbound traffic, causing cold starts.
- Free usage is subject to the workspace's runtime, bandwidth, and build-minute
  allowances.
- The externally managed Neon Free database is subject to its own service and
  retention limits; verify current Neon policy before relying on it.
- This design has no high availability, autoscaling, disaster recovery,
  production secrets manager, authentication, or SLA.
- Check the current Render pricing and Free-tier policy before deployment;
  platform limits can change.

## Shutdown and teardown

For a time-limited proof, suspend or delete `civiclens-ui` and `civiclens-api`
from Render after capturing dated evidence. Disabling automatic deploys avoids
unplanned redeploys. Review workspace usage and spend limits in the Dashboard.

The Neon database is an externally managed resource. Do not delete it as part
of web-service teardown unless the owner separately decides that its data is
no longer needed. Keep `DATABASE_URL` operator-managed and secret. Retain the
previous Render PostgreSQL database until its rollback role is no longer
required.

## Limitations

- Free web services can cold-start and cannot receive private-network traffic;
  the UI therefore uses the API's generated external HTTPS URL.
- Bootstrap runs before each API process start, so Free-service cold starts can
  take longer than a plain Uvicorn start.
- The live URLs are time-limited evidence and may later be suspended or removed
  under the documented teardown procedure.
- This is a curated, bounded, non-production portfolio demo, not a production
  NYC 311 service or a production reliability claim.
