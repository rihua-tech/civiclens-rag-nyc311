# Render Deployment Proof

> **NON-PRODUCTION PORTFOLIO DEMO**
>
> A live, dated Render smoke test passed on 2026-08-23 UTC. This is deployment
> proof for a time-limited portfolio demo, not a production availability or
> reliability commitment.

## Purpose and target

Issue 15 uses Render for one small, time-limited cloud deployment proof while
preserving the Issue 14 architecture:

```text
Streamlit Web Service
        |
        | HTTPS through the public FastAPI contract
        v
FastAPI Web Service
        |
        v
shared orchestration -> analytics or RAG -> grounded generation
        |
        v
existing Render PostgreSQL/pgvector database
```

Render was selected because it supports Docker-based web services, managed
PostgreSQL with pgvector, declarative Blueprints, and straightforward teardown.
The deployment uses Render Free services for dated portfolio proof. Their
sufficiency for this bounded smoke test was validated; ongoing availability
remains subject to Free-tier cold starts and platform limits.

## Existing-resource assumption

The Render workspace already contains one PostgreSQL instance named
`civiclens-postgres` in Oregon. Public/external database access is disabled.
The Blueprint deliberately contains no `databases` block and does not create,
replace, resize, or otherwise manage that database.

`civiclens-api` receives `DATABASE_URL` through a `fromDatabase` reference to
the existing database's internal `connectionString`. Never copy the internal
URL or its credentials into Git, documentation, screenshots, or logs. The
Blueprint must be created in the same Render workspace as the existing
database or the reference will fail to resolve.

The two web services are top-level Blueprint resources. They do not take
ownership of the manually created Render Project or its `Demo` environment.
After deployment, they can be moved into that project/environment manually if
desired.

## Blueprint services

### Source branch behavior

During the live Blueprint creation, the Blueprint definition was selected from
the `issue-15-cloud-deployment` branch. Although neither service sets a
`branch` in `render.yaml`, Render linked both created services to that selected
Blueprint branch. This was verified from the live service configuration and
deploy logs. Before the feature branch is eventually deleted, the services
must be relinked to `main` after the reviewed Issue 15 changes are merged.

`render.yaml` creates only:

- `civiclens-api`: Free Docker web service built with `Dockerfile.api`, with
  `/health` as its Render health check.
- `civiclens-ui`: Free Docker web service built with `Dockerfile.ui`, with
  `/_stcore/health` as its Render health check.

Automatic deploys are disabled. The API uses deterministic 1536-dimensional
embeddings, hybrid retrieval, disabled reranking, local deterministic answers,
disabled OpenAI paths, and disabled observability. No OpenAI key is needed.

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
                   -> deterministic embeddings/index preparation
```

It uses stable IDs and upserts, does not reset the database, and never performs
an automatic destructive reindex. An incompatible stored embedding profile
fails clearly and requires explicit operator review. Historical migrations
`0001` and `0002` are used unchanged.

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

## Manual Blueprint deployment

1. Review, commit, and push the Issue 15 branch containing `render.yaml` only
   after local human review.
2. In the Render Dashboard, open **Blueprints** and choose **New Blueprint
   Instance**.
3. Select the `civiclens-rag-nyc311` repository and the branch containing the
   reviewed `render.yaml`.
4. Confirm Render detects the root `render.yaml`.
5. Confirm the plan contains exactly two new Free web services,
   `civiclens-api` and `civiclens-ui`.
6. Confirm `DATABASE_URL` references the existing `civiclens-postgres`; no new
   PostgreSQL instance should appear in the plan.
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

Do not delete or recreate `civiclens-postgres` during this process.

## Reproduction validation checklist

Live proof requires all of the following:

- API `/health` returns `200`.
- API `/ready` returns `200` after bootstrap.
- Streamlit can submit a question through the generated API HTTPS URL.
- A documentation question returns a grounded answer with validated source
  identifiers and provenance.
- An unsupported question returns the safe no-answer behavior.
- No raw database URL, password, provider payload, or stack trace appears in
  public responses or captured evidence.

The dated checks above satisfy this checklist for the recorded deployment.

## Free-tier and cost assumptions

- Both web services use the Free plan. Under Render's current policy they spin
  down after 15 minutes without inbound traffic, causing cold starts.
- Free usage is subject to the workspace's runtime, bandwidth, and build-minute
  allowances.
- Free Render PostgreSQL is limited to 1 GB, has no managed backups, and under
  Render's current policy expires 30 days after creation unless upgraded.
- This design has no high availability, autoscaling, disaster recovery,
  production secrets manager, authentication, or SLA.
- Check the current Render pricing and Free-tier policy before deployment;
  platform limits can change.

## Shutdown and teardown

For a time-limited proof, suspend or delete `civiclens-ui` and `civiclens-api`
from Render after capturing dated evidence. Disabling automatic deploys avoids
unplanned redeploys. Review workspace usage and spend limits in the Dashboard.

The database is an existing separately managed resource. Do not delete it as
part of web-service teardown unless the owner separately decides that its data
is no longer needed. Keeping external database access disabled is the expected
security posture.

## Limitations

- Free web services can cold-start and cannot receive private-network traffic;
  the UI therefore uses the API's generated external HTTPS URL.
- Bootstrap runs before each API process start, so Free-service cold starts can
  take longer than a plain Uvicorn start.
- The live URLs are time-limited evidence and may later be suspended or removed
  under the documented teardown procedure.
- This is a curated, deterministic portfolio demo, not a production NYC 311
  service or a production reliability claim.
