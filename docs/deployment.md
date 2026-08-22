# Render Deployment Proof Plan

> **NON-PRODUCTION PORTFOLIO DEMO**
>
> Deployment configuration is prepared; live Render proof is pending. Do not
> treat this document as evidence that CivicLens is currently deployed.

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
The deployment targets Render Free services for dated portfolio proof. Live
resource sufficiency remains to be validated during the cloud smoke test.

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

During Phase 1, the Blueprint definition is reviewed and selected from the
`issue-15-cloud-deployment` branch. Neither service sets a `branch`. Under
Render's current Blueprint behavior, a service without that setting uses the
repository's default branch, which is currently `main`. This is intentional:
Phase 1 adds deployment configuration, documentation, and tests without
changing the application runtime, so the services use the merged Issue 14
application baseline on `main` rather than pinning a temporary feature branch.

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

The API service uses exactly one initialization mechanism:

```text
initialDeployHook: python -m scripts.bootstrap
```

Render runs the hook once after the API service's first successful deploy.
This is preferable to placing bootstrap in the process start command because a
Free service can cold-start repeatedly. It also avoids `preDeployCommand`,
which Render reserves for paid web services.

The existing bootstrap remains unchanged and runs:

```text
ordered migrations -> manifest ingestion -> section-aware chunking
                   -> deterministic embeddings/index preparation
```

It uses stable IDs and upserts, does not reset the database, and never performs
an automatic destructive reindex. An incompatible stored embedding profile
fails clearly and requires explicit operator review. Historical migrations
`0001` and `0002` are used unchanged.

`initialDeployHook` is a one-time initialization hook, not an ongoing migration
or corpus-update mechanism. Future schema or corpus changes need a separately
reviewed operator workflow. If the hook does not complete successfully, stop
the deployment review and investigate the Render logs; do not claim the RAG
backend is ready.

## Health and readiness

- `GET /health` checks only that FastAPI is alive. Render uses this endpoint
  while the first bootstrap may still be pending.
- `GET /ready` checks the PostgreSQL schema, current manifest documents and
  chunks, embedding profile, and vector availability.

During initialization, `/health` can correctly return `200` while `/ready`
returns `503`. After the hook succeeds, `/ready` must return `200` before any
deployment proof is accepted.

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
8. Sync/deploy the Blueprint and inspect build, deploy, and initial-hook logs.
9. Wait for `python -m scripts.bootstrap` to report successful migrations,
   ingestion, chunking, and embedding storage.
10. Open the generated API URL and verify `/health`, then `/ready`.
11. Open the generated Streamlit URL and verify that it can call the API.
12. Only after readiness succeeds, run one cited-answer test and one safe
    no-answer test and verify returned provenance.
13. Save dated, secret-free screenshots or logs and the live URLs if the demo
    will remain available.

Do not delete or recreate `civiclens-postgres` during this process.

## Validation plan

Live proof requires all of the following:

- API `/health` returns `200`.
- API `/ready` returns `200` after bootstrap.
- Streamlit can submit a question through the generated API HTTPS URL.
- A documentation question returns a grounded answer with validated source
  identifiers and provenance.
- An unsupported question returns the safe no-answer behavior.
- No raw database URL, password, provider payload, or stack trace appears in
  public responses or captured evidence.

Until those checks run against Render, Issue 15 remains incomplete.

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

- Live deployment, cloud readiness, URLs, screenshots, and smoke results are
  still pending.
- Free web services can cold-start and cannot receive private-network traffic;
  the UI therefore uses the API's generated external HTTPS URL.
- The one-time initialization hook does not automatically bootstrap later
  corpus or migration changes.
- This is a curated, deterministic portfolio demo, not a production NYC 311
  service or a production reliability claim.
