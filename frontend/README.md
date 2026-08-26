# CivicLens Next.js frontend

This directory contains the recruiter-facing CivicLens product UI. It is a
presentation layer and typed browser client for the existing FastAPI contract;
it contains no retrieval, generation, analytics-tool, citation, database, or
provider logic.

## Local setup

Use Node 22 (the `.nvmrc` version), then install from the committed lockfile:

```bash
npm ci
```

Copy `.env.example` to `.env.local` and configure the public FastAPI origin:

```dotenv
NEXT_PUBLIC_CIVICLENS_API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_CIVICLENS_SITE_URL=http://localhost:3000
```

Both URLs are public configuration, not credentials. The site URL supplies the
canonical origin for build-time Open Graph and social image metadata. Never
place API keys, database URLs, provider settings, or other secrets in
`NEXT_PUBLIC_*` variables.

Start FastAPI separately, then run:

```bash
npm run dev
```

The frontend uses direct browser-to-FastAPI requests. FastAPI must allow the
frontend's exact origin through `CIVICLENS_CORS_ALLOWED_ORIGINS`.

## Quality commands

```bash
npm run lint
npm run typecheck
npm test
npm run build
```

Tests mock the browser HTTP boundary and do not call Render, Vercel, OpenAI,
Pinecone, model registries, or other live services.

## Vercel deployment boundary

1. Create or select the Vercel project only after repository review.
2. Set `NEXT_PUBLIC_CIVICLENS_API_BASE_URL` to the existing Render FastAPI
   origin.
3. Obtain the stable Vercel production origin and set
   `NEXT_PUBLIC_CIVICLENS_SITE_URL` to that exact HTTPS origin.
4. Add that exact origin to the Render API's
   `CIVICLENS_CORS_ALLOWED_ORIGINS` value, preserving approved local origins if
   still needed.
5. Redeploy or restart FastAPI, then verify CORS preflight, RAG, analytics, and
   safe-abstention behavior from the hosted browser.

Do not use a Vercel route handler, Server Action, or server-side proxy for the
CivicLens answer path. This is a non-production portfolio interface without an
SLA, authentication, rate limiting, high availability, or live NYC 311 data.
