import { describe, expect, it, vi } from "vitest";

import { CivicLensClientError, askCivicLens } from "./api-client";

const BASE_URL = "https://civiclens-api.example";

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function answerPayload() {
  return {
    answer: "Use the documented field definition [7].",
    route: "rag",
    status: "answered",
    sources: [
      {
        source_name: "NYC 311 Service Request Field Guide",
        source_path: "docs/knowledge/nyc311-service-request-fields.md",
        chunk_id: "chunk_field_definition",
        section_title: "Problem / Complaint Type",
        citation_number: 7,
        future_source_field: "preserved",
      },
    ],
    confidence_note: "Citations validated.",
    future_response_field: true,
  };
}

describe("askCivicLens", () => {
  it("posts the public request and validates a future-safe answer response", async () => {
    const fetchMock = vi.fn(async () => jsonResponse(answerPayload()));

    const response = await askCivicLens("  What does complaint_type mean?  ", {
      baseUrl: `${BASE_URL}/`,
      fetchImpl: fetchMock as typeof fetch,
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock).toHaveBeenCalledWith(
      `${BASE_URL}/api/v1/answer`,
      expect.objectContaining({
        method: "POST",
        credentials: "omit",
        body: JSON.stringify({
          question: "What does complaint_type mean?",
          top_k: 5,
        }),
      }),
    );
    expect(response.sources[0].citation_number).toBe(7);
    expect(response.future_response_field).toBe(true);
    expect(response.sources[0].future_source_field).toBe("preserved");
  });

  it.each([
    [422, "invalid_request", "validation"],
    [500, "internal_error", "server"],
    [503, "backend_unavailable", "backend_unavailable"],
  ] as const)(
    "validates and maps a sanitized %i API error",
    async (status, code, expectedKind) => {
      const fetchMock = vi.fn(async () =>
        jsonResponse({ error: { code, message: "Sanitized backend message." } }, status),
      );

      const request = askCivicLens("What is CivicLens?", {
        baseUrl: BASE_URL,
        fetchImpl: fetchMock as typeof fetch,
      });

      await expect(request).rejects.toMatchObject({
        kind: expectedKind,
        code,
        status,
      });
    },
  );

  it("maps network failures without exposing the raw exception", async () => {
    const fetchMock = vi.fn(async () => {
      throw new Error("socket detail with private infrastructure");
    });

    const request = askCivicLens("What is CivicLens?", {
      baseUrl: BASE_URL,
      fetchImpl: fetchMock as typeof fetch,
    });

    await expect(request).rejects.toMatchObject({
      kind: "network",
      code: "network_error",
    });
    await expect(request).rejects.not.toThrow("private infrastructure");
  });

  it("aborts and reports a bounded timeout", async () => {
    const fetchMock = vi.fn(
      (_input: RequestInfo | URL, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => {
            reject(new DOMException("Aborted", "AbortError"));
          });
        }),
    );

    const request = askCivicLens("What is CivicLens?", {
      baseUrl: BASE_URL,
      timeoutMs: 5,
      fetchImpl: fetchMock as typeof fetch,
    });

    await expect(request).rejects.toMatchObject({
      kind: "timeout",
      code: "request_timeout",
    });
  });

  it("rejects malformed JSON", async () => {
    const fetchMock = vi.fn(async () => new Response("not-json", { status: 200 }));

    const request = askCivicLens("What is CivicLens?", {
      baseUrl: BASE_URL,
      fetchImpl: fetchMock as typeof fetch,
    });

    await expect(request).rejects.toMatchObject({
      kind: "malformed_response",
      code: "malformed_response",
    });
  });

  it("rejects valid JSON with the wrong public schema", async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({ answer: "Missing route, status, and sources." }),
    );

    const request = askCivicLens("What is CivicLens?", {
      baseUrl: BASE_URL,
      fetchImpl: fetchMock as typeof fetch,
    });

    await expect(request).rejects.toBeInstanceOf(CivicLensClientError);
    await expect(request).rejects.toMatchObject({ kind: "malformed_response" });
  });
});
