import { z } from "zod";

export const DEFAULT_TOP_K = 5;
export const DEFAULT_TIMEOUT_MS = 75_000;
export const MAX_QUESTION_LENGTH = 2_000;

const answerSourceSchema = z
  .object({
    source_name: z.string(),
    source_path: z.string(),
    chunk_id: z.string(),
    section_title: z.string().optional(),
    citation_number: z.number().int().positive().optional(),
  })
  .passthrough();

export const answerResponseSchema = z
  .object({
    answer: z.string(),
    route: z.enum(["rag", "analytics"]),
    status: z.enum(["answered", "abstained"]),
    sources: z.array(answerSourceSchema),
    confidence_note: z.string().optional(),
    query_id: z.uuid().optional(),
  })
  .passthrough();

export const errorResponseSchema = z
  .object({
    error: z
      .object({
        code: z.string(),
        message: z.string(),
      })
      .passthrough(),
  })
  .passthrough();

export type AnswerSource = z.infer<typeof answerSourceSchema>;
export type AnswerResponse = z.infer<typeof answerResponseSchema>;
export type ErrorResponse = z.infer<typeof errorResponseSchema>;

export type CivicLensErrorKind =
  | "configuration"
  | "validation"
  | "backend_unavailable"
  | "server"
  | "network"
  | "timeout"
  | "malformed_response"
  | "api";

export class CivicLensClientError extends Error {
  constructor(
    public readonly kind: CivicLensErrorKind,
    public readonly userMessage: string,
    public readonly code: string,
    public readonly status?: number,
  ) {
    super(userMessage);
    this.name = "CivicLensClientError";
  }
}

export interface AskQuestionOptions {
  baseUrl?: string;
  timeoutMs?: number;
  topK?: number;
  fetchImpl?: typeof fetch;
}

export type AskQuestion = (
  question: string,
  options?: AskQuestionOptions,
) => Promise<AnswerResponse>;

function resolveApiBaseUrl(override?: string): string {
  const configured = (
    override ?? process.env.NEXT_PUBLIC_CIVICLENS_API_BASE_URL ?? ""
  ).trim();
  if (!configured) {
    throw new CivicLensClientError(
      "configuration",
      "The CivicLens API is not configured for this frontend.",
      "missing_api_url",
    );
  }

  try {
    const url = new URL(configured);
    if (
      !["http:", "https:"].includes(url.protocol) ||
      url.username ||
      url.password ||
      (url.pathname !== "/" && url.pathname !== "") ||
      url.search ||
      url.hash
    ) {
      throw new Error("not an HTTP origin");
    }
    return url.origin;
  } catch {
    throw new CivicLensClientError(
      "configuration",
      "The CivicLens API address is invalid.",
      "invalid_api_url",
    );
  }
}

function apiErrorFor(status: number, payload: ErrorResponse): CivicLensClientError {
  if (status === 422) {
    return new CivicLensClientError(
      "validation",
      "That question could not be submitted. Check it and try again.",
      payload.error.code,
      status,
    );
  }
  if (status === 503) {
    return new CivicLensClientError(
      "backend_unavailable",
      "CivicLens is warming up or temporarily unavailable. Please wait a moment and try again.",
      payload.error.code,
      status,
    );
  }
  if (status >= 500) {
    return new CivicLensClientError(
      "server",
      "CivicLens could not complete the request. Please try again later.",
      payload.error.code,
      status,
    );
  }
  return new CivicLensClientError(
    "api",
    "The CivicLens API rejected the request.",
    payload.error.code,
    status,
  );
}

function malformedResponse(): CivicLensClientError {
  return new CivicLensClientError(
    "malformed_response",
    "CivicLens returned an unexpected response. Please try again later.",
    "malformed_response",
  );
}

export const askCivicLens: AskQuestion = async (question, options = {}) => {
  const normalizedQuestion = question.trim();
  const topK = options.topK ?? DEFAULT_TOP_K;
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  if (!normalizedQuestion || normalizedQuestion.length > MAX_QUESTION_LENGTH) {
    throw new CivicLensClientError(
      "validation",
      `Enter a question between 1 and ${MAX_QUESTION_LENGTH.toLocaleString()} characters.`,
      "invalid_question",
    );
  }
  if (!Number.isInteger(topK) || topK < 1 || topK > 100) {
    throw new CivicLensClientError(
      "validation",
      "The retrieval limit is invalid.",
      "invalid_top_k",
    );
  }
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    throw new CivicLensClientError(
      "configuration",
      "The CivicLens request timeout is invalid.",
      "invalid_timeout",
    );
  }

  const controller = new AbortController();
  let timedOut = false;
  const timeout = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  let response: Response;
  try {
    response = await (options.fetchImpl ?? fetch)(
      `${resolveApiBaseUrl(options.baseUrl)}/api/v1/answer`,
      {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        credentials: "omit",
        body: JSON.stringify({ question: normalizedQuestion, top_k: topK }),
        signal: controller.signal,
      },
    );
  } catch (error) {
    if (timedOut || (error instanceof DOMException && error.name === "AbortError")) {
      throw new CivicLensClientError(
        "timeout",
        "CivicLens took too long to respond. A Render cold start may still be in progress; please try again.",
        "request_timeout",
      );
    }
    throw new CivicLensClientError(
      "network",
      "CivicLens could not reach the API. Check your connection and try again.",
      "network_error",
    );
  } finally {
    window.clearTimeout(timeout);
  }

  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw malformedResponse();
  }

  if (!response.ok) {
    const parsedError = errorResponseSchema.safeParse(payload);
    if (!parsedError.success) {
      throw malformedResponse();
    }
    throw apiErrorFor(response.status, parsedError.data);
  }

  const parsedAnswer = answerResponseSchema.safeParse(payload);
  if (!parsedAnswer.success) {
    throw malformedResponse();
  }
  return parsedAnswer.data;
};
