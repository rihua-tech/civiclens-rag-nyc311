"use client";

import { FormEvent, KeyboardEvent, useRef, useState } from "react";

import {
  AnswerResponse,
  AskQuestion,
  CivicLensClientError,
  MAX_QUESTION_LENGTH,
  askCivicLens,
} from "@/lib/api-client";

const EXAMPLE_QUESTIONS = [
  "What does complaint_type mean?",
  "What are the top complaint types?",
  "What is the local retrieval and cited answer flow?",
  "What is the verified Issue 19 browser request path and hosted RAG configuration?",
  "Explain the orbital pineapple parking treaty.",
] as const;

const PRIMARY_EXAMPLE_COUNT = 2;

type ProductIconName =
  | "analytics"
  | "citation"
  | "database"
  | "evidence"
  | "retrieval"
  | "sparkle"
  | "shield";

const CAPABILITIES = [
  {
    icon: "evidence",
    label: "Grounded RAG",
    description: "Answers grounded in documentation",
  },
  {
    icon: "analytics",
    label: "Analytics",
    description: "Approved sample analytics",
  },
  {
    icon: "shield",
    label: "Safe abstention",
    description: "No answer when evidence is insufficient",
  },
] as const satisfies ReadonlyArray<{
  icon: ProductIconName;
  label: string;
  description: string;
}>;

const TECHNICAL_PROOF = [
  {
    icon: "retrieval",
    label: "Hybrid retrieval",
    description: "Semantic + lexical evidence",
  },
  {
    icon: "citation",
    label: "Validated citations",
    description: "Backend-owned provenance",
  },
  {
    icon: "shield",
    label: "Safe abstention",
    description: "No unsupported answer",
  },
  {
    icon: "database",
    label: "PostgreSQL + pgvector",
    description: "Retrieval infrastructure",
  },
] as const satisfies ReadonlyArray<{
  icon: ProductIconName;
  label: string;
  description: string;
}>;

interface CivicLensExperienceProps {
  askQuestion?: AskQuestion;
}

function ProductIcon({ name }: { name: ProductIconName }) {
  return (
    <svg
      aria-hidden="true"
      className="product-icon"
      fill="none"
      focusable="false"
      viewBox="0 0 24 24"
    >
      {name === "evidence" && (
        <>
          <path d="M7 3.75h7l3 3v13.5H7z" />
          <path d="M14 3.75v3h3M9.5 13l1.6 1.6 3.6-3.6" />
        </>
      )}
      {name === "analytics" && (
        <>
          <path d="M4 19.5h16M6 17v-5M12 17V6.5M18 17V9" />
          <path d="m5.5 8.5 4-3 3.5 2 5-4" />
        </>
      )}
      {name === "shield" && (
        <>
          <path d="M12 3.5 19 6v5c0 4.5-2.9 7.8-7 9.5-4.1-1.7-7-5-7-9.5V6z" />
          <path d="M9 12h6" />
        </>
      )}
      {name === "retrieval" && (
        <>
          <path d="m7 8.5 5-3 5 3M7 15.5l5 3 5-3M7 8.5v7M17 8.5v7" />
          <circle cx="7" cy="8.5" r="1.5" />
          <circle cx="17" cy="8.5" r="1.5" />
          <circle cx="7" cy="15.5" r="1.5" />
          <circle cx="17" cy="15.5" r="1.5" />
        </>
      )}
      {name === "sparkle" && (
        <>
          <path d="M12 3.5c.45 4.65 2.85 7.05 7.5 7.5-4.65.45-7.05 2.85-7.5 7.5-.45-4.65-2.85-7.05-7.5-7.5 4.65-.45 7.05-2.85 7.5-7.5Z" />
          <path d="M18.5 3.5c.13 1.37.83 2.07 2.2 2.2-1.37.13-2.07.83-2.2 2.2-.13-1.37-.83-2.07-2.2-2.2 1.37-.13 2.07-.83 2.2-2.2Z" />
        </>
      )}
      {name === "citation" && (
        <>
          <path d="M7 3.75h7l3 3v13.5H7z" />
          <path d="M14 3.75v3h3M9.5 13l1.6 1.6 3.6-3.6" />
        </>
      )}
      {name === "database" && (
        <>
          <ellipse cx="12" cy="6" rx="6.5" ry="2.5" />
          <path d="M5.5 6v6c0 1.4 2.9 2.5 6.5 2.5s6.5-1.1 6.5-2.5V6" />
          <path d="M5.5 12v6c0 1.4 2.9 2.5 6.5 2.5s6.5-1.1 6.5-2.5v-6" />
        </>
      )}
    </svg>
  );
}

function routeLabel(response: AnswerResponse): string {
  if (response.status === "abstained") return "Safe abstention";
  return response.route === "rag" ? "Grounded RAG" : "Approved analytics";
}

function resultBoundaryLabel(response: AnswerResponse): string {
  if (response.status === "abstained") return "Zero fabricated sources";
  return response.route === "rag" ? "Validated by backend" : "Approved tool result";
}

function errorTitle(kind: CivicLensClientError["kind"]): string {
  const labels: Record<CivicLensClientError["kind"], string> = {
    configuration: "Configuration needed",
    validation: "Question not accepted",
    backend_unavailable: "Backend unavailable",
    server: "Request could not complete",
    network: "Connection issue",
    timeout: "Request timed out",
    malformed_response: "Unexpected API response",
    api: "API request rejected",
  };
  return labels[kind];
}

export function CivicLensExperience({ askQuestion = askCivicLens }: CivicLensExperienceProps) {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState<AnswerResponse | null>(null);
  const [error, setError] = useState<CivicLensClientError | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [examplesExpanded, setExamplesExpanded] = useState(false);
  const requestPending = useRef(false);
  const questionInput = useRef<HTMLTextAreaElement>(null);

  const submitQuestion = async () => {
    const normalizedQuestion = question.trim();
    if (!normalizedQuestion || requestPending.current) return;

    requestPending.current = true;
    setIsLoading(true);
    setError(null);
    setResponse(null);
    try {
      setResponse(await askQuestion(normalizedQuestion));
    } catch (caught) {
      setError(
        caught instanceof CivicLensClientError
          ? caught
          : new CivicLensClientError(
              "network",
              "CivicLens could not complete the request. Please try again.",
              "unknown_client_error",
            ),
      );
    } finally {
      requestPending.current = false;
      setIsLoading(false);
    }
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void submitQuestion();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submitQuestion();
    }
  };

  const selectExample = (example: string) => {
    setQuestion(example);
    questionInput.current?.focus();
  };

  return (
    <main id="top">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="CivicLens home">
          <span className="brand-mark" aria-hidden="true">CL</span>
          <span>
            <strong>CivicLens</strong>
            <small>NYC 311 Operations Copilot</small>
          </span>
        </a>
        <div className="header-actions">
          <nav className="primary-nav" aria-label="Primary navigation">
            <a href="#main-workspace">Demo</a>
            <a href="#system-architecture">Architecture</a>
            <a href="https://github.com/rihua-tech/civiclens-rag-nyc311">GitHub</a>
          </nav>
          <span className="portfolio-badge">Portfolio demo · non-production</span>
        </div>
      </header>

      <section className="hero" aria-labelledby="page-title">
        <h1 id="page-title">Ask the system. <em>Inspect the evidence.</em></h1>
        <p>
          Hybrid retrieval, grounded generation, validated citations, and safe abstention
          over NYC 311 knowledge.
        </p>
      </section>

      <section className="workspace" id="main-workspace" aria-label="CivicLens AI assistant">
        <div className="question-panel">
          <div className="composer-heading">
            <span className="composer-mark"><ProductIcon name="sparkle" /></span>
            <h2>What would you like to know about NYC 311?</h2>
          </div>

          <form onSubmit={handleSubmit} className="question-form">
            <label className="visually-hidden" htmlFor="civiclens-question">Question</label>
            <div className="input-shell">
              <textarea
                aria-describedby="question-shortcuts question-count"
                id="civiclens-question"
                maxLength={MAX_QUESTION_LENGTH}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about complaint types, fields, runbooks, analytics, or documentation…"
                ref={questionInput}
                rows={3}
                value={question}
              />
            </div>

            <div className="composer-action-row">
              <button
                aria-busy={isLoading}
                className="submit-button"
                disabled={!question.trim() || isLoading}
                type="submit"
              >
                <span>{isLoading ? "Checking evidence…" : "Ask CivicLens"}</span>
                {isLoading ? (
                  <span className="submit-spinner" aria-hidden="true" />
                ) : (
                  <span className="button-arrow" aria-hidden="true">→</span>
                )}
              </button>
              <span className="keyboard-hint" id="question-shortcuts">
                <kbd>Enter</kbd> to send <span aria-hidden="true">·</span> Shift + Enter for new line
              </span>
              <span className="character-count" id="question-count">
                {question.length.toLocaleString()} / {MAX_QUESTION_LENGTH.toLocaleString()}
              </span>
            </div>
          </form>

          <div className="prompt-suggestions" aria-label="Example questions">
            <span className="prompt-label">Try a prompt</span>
            <div className="prompt-list">
              {EXAMPLE_QUESTIONS.slice(0, PRIMARY_EXAMPLE_COUNT).map((example) => (
                <button
                  className="prompt-button"
                  disabled={isLoading}
                  key={example}
                  onClick={() => selectExample(example)}
                  type="button"
                >
                  {example}
                </button>
              ))}
              <button
                aria-controls="additional-examples"
                aria-expanded={examplesExpanded}
                className="prompt-toggle"
                disabled={isLoading}
                onClick={() => setExamplesExpanded((current) => !current)}
                type="button"
              >
                {examplesExpanded ? "Fewer examples" : "More examples"}
                <span aria-hidden="true">⌄</span>
              </button>
            </div>
            <div
              className="additional-examples"
              hidden={!examplesExpanded}
              id="additional-examples"
            >
              {EXAMPLE_QUESTIONS.slice(PRIMARY_EXAMPLE_COUNT).map((example) => (
                <button
                  className="prompt-button"
                  disabled={isLoading}
                  key={example}
                  onClick={() => selectExample(example)}
                  type="button"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>

        </div>

        <div
          className={`answer-panel${error ? " answer-panel-error" : ""}${
            !isLoading && !error && !response ? " answer-panel-empty" : ""
          }`}
          aria-busy={isLoading}
          aria-live="polite"
        >
          {isLoading && (
            <div className="state-card loading-state" role="status">
              <span className="loading-pulse" aria-hidden="true" />
              <div>
                <span className="state-label">Request in progress</span>
                <h2>Reviewing the available evidence…</h2>
                <p>Free demo infrastructure can take extra time to wake from a cold start.</p>
              </div>
            </div>
          )}

          {!isLoading && error && (
            <div className="state-card error-state" role="alert">
              <span className="state-index" aria-hidden="true">!</span>
              <div>
                <span className="state-label">{errorTitle(error.kind)}</span>
                <h2>The answer panel is unavailable.</h2>
                <p>{error.userMessage}</p>
                <p className="error-code">Reference: {error.code}</p>
              </div>
            </div>
          )}

          {!isLoading && response && (
            <article className={`result-card result-${response.status} result-${response.route}`}>
              <header className="result-header">
                <h2>{routeLabel(response)}</h2>
                <span className="result-validation">
                  {response.status === "answered" && response.route === "rag" && (
                    <span aria-hidden="true">✓</span>
                  )}
                  <span>{resultBoundaryLabel(response)}</span>
                </span>
              </header>

              <div className="answer-copy">
                <span className="content-label">Answer</span>
                <p>{response.answer}</p>
              </div>

              {response.confidence_note && (
                <aside className="confidence-note">
                  <span className="confidence-marker"><ProductIcon name="shield" /></span>
                  <span>
                    <strong>Evidence note</strong>
                    <p>{response.confidence_note}</p>
                  </span>
                </aside>
              )}

              <section className="sources-section" aria-labelledby="sources-heading">
                <div className="sources-heading-row">
                  <h3 id="sources-heading">Sources &amp; provenance</h3>
                  <span>{response.sources.length} returned</span>
                </div>

                {response.sources.length === 0 ? (
                  <div className="zero-sources">
                    <strong>No sources returned</strong>
                    <p>
                      This is expected for a safe abstention; CivicLens does not fabricate
                      evidence when it cannot support an answer.
                    </p>
                  </div>
                ) : (
                  <div className="source-list">
                    {response.sources.map((source) => (
                      <article
                        className="source-card"
                        key={`${source.chunk_id}-${source.citation_number ?? "source"}`}
                      >
                        <div className="source-card-topline">
                          <span>
                            {source.citation_number
                              ? `Citation ${source.citation_number}`
                              : "Provenance"}
                          </span>
                          <span>Validated by backend</span>
                        </div>
                        <h4>{source.source_name}</h4>
                        {source.section_title && (
                          <p className="section-title">{source.section_title}</p>
                        )}
                        <code>{source.source_path}</code>
                        <details>
                          <summary>Technical provenance</summary>
                          <dl>
                            <div><dt>Chunk ID</dt><dd>{source.chunk_id}</dd></div>
                            {source.citation_number && (
                              <div><dt>Citation number</dt><dd>{source.citation_number}</dd></div>
                            )}
                          </dl>
                        </details>
                      </article>
                    ))}
                  </div>
                )}
              </section>

              {response.query_id && (
                <details className="query-details">
                  <summary>Request metadata</summary>
                  <dl><div><dt>Query ID</dt><dd>{response.query_id}</dd></div></dl>
                </details>
              )}
            </article>
          )}

          {!isLoading && !error && !response && (
            <div className="state-card empty-state">
              <span className="evidence-marker"><ProductIcon name="evidence" /></span>
              <div>
                <span className="state-label">Evidence panel</span>
                <h2>Your answer will arrive with its evidence attached.</h2>
                <p>
                  Ask a documentation or approved sample analytics question. CivicLens
                  returns its public answer fields without exposing provider diagnostics or
                  internal tool rows.
                </p>
              </div>
              <div className="capability-strip" aria-label="CivicLens answer behaviors">
                {CAPABILITIES.map(({ description, icon, label }) => (
                  <div className="capability-item" key={label}>
                    <span className="capability-marker"><ProductIcon name={icon} /></span>
                    <span><strong>{label}</strong><small>{description}</small></span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </section>

      <section
        className="technical-proof"
        id="system-architecture"
        aria-label="CivicLens technical proof"
      >
        {TECHNICAL_PROOF.map(({ description, icon, label }) => (
          <div key={label}>
            <span className="proof-marker"><ProductIcon name={icon} /></span>
            <span className="proof-copy">
              <strong>{label}</strong>
              <span>{description}</span>
            </span>
          </div>
        ))}
      </section>

      <footer className="site-footer">
        <div>
          <a className="footer-brand" href="#top">CivicLens</a>
          <p>
            Curated documentation and checked-in sample analytics only. Not connected to
            live NYC 311 operations.
          </p>
        </div>
        <nav className="footer-nav" aria-label="Footer navigation">
          <a href="https://github.com/rihua-tech/civiclens-rag-nyc311">GitHub</a>
          <a href="#system-architecture">Architecture</a>
          <span>Non-production portfolio demo</span>
        </nav>
      </footer>
    </main>
  );
}
