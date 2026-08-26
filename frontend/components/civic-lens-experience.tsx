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
  {
    label: "Field definition",
    question: "What does complaint_type mean?",
  },
  {
    label: "Architecture",
    question: "What is the local retrieval and cited answer flow?",
  },
  {
    label: "Sample analytics",
    question: "What are the top complaint types?",
  },
  {
    label: "Safe abstention",
    question: "Explain the orbital pineapple parking treaty.",
  },
] as const;

const SYSTEM_PATH = [
  ["Interface", "Next.js client"],
  ["Contract", "FastAPI"],
  ["Intelligence", "Hybrid RAG"],
  ["Authority", "PostgreSQL + pgvector"],
] as const;

interface CivicLensExperienceProps {
  askQuestion?: AskQuestion;
}

function routeLabel(response: AnswerResponse): string {
  if (response.status === "abstained") return "Safe abstention";
  return response.route === "rag" ? "Grounded RAG" : "Approved analytics";
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
  const requestPending = useRef(false);

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
            <a href="#main-workspace">Try demo</a>
            <a href="#system-architecture">Architecture</a>
            <a href="https://github.com/rihua-tech/civiclens-rag-nyc311">GitHub</a>
          </nav>
          <span className="portfolio-badge">Portfolio demo · non-production</span>
        </div>
      </header>

      <section className="hero" aria-labelledby="page-title">
        <div className="hero-copy-block">
          <div className="eyebrow"><span /> Evidence-first civic intelligence</div>
          <h1 id="page-title">Ask the system.<br /><em>Inspect the evidence.</em></h1>
          <p className="hero-copy">
            CivicLens pairs hybrid retrieval with grounded generation, application-owned
            citation validation, and safe abstention over a curated NYC 311 knowledge base.
          </p>
          <div className="hero-actions">
            <a className="primary-cta" href="#main-workspace">
              Try CivicLens <span aria-hidden="true">↓</span>
            </a>
            <a className="secondary-link" href="#system-architecture">View architecture</a>
          </div>
        </div>

        <aside className="hero-proof" aria-labelledby="system-proof-heading">
          <span className="proof-kicker">System proof</span>
          <h2 id="system-proof-heading">Why the answer is defensible.</h2>
          <dl>
            <div>
              <dt>Hybrid retrieval</dt>
              <dd>Semantic + lexical evidence</dd>
            </div>
            <div>
              <dt>Grounded answers</dt>
              <dd>Answers stay constrained to retrieved evidence</dd>
            </div>
            <div>
              <dt>Validated citations</dt>
              <dd>Backend-owned provenance</dd>
            </div>
            <div>
              <dt>Safe abstention</dt>
              <dd>No answer when evidence is insufficient</dd>
            </div>
          </dl>
        </aside>
      </section>

      <section
        className="system-path"
        id="system-architecture"
        aria-label="CivicLens application path"
      >
        {SYSTEM_PATH.map(([label, value], index) => (
          <div className="system-node" key={label}>
            <span className="node-number">0{index + 1}</span>
            <span><small>{label}</small><strong>{value}</strong></span>
          </div>
        ))}
      </section>

      <section className="workspace" id="main-workspace" aria-label="CivicLens question workspace">
        <div className="question-panel">
          <div className="panel-kicker">Start with a scenario</div>
          <h2>Choose a question or write your own.</h2>
          <div className="example-grid" aria-label="Example questions">
            {EXAMPLE_QUESTIONS.map((example) => (
              <button
                className="example-button"
                disabled={isLoading}
                key={example.label}
                onClick={() => setQuestion(example.question)}
                type="button"
              >
                <span>{example.label}</span>
                {example.question}
              </button>
            ))}
          </div>

          <form onSubmit={handleSubmit} className="question-form">
            <label htmlFor="civiclens-question">Question</label>
            <div className="input-shell">
              <textarea
                id="civiclens-question"
                maxLength={MAX_QUESTION_LENGTH}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about NYC 311 fields, the CivicLens architecture, or approved sample analytics…"
                rows={4}
                value={question}
              />
              <div className="input-meta">
                <span>Enter to ask · Shift + Enter for a new line</span>
                <span>{question.length.toLocaleString()} / {MAX_QUESTION_LENGTH.toLocaleString()}</span>
              </div>
            </div>
            <button
              className="submit-button"
              disabled={!question.trim() || isLoading}
              type="submit"
            >
              <span>{isLoading ? "Checking evidence…" : "Ask CivicLens"}</span>
              <span className="button-arrow" aria-hidden="true">→</span>
            </button>
          </form>
        </div>

        <div className="answer-panel" aria-busy={isLoading} aria-live="polite">
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
            <article
              className={`result-card result-${response.status} result-${response.route}`}
            >
              <header className="result-header">
                <div>
                  <span className="state-label">Public answer contract</span>
                  <h2>{routeLabel(response)}</h2>
                </div>
                <div className="result-tags" aria-label="Answer route and status">
                  <span>{response.route === "rag" ? "RAG" : "Analytics"}</span>
                  <span>{response.status === "answered" ? "Answered" : "Abstained"}</span>
                </div>
              </header>

              <div className="answer-copy">
                <p>{response.answer}</p>
              </div>

              {response.confidence_note && (
                <aside className="confidence-note">
                  <strong>Evidence note</strong>
                  <p>{response.confidence_note}</p>
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
                      <article className="source-card" key={`${source.chunk_id}-${source.citation_number ?? "source"}`}>
                        <div className="source-card-topline">
                          <span>{source.citation_number ? `Citation ${source.citation_number}` : "Provenance"}</span>
                          <span>Validated by backend</span>
                        </div>
                        <h4>{source.source_name}</h4>
                        {source.section_title && <p className="section-title">{source.section_title}</p>}
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
              <span className="state-index" aria-hidden="true">01</span>
              <div>
                <span className="state-label">Evidence panel</span>
                <h2>Answers stay attached to their provenance.</h2>
                <p>
                  Ask a supported documentation or sample analytics question. CivicLens will
                  return only its public answer fields—never raw chunks, internal tool rows,
                  or provider diagnostics.
                </p>
                <ul className="state-guide">
                  <li><span /> Grounded RAG with validated citations</li>
                  <li><span /> Approved analytics with sample-data provenance</li>
                  <li><span /> Safe abstention when evidence is insufficient</li>
                </ul>
              </div>
            </div>
          )}
        </div>
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
