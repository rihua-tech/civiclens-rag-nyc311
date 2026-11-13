import type { EvaluationSummary } from "@/lib/evaluation-summary";

interface EvaluationProofProps {
  summary: EvaluationSummary | null;
}

const percentFormatter = new Intl.NumberFormat("en-US", {
  style: "percent",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

function formatEvaluationDate(value: string): string {
  return new Intl.DateTimeFormat("en-US", {
    day: "numeric",
    month: "long",
    timeZone: "UTC",
    year: "numeric",
  }).format(new Date(`${value}T00:00:00Z`));
}

export function EvaluationProof({ summary }: EvaluationProofProps) {
  if (!summary) {
    return (
      <section className="evaluation-proof evaluation-unavailable" aria-labelledby="evaluation-title">
        <span className="evaluation-eyebrow">Evaluation proof</span>
        <h2 id="evaluation-title">Benchmark summary unavailable</h2>
        <p>
          The version-controlled evaluation summary is missing or invalid, so no metrics
          are shown.
        </p>
      </section>
    );
  }

  const { configuration } = summary;

  return (
    <section className="evaluation-proof" aria-labelledby="evaluation-title">
      <header className="evaluation-heading">
        <div>
          <span className="evaluation-eyebrow">Evaluation proof</span>
          <h2 id="evaluation-title">Measured quality, with its boundaries attached.</h2>
          <p className="evaluation-summary">
            Evaluated locally using Hybrid Retrieval + OpenAI. Not a measurement of the
            hosted demo.
          </p>
        </div>
        <span className="evaluation-scope">Local benchmark</span>
      </header>

      <dl className="evaluation-metadata">
        <div>
          <dt>Evaluation date</dt>
          <dd><time dateTime={summary.evaluation_date}>{formatEvaluationDate(summary.evaluation_date)}</time></dd>
        </div>
        <div>
          <dt>Dataset</dt>
          <dd>{summary.dataset_version} · {summary.question_count} questions</dd>
        </div>
        <div>
          <dt>Metric definition</dt>
          <dd>{summary.metric_definition_version}</dd>
        </div>
      </dl>

      <div className="evaluation-metrics" aria-label="Local evaluation metrics">
        {summary.metrics.map((metric) => (
          <article
            aria-label={`${metric.label}: ${percentFormatter.format(metric.value)}, n=${metric.sample_count}`}
            className="metric-card"
            key={metric.id}
          >
            <span className="metric-label">{metric.label}</span>
            <div className="metric-value-row">
              <strong>{percentFormatter.format(metric.value)}</strong>
              <span className="metric-sample">n={metric.sample_count}</span>
            </div>
            <p>{metric.description}</p>
          </article>
        ))}
      </div>

      <div className="evaluation-details">
        <dl className="evaluation-configuration">
          <div>
            <dt>Retrieval</dt>
            <dd>
              {configuration.retrieval_profile} · {configuration.embedding_model} ·{" "}
              {configuration.embedding_dimensions} dimensions · {configuration.vector_store} ·
              k={configuration.retrieval_k} · {configuration.reranking_enabled ? "reranked" : "no reranking"}
            </dd>
          </div>
          <div>
            <dt>Answers</dt>
            <dd>
              {configuration.answer_provider} {configuration.answer_model} · structured answers ·{" "}
              {configuration.provider_calls} calls · {configuration.provider_fallbacks} fallbacks
            </dd>
          </div>
        </dl>

        <div className="evaluation-notes">
          <p className="evaluation-boundary">{summary.scope_note}</p>
          <p>{summary.citation_validity_note}</p>
          <p className="evaluation-publication">{summary.publication_note}</p>
        </div>
      </div>

      <a className="evaluation-report-link" href={summary.source_report.url}>
        {summary.source_report.label}
        <span aria-hidden="true">↗</span>
      </a>
    </section>
  );
}
