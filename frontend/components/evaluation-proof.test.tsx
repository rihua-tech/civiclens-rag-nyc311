import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import {
  loadEvaluationSummary,
  parseEvaluationSummary,
} from "@/lib/evaluation-summary";

import { EvaluationProof } from "./evaluation-proof";

describe("EvaluationProof", () => {
  it("renders the approved local metrics, metadata, configuration, and report link", () => {
    const summary = loadEvaluationSummary();
    expect(summary).not.toBeNull();

    render(<EvaluationProof summary={summary} />);

    const expectedMetrics = [
      ["Hybrid Recall@5", "83.93%", "n=14"],
      ["Abstention Recall", "100.00%", "n=6"],
      ["False Abstention Rate", "5.56%", "n=18"],
      ["Citation ID Validity", "92.86%", "n=14"],
    ] as const;

    for (const [label, value, sampleCount] of expectedMetrics) {
      const card = screen.getByRole("article", {
        name: `${label}: ${value}, ${sampleCount}`,
      });
      expect(within(card).getByText(value)).toBeVisible();
      expect(within(card).getByText(sampleCount)).toBeVisible();
    }

    expect(screen.getByText("November 13, 2026")).toBeVisible();
    expect(screen.getByText("issue10-v1 · 24 questions")).toBeVisible();
    expect(screen.getByText("p0-report-v2")).toBeVisible();
    expect(screen.getByText(/Hybrid retrieval.*all-MiniLM-L6-v2/)).toBeVisible();
    expect(screen.getByText(/OpenAI gpt-4o-mini.*16 calls.*0 fallbacks/)).toBeVisible();
    expect(screen.getByText(/Local benchmark only/)).toHaveTextContent(
      "do not represent hosted Vercel or Render performance or latency",
    );
    expect(screen.getByText(/Citation ID validity verifies/)).toHaveTextContent(
      "does not measure factual answer accuracy",
    );
    expect(screen.getByRole("link", { name: /Read the full evaluation report/ })).toHaveAttribute(
      "href",
      "https://github.com/rihua-tech/civiclens-rag-nyc311/blob/main/docs/evaluation-p0-report.md",
    );
  });

  it("shows no scores when the summary file is missing", () => {
    render(<EvaluationProof summary={loadEvaluationSummary("missing-eval-summary.json")} />);

    expect(screen.getByRole("heading", { name: "Benchmark summary unavailable" })).toBeVisible();
    expect(screen.queryByLabelText("Local evaluation metrics")).not.toBeInTheDocument();
  });

  it("rejects invalid numeric values and shows the unavailable state", () => {
    const validSummary = loadEvaluationSummary();
    expect(validSummary).not.toBeNull();
    if (!validSummary) return;

    const invalidSummary = {
      ...validSummary,
      metrics: validSummary.metrics.map((metric, index) =>
        index === 0 ? { ...metric, value: 1.01 } : metric,
      ),
    };

    render(<EvaluationProof summary={parseEvaluationSummary(invalidSummary)} />);

    expect(screen.getByRole("heading", { name: "Benchmark summary unavailable" })).toBeVisible();
    expect(screen.queryByText("101.00%")).not.toBeInTheDocument();
  });
});
