import { readFileSync } from "node:fs";
import { join } from "node:path";

import { z } from "zod";

const METRIC_IDS = [
  "hybrid_recall_at_5",
  "abstention_recall",
  "false_abstention_rate",
  "citation_id_validity",
] as const;

const metricSchema = z
  .object({
    id: z.enum(METRIC_IDS),
    label: z.string().min(1),
    value: z.number().min(0).max(1),
    sample_count: z.number().int().positive(),
    description: z.string().min(1),
  })
  .strict();

export const evaluationSummarySchema = z
  .object({
    schema_version: z.literal(1),
    evaluation_date: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
    dataset_version: z.string().min(1),
    metric_definition_version: z.string().min(1),
    question_count: z.number().int().positive(),
    benchmark_scope: z.literal("local"),
    configuration: z
      .object({
        retrieval_profile: z.string().min(1),
        embedding_model: z.string().min(1),
        embedding_dimensions: z.number().int().positive(),
        vector_store: z.string().min(1),
        retrieval_k: z.number().int().positive(),
        reranking_enabled: z.boolean(),
        answer_provider: z.string().min(1),
        answer_model: z.string().min(1),
        provider_calls: z.number().int().nonnegative(),
        provider_fallbacks: z.number().int().nonnegative(),
      })
      .strict(),
    metrics: z.array(metricSchema).length(METRIC_IDS.length),
    scope_note: z.string().min(1),
    citation_validity_note: z.string().min(1),
    publication_note: z.string().min(1),
    source_report: z
      .object({
        label: z.string().min(1),
        url: z.string().url(),
      })
      .strict(),
  })
  .strict()
  .superRefine((summary, context) => {
    const observedIds = new Set(summary.metrics.map((metric) => metric.id));

    for (const expectedId of METRIC_IDS) {
      if (!observedIds.has(expectedId)) {
        context.addIssue({
          code: "custom",
          message: `Missing required metric: ${expectedId}`,
          path: ["metrics"],
        });
      }
    }

    for (const [index, metric] of summary.metrics.entries()) {
      if (metric.sample_count > summary.question_count) {
        context.addIssue({
          code: "custom",
          message: "Metric sample count cannot exceed the dataset question count.",
          path: ["metrics", index, "sample_count"],
        });
      }
    }
  });

export type EvaluationSummary = z.infer<typeof evaluationSummarySchema>;

export function parseEvaluationSummary(value: unknown): EvaluationSummary | null {
  const result = evaluationSummarySchema.safeParse(value);
  return result.success ? result.data : null;
}

export function loadEvaluationSummary(
  summaryPath = join(process.cwd(), "data", "eval-summary.json"),
): EvaluationSummary | null {
  try {
    return parseEvaluationSummary(
      JSON.parse(readFileSync(summaryPath, "utf8")) as unknown,
    );
  } catch {
    return null;
  }
}
