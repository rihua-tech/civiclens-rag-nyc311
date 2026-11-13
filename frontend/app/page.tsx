import { CivicLensExperience } from "@/components/civic-lens-experience";
import { EvaluationProof } from "@/components/evaluation-proof";
import { loadEvaluationSummary } from "@/lib/evaluation-summary";

export default function Home() {
  const evaluationSummary = loadEvaluationSummary();

  return (
    <CivicLensExperience
      evaluationProof={<EvaluationProof summary={evaluationSummary} />}
    />
  );
}
