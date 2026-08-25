-- Issue 17: allow-listed operational metadata for bounded orchestration.
-- Raw questions, answers, prompts, reasoning, and graph state remain excluded.

ALTER TABLE queries ADD COLUMN IF NOT EXISTS orchestration_mode TEXT;
ALTER TABLE queries ADD COLUMN IF NOT EXISTS orchestration_step_count INTEGER;
ALTER TABLE queries ADD COLUMN IF NOT EXISTS orchestration_tool_call_count INTEGER;
ALTER TABLE queries ADD COLUMN IF NOT EXISTS orchestration_outcome TEXT;
