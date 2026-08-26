import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { CivicLensClientError } from "@/lib/api-client";

import { CivicLensExperience } from "./civic-lens-experience";

const RAG_RESPONSE = {
  answer: "Complaint Type describes the reported problem [4].",
  route: "rag" as const,
  status: "answered" as const,
  sources: [
    {
      source_name: "NYC 311 Service Request Field Guide",
      source_path: "docs/knowledge/nyc311-service-request-fields.md",
      chunk_id: "chunk_problem_type",
      section_title: "Problem / Complaint Type",
      citation_number: 4,
    },
  ],
  confidence_note: "This answer uses retrieved local context only.",
  query_id: "77b5a698-16da-4a3a-a492-e03c26b02cc7",
};

async function enterQuestion(user: ReturnType<typeof userEvent.setup>, value: string) {
  const input = screen.getByRole("textbox", { name: "Question" });
  await user.clear(input);
  await user.type(input, value);
  return input;
}

describe("CivicLensExperience", () => {
  it("renders an answered RAG response with backend citation numbering and provenance", async () => {
    const user = userEvent.setup();
    const askQuestion = vi.fn(async () => RAG_RESPONSE);
    render(<CivicLensExperience askQuestion={askQuestion} />);

    await user.click(screen.getByRole("button", { name: /Field definition/i }));
    await user.click(screen.getByRole("button", { name: "Ask CivicLens" }));

    expect(await screen.findByRole("heading", { name: "Grounded RAG" })).toBeVisible();
    expect(screen.getByText(RAG_RESPONSE.answer)).toBeVisible();
    expect(screen.getByText("Citation 4")).toBeVisible();
    expect(screen.getByText("Problem / Complaint Type")).toBeVisible();
    expect(screen.getByText(RAG_RESPONSE.sources[0].source_path)).toBeVisible();
    expect(screen.getByText("Request metadata")).toBeVisible();
    expect(screen.getByText(RAG_RESPONSE.query_id)).toBeInTheDocument();
    expect(screen.queryByText("Citation 1")).not.toBeInTheDocument();
  });

  it("renders an answered analytics response only through public answer fields", async () => {
    const user = userEvent.setup();
    const askQuestion = vi.fn(async () => ({
      answer: "Noise - Residential is the top checked-in sample category.",
      route: "analytics" as const,
      status: "answered" as const,
      sources: [
        {
          source_name: "top_complaint_types.csv",
          source_path: "data/sample_outputs/top_complaint_types.csv",
          chunk_id: "sample_output" as const,
        },
      ],
      confidence_note: "Checked-in sample analytics only.",
    }));
    render(<CivicLensExperience askQuestion={askQuestion} />);

    await enterQuestion(user, "What are the top complaint types?");
    await user.click(screen.getByRole("button", { name: "Ask CivicLens" }));

    expect(await screen.findByRole("heading", { name: "Approved analytics" })).toBeVisible();
    expect(screen.getByText("top_complaint_types.csv")).toBeVisible();
    expect(screen.queryByText("Request metadata")).not.toBeInTheDocument();
    expect(screen.queryByText("complaint_type")).not.toBeInTheDocument();
  });

  it("renders safe abstention as a valid zero-source result rather than an error", async () => {
    const user = userEvent.setup();
    const askQuestion = vi.fn(async () => ({
      answer: "I do not have enough source context to answer that.",
      route: "rag" as const,
      status: "abstained" as const,
      sources: [],
      confidence_note: "No usable source chunks were retrieved.",
    }));
    render(<CivicLensExperience askQuestion={askQuestion} />);

    await enterQuestion(user, "Explain the orbital pineapple parking treaty.");
    await user.click(screen.getByRole("button", { name: "Ask CivicLens" }));

    expect(await screen.findByRole("heading", { name: "Safe abstention" })).toBeVisible();
    expect(screen.getByText("No sources returned")).toBeVisible();
    expect(screen.getByText("0 returned")).toBeVisible();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("prevents duplicate submissions while a request is pending", async () => {
    const user = userEvent.setup();
    let resolveRequest: ((value: typeof RAG_RESPONSE) => void) | undefined;
    const askQuestion = vi.fn(
      () =>
        new Promise<typeof RAG_RESPONSE>((resolve) => {
          resolveRequest = resolve;
        }),
    );
    render(<CivicLensExperience askQuestion={askQuestion} />);

    const input = await enterQuestion(user, "What does complaint_type mean?");
    fireEvent.keyDown(input, { key: "Enter" });
    fireEvent.keyDown(input, { key: "Enter" });

    expect(askQuestion).toHaveBeenCalledOnce();
    expect(screen.getByRole("button", { name: "Checking evidence…" })).toBeDisabled();

    resolveRequest?.(RAG_RESPONSE);
    expect(await screen.findByRole("heading", { name: "Grounded RAG" })).toBeVisible();
  });

  it("renders a sanitized backend failure without citations or diagnostics", async () => {
    const user = userEvent.setup();
    const askQuestion = vi.fn(async () => {
      throw new CivicLensClientError(
        "backend_unavailable",
        "CivicLens is warming up or temporarily unavailable.",
        "backend_unavailable",
        503,
      );
    });
    render(<CivicLensExperience askQuestion={askQuestion} />);

    await enterQuestion(user, "What does complaint_type mean?");
    await user.click(screen.getByRole("button", { name: "Ask CivicLens" }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Backend unavailable");
    expect(alert).toHaveTextContent("backend_unavailable");
    expect(screen.queryByText("Sources & provenance")).not.toBeInTheDocument();
    expect(alert).not.toHaveTextContent("Traceback");
  });
});
