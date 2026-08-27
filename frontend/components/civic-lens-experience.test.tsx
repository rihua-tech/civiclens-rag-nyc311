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

    await user.click(
      screen.getByRole("button", { name: "What does complaint_type mean?" }),
    );
    await user.click(screen.getByRole("button", { name: "Ask CivicLens" }));

    expect(askQuestion).toHaveBeenCalledOnce();
    expect(askQuestion).toHaveBeenCalledWith("What does complaint_type mean?");
    expect(await screen.findByRole("heading", { name: "Grounded RAG" })).toBeVisible();
    expect(screen.getByText(RAG_RESPONSE.answer)).toBeVisible();
    expect(screen.getByText("Citation 4")).toBeVisible();
    expect(screen.getByText("Problem / Complaint Type")).toBeVisible();
    expect(screen.getByText(RAG_RESPONSE.sources[0].source_path)).toBeVisible();
    expect(screen.getByText("Request metadata")).toBeVisible();
    expect(screen.getByText(RAG_RESPONSE.query_id)).toBeInTheDocument();
    expect(screen.queryByText("Citation 1")).not.toBeInTheDocument();
    expect(screen.getAllByText("Validated by backend")).toHaveLength(2);
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

    expect(askQuestion).toHaveBeenCalledOnce();
    expect(askQuestion).toHaveBeenCalledWith("What are the top complaint types?");
    expect(await screen.findByRole("heading", { name: "Approved analytics" })).toBeVisible();
    expect(screen.getByText("Approved tool result")).toBeVisible();
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
    expect(screen.getByText("Zero fabricated sources")).toBeVisible();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("keeps additional prompt guidance collapsed until requested", async () => {
    const user = userEvent.setup();
    render(<CivicLensExperience />);

    const toggle = screen.getByRole("button", { name: "More examples" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(
      screen.queryByRole("button", {
        name: "What is the local retrieval and cited answer flow?",
      }),
    ).not.toBeInTheDocument();

    await user.click(toggle);

    expect(screen.getByRole("button", { name: "Fewer examples" })).toHaveAttribute(
      "aria-expanded",
      "true",
    );
    await user.click(
      screen.getByRole("button", {
        name: "What is the local retrieval and cited answer flow?",
      }),
    );
    expect(screen.getByRole("textbox", { name: "Question" })).toHaveValue(
      "What is the local retrieval and cited answer flow?",
    );
    expect(screen.getByRole("textbox", { name: "Question" })).toHaveFocus();
  });

  it("shows capability guidance only inside the empty Evidence Panel", async () => {
    const user = userEvent.setup();
    const askQuestion = vi.fn(async () => RAG_RESPONSE);
    render(<CivicLensExperience askQuestion={askQuestion} />);

    const capabilityStrip = screen.getByLabelText("CivicLens answer behaviors");
    expect(capabilityStrip.closest(".answer-panel")).toBeInTheDocument();
    expect(capabilityStrip.closest(".question-panel")).toBeNull();

    await enterQuestion(user, "What does complaint_type mean?");
    await user.click(screen.getByRole("button", { name: "Ask CivicLens" }));

    expect(await screen.findByRole("heading", { name: "Grounded RAG" })).toBeVisible();
    expect(screen.queryByLabelText("CivicLens answer behaviors")).not.toBeInTheDocument();
  });

  it("enters the loading state immediately after one submit-button click", async () => {
    const user = userEvent.setup();
    let resolveRequest: ((value: typeof RAG_RESPONSE) => void) | undefined;
    const askQuestion = vi.fn(
      () =>
        new Promise<typeof RAG_RESPONSE>((resolve) => {
          resolveRequest = resolve;
        }),
    );
    render(<CivicLensExperience askQuestion={askQuestion} />);

    await enterQuestion(user, "What does complaint_type mean?");
    await user.click(screen.getByRole("button", { name: "Ask CivicLens" }));

    expect(askQuestion).toHaveBeenCalledOnce();
    expect(askQuestion).toHaveBeenCalledWith("What does complaint_type mean?");
    expect(screen.getByRole("button", { name: "Checking evidence…" })).toHaveAttribute(
      "aria-busy",
      "true",
    );
    expect(screen.getByRole("button", { name: "Checking evidence…" })).toBeDisabled();
    expect(screen.getByRole("status")).toHaveTextContent(
      "Reviewing the available evidence…",
    );

    resolveRequest?.(RAG_RESPONSE);
    expect(await screen.findByRole("heading", { name: "Grounded RAG" })).toBeVisible();
  });

  it("prevents repeated submit-button clicks while a request is pending", async () => {
    const user = userEvent.setup();
    let resolveRequest: ((value: typeof RAG_RESPONSE) => void) | undefined;
    const askQuestion = vi.fn(
      () =>
        new Promise<typeof RAG_RESPONSE>((resolve) => {
          resolveRequest = resolve;
        }),
    );
    render(<CivicLensExperience askQuestion={askQuestion} />);

    await enterQuestion(user, "What does complaint_type mean?");
    const submitButton = screen.getByRole("button", { name: "Ask CivicLens" });
    fireEvent.click(submitButton);
    fireEvent.click(submitButton);

    expect(askQuestion).toHaveBeenCalledOnce();
    expect(screen.getByRole("button", { name: "Checking evidence…" })).toBeDisabled();

    resolveRequest?.(RAG_RESPONSE);
    expect(await screen.findByRole("heading", { name: "Grounded RAG" })).toBeVisible();
  });

  it("submits once with Enter and keeps Shift + Enter as a newline gesture", async () => {
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
    fireEvent.keyDown(input, { key: "Enter", shiftKey: true });
    expect(askQuestion).not.toHaveBeenCalled();

    fireEvent.keyDown(input, { key: "Enter" });
    fireEvent.keyDown(input, { key: "Enter" });

    expect(askQuestion).toHaveBeenCalledOnce();
    expect(askQuestion).toHaveBeenCalledWith("What does complaint_type mean?");

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
