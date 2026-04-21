import { describe, expect, it, vi } from "vitest";

import {
  FRIENDLY_FEEDBACK_ERROR,
  createFeedbackRow,
} from "./feedback.js";

function flushPromises() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

describe("createFeedbackRow", () => {
  it("acknowledges yes feedback without appending a retry answer", async () => {
    const onRetryAnswer = vi.fn();
    const onAcknowledge = vi.fn();
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      json: async () => ({ action: "acknowledged", answer_payload: null }),
    }));

    const row = createFeedbackRow({
      userQuery: "How much is Butler Court?",
      answerText: "Butler Court costs GBP 126.68 per week.",
      feedbackEndpoint: "http://127.0.0.1:8000/feedback",
      fetchImpl,
      onRetryAnswer,
      onAcknowledge,
    });
    document.body.appendChild(row);

    const yesBtn = row.querySelector(".feedback-yes");
    const noBtn = row.querySelector(".feedback-no");
    yesBtn.click();
    await flushPromises();

    expect(yesBtn.disabled).toBe(true);
    expect(noBtn.disabled).toBe(true);
    expect(row.textContent).toContain("Glad we could help!");
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(onAcknowledge).toHaveBeenCalledTimes(1);
    expect(onRetryAnswer).not.toHaveBeenCalled();
  });

  it("retries once on no feedback and appends the improved answer", async () => {
    const onRetryAnswer = vi.fn();
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        action: "retried",
        answer_payload: {
          answer: "Here is a more specific grounded answer.",
        },
      }),
    }));

    const row = createFeedbackRow({
      userQuery: "Tell me about accommodation fees",
      answerText: "Accommodation costs vary.",
      feedbackEndpoint: "http://127.0.0.1:8000/feedback",
      fetchImpl,
      onRetryAnswer,
    });
    document.body.appendChild(row);

    const noBtn = row.querySelector(".feedback-no");
    const yesBtn = row.querySelector(".feedback-yes");
    noBtn.click();

    expect(noBtn.disabled).toBe(true);
    expect(yesBtn.disabled).toBe(true);
    expect(row.textContent).toContain("Trying again...");

    await flushPromises();

    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(onRetryAnswer).toHaveBeenCalledWith(
      "Here is a more specific grounded answer."
    );
    expect(row.textContent).toContain("Thanks for the feedback.");
  });

  it("shows a friendly fallback when retry fails", async () => {
    const onRetryError = vi.fn();
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const fetchImpl = vi.fn(async () => {
      throw new Error("network down");
    });

    const row = createFeedbackRow({
      userQuery: "What are the entry requirements?",
      answerText: "It depends.",
      feedbackEndpoint: "http://127.0.0.1:8000/feedback",
      fetchImpl,
      onRetryError,
    });
    document.body.appendChild(row);

    row.querySelector(".feedback-no").click();
    await flushPromises();

    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(onRetryError).toHaveBeenCalledWith(FRIENDLY_FEEDBACK_ERROR);
    expect(row.textContent).toContain("Thanks for the feedback.");
    consoleErrorSpy.mockRestore();
  });
});
