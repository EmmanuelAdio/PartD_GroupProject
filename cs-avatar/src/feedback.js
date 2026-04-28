/**
 * @file feedback.js
 * Feedback UI component for the Loughborough University Virtual Assistant.
 *
 * Exports `createFeedbackRow`, which attaches "Was your question answered?" Yes/No
 * buttons below each bot answer. Clicking Yes logs a resolved=true event; clicking
 * No sends the original query and answer to POST /feedback and displays the retried
 * answer returned by the backend.
 */

/** Fallback message shown when the feedback retry request itself fails. */
export const FRIENDLY_FEEDBACK_ERROR =
  "I couldn't improve that answer just now. Please try rephrasing your question.";

/**
 * Create a small <span> element containing a feedback status message.
 * @param {string} text - Message to display.
 * @returns {HTMLSpanElement}
 */
function createFeedbackMessage(text) {
  const message = document.createElement("span");
  message.className = "feedback-thanks";
  message.textContent = text;
  return message;
}

/**
 * POST a feedback payload to the backend and return the parsed response JSON.
 *
 * @param {object} opts
 * @param {string}   opts.feedbackEndpoint - Full URL for POST /feedback.
 * @param {object}   opts.payload          - Request body: { user_query, last_answer, resolved, reason? }.
 * @param {Function} opts.fetchImpl        - Fetch implementation (defaults to global fetch in callers).
 * @returns {Promise<object>} Parsed response body: { action, answer_payload? }.
 * @throws {Error} If the response status is not 2xx.
 */
async function postFeedback({
  feedbackEndpoint,
  payload,
  fetchImpl,
}) {
  const response = await fetchImpl(feedbackEndpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  let responsePayload = null;
  try {
    responsePayload = await response.json();
  } catch {
    responsePayload = null;
  }

  if (!response.ok) {
    const detail =
      responsePayload &&
      typeof responsePayload === "object" &&
      responsePayload.detail
        ? String(responsePayload.detail)
        : `Request failed with status ${response.status}`;
    throw new Error(detail);
  }

  return responsePayload;
}

/**
 * Build a feedback row DOM element and attach Yes/No click handlers.
 *
 * The row shows "Was your question answered?" with two buttons:
 * - Yes  → fires onAcknowledge() and sends resolved=true to the backend (best-effort, no retry).
 * - No   → calls POST /feedback with resolved=false, shows the retried answer via onRetryAnswer(),
 *           or shows an error message via onRetryError() if the retry fails.
 *
 * @param {object}   opts
 * @param {string}   opts.userQuery        - The original user question.
 * @param {string}   opts.answerText       - The bot answer the user is rating.
 * @param {string}   opts.feedbackEndpoint - Full URL for POST /feedback.
 * @param {Function} [opts.fetchImpl]      - Fetch implementation; defaults to global fetch.
 * @param {Function} [opts.onRetryAnswer]  - Called with the retried answer string on success.
 * @param {Function} [opts.onRetryError]   - Called with an error message string if retry fails.
 * @param {Function} [opts.onAcknowledge]  - Called when the user clicks Yes.
 * @returns {HTMLDivElement} The fully wired feedback row element.
 */
export function createFeedbackRow({
  userQuery,
  answerText,
  feedbackEndpoint,
  fetchImpl = fetch,
  onRetryAnswer = () => {},
  onRetryError = () => {},
  onAcknowledge = () => {},
}) {
  const feedbackRow = document.createElement("div");
  feedbackRow.className = "feedback-row";

  const label = document.createElement("span");
  label.className = "feedback-label";
  label.textContent = "Was your question answered?";

  const yesBtn = document.createElement("button");
  yesBtn.className = "feedback-btn feedback-yes";
  yesBtn.textContent = "Yes";

  const noBtn = document.createElement("button");
  noBtn.className = "feedback-btn feedback-no";
  noBtn.textContent = "No";

  function setRowMessage(text) {
    feedbackRow.replaceChildren(createFeedbackMessage(text));
  }

  async function sendAcknowledgement() {
    try {
      await postFeedback({
        feedbackEndpoint,
        fetchImpl,
        payload: {
          user_query: userQuery,
          last_answer: answerText,
          resolved: true,
        },
      });
    } catch (error) {
      console.warn("Feedback acknowledgement failed:", error);
    }
  }

  async function retryAnswer() {
    try {
      const payload = await postFeedback({
        feedbackEndpoint,
        fetchImpl,
        payload: {
          user_query: userQuery,
          last_answer: answerText,
          resolved: false,
        },
      });
      const answer =
        payload &&
        payload.action === "retried" &&
        payload.answer_payload &&
        typeof payload.answer_payload.answer === "string"
          ? payload.answer_payload.answer.trim()
          : "";
      if (!answer) {
        throw new Error("Retry response did not include an answer.");
      }
      setRowMessage("Thanks for the feedback.");
      // Pass the full answer_payload as a second argument so callers (e.g. the
      // debug panel) can read grounded, confidence, citations without re-fetching.
      onRetryAnswer(answer, payload.answer_payload);
    } catch (error) {
      console.error("Feedback retry failed:", error);
      setRowMessage("Thanks for the feedback.");
      onRetryError(FRIENDLY_FEEDBACK_ERROR);
    }
  }

  function handleFeedback(answered) {
    yesBtn.disabled = true;
    noBtn.disabled = true;

    if (answered) {
      setRowMessage("Glad we could help!");
      onAcknowledge();
      void sendAcknowledgement();
      return;
    }

    setRowMessage("Trying again...");
    void retryAnswer();
  }

  yesBtn.addEventListener("click", () => handleFeedback(true));
  noBtn.addEventListener("click", () => handleFeedback(false));

  feedbackRow.appendChild(label);
  feedbackRow.appendChild(yesBtn);
  feedbackRow.appendChild(noBtn);
  return feedbackRow;
}
