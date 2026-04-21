export const FRIENDLY_FEEDBACK_ERROR =
  "I couldn't improve that answer just now. Please try rephrasing your question.";

function createFeedbackMessage(text) {
  const message = document.createElement("span");
  message.className = "feedback-thanks";
  message.textContent = text;
  return message;
}

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
      onRetryAnswer(answer);
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
