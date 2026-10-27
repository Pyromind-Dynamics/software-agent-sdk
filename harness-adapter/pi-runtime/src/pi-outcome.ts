import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { AssistantMessage } from "@earendil-works/pi-ai";
import type { AgentSessionEvent } from "@earendil-works/pi-coding-agent";
import type { RunOutcome } from "./protocol.js";
import { REPETITION_ERROR, REPETITION_MESSAGE } from "./no-spin-extension.js";

export class PiOutcomeNormalizer {
  private latestAssistant: AssistantMessage | undefined;

  reset(): void {
    this.latestAssistant = undefined;
  }

  observe(event: AgentSessionEvent): void {
    if (event.type === "message_end" && event.message.role === "assistant") {
      this.latestAssistant = event.message;
    }
  }

  normalize(messages: AgentMessage[]): RunOutcome {
    return normalizePiOutcome(messages, this.latestAssistant);
  }
}

export function normalizePiOutcome(
  messages: AgentMessage[],
  latest?: AssistantMessage,
): RunOutcome {
  const message = latest ?? [...messages]
    .reverse()
    .find((item): item is AssistantMessage => item.role === "assistant");
  if (!message) {
    return {
      status: "failed",
      error_code: "no_assistant_response",
      message: "Pi run ended without an assistant response",
    };
  }
  switch (message.stopReason) {
    case "stop":
    case "toolUse":
      if (!message.content.some((block) =>
        (block.type === "text" && block.text.trim().length > 0) || block.type === "toolCall")) {
        return { status: "failed", stop_reason: message.stopReason,
          error_code: "model_empty_response", message: "模型没有返回可展示的回复或工具调用，请重试。" };
      }
      return { status: "completed", stop_reason: message.stopReason };
    case "error":
      return {
        status: "failed",
        stop_reason: "error",
        error_code: message.errorMessage?.startsWith(`${REPETITION_ERROR}:`) ? REPETITION_ERROR : "model_error",
        message: message.errorMessage?.startsWith(`${REPETITION_ERROR}:`) ? REPETITION_MESSAGE : message.errorMessage ?? "Pi model request failed",
      };
    case "aborted":
      return { status: "cancelled", stop_reason: "aborted" };
    case "deferred":
      return { status: "suspended", stop_reason: "deferred" };
    case "length":
      return {
        status: "failed",
        stop_reason: "length",
        error_code: "output_truncated",
        message: "Model output was truncated",
      };
    default:
      return {
        status: "failed",
        stop_reason: String(message.stopReason),
        error_code: "unknown_pi_outcome",
        message: `Unknown Pi stop reason: ${message.stopReason}`,
      };
  }
}
