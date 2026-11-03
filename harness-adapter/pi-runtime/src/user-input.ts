import { randomUUID } from "node:crypto";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import type { TSchema } from "@earendil-works/pi-ai";
import { isRecord, type JsonObject } from "./protocol.js";
import type { JsonlRpcPeer } from "./rpc-peer.js";

export interface AskUserToolSpec {
  description: string;
  inputSchema: JsonObject;
}

/**
 * Tracks live ask_user waits so prompt admission can reject new turns while a
 * question is being asked, and so a cancel that races the registration never
 * leaves a fresh wait hanging.
 */
export class InputWaitRegistry {
  private readonly active = new Set<string>();
  private readonly cancelled = new Set<string>();
  private admission: { runId: string; done: Promise<void>; release: () => void } | undefined;

  reservePrompt(runId: string): void {
    if (this.hasActive()) throw inputPendingError();
    if (this.admission) throw new Error("another prompt is being admitted");
    let release!: () => void;
    const done = new Promise<void>((resolve) => { release = resolve; });
    this.admission = { runId, done, release };
  }

  releasePrompt(runId: string): void {
    if (this.admission?.runId !== runId) return;
    this.admission.release();
    this.admission = undefined;
  }

  async beginAfterAdmission(requestId: string, signal?: AbortSignal): Promise<boolean> {
    while (this.admission) {
      signal?.throwIfAborted();
      let onAbort: (() => void) | undefined;
      try {
        await Promise.race([
          this.admission.done,
          new Promise<never>((_resolve, reject) => {
            onAbort = () => reject(signal?.reason ?? new Error("request aborted"));
            signal?.addEventListener("abort", onAbort, { once: true });
          }),
        ]);
      } finally {
        if (onAbort) signal?.removeEventListener("abort", onAbort);
      }
    }
    signal?.throwIfAborted();
    return this.begin(requestId);
  }

  hasActive(): boolean {
    return this.active.size > 0;
  }

  /** Returns false when the request was already cancelled before it began. */
  begin(requestId: string): boolean {
    if (this.cancelled.has(requestId)) {
      this.cancelled.delete(requestId);
      return false;
    }
    this.active.add(requestId);
    return true;
  }

  end(requestId: string): void {
    this.active.delete(requestId);
    this.cancelled.delete(requestId);
  }

  markCancelled(requestId: string): void {
    this.cancelled.add(requestId);
  }
}

export function createAskUserTool(
  peer: JsonlRpcPeer,
  spec: AskUserToolSpec,
  waits: InputWaitRegistry,
  isAvailable: () => boolean,
): AgentTool<TSchema, unknown> {
  return {
    name: "ask_user",
    label: "ask_user",
    description: spec.description,
    parameters: spec.inputSchema as TSchema,
    // The human wait must serialize with the rest of the tool batch: no other
    // tool call may run while the question is open.
    executionMode: "sequential",
    async execute(callId, params: any, signal) {
      if (!isAvailable()) {
        throw new Error("ask_user is not available in this execution context");
      }
      if (!isRecord(params) || !Array.isArray(params.questions)) {
        throw new Error("questions must be an array");
      }
      const questions = params.questions;
      const requestId = `input:${randomUUID()}`;
      if (!await waits.beginAfterAdmission(requestId, signal)) {
        return answeredResult({ status: "cancelled" });
      }
      try {
        const answer = await peer.request(
          "input.request",
          {
            request_id: requestId,
            tool_call_id: callId,
            questions,
          },
          signal,
        );
        if (!isRecord(answer) || typeof answer.status !== "string") {
          throw new Error("invalid input response");
        }
        if (answer.status === "answered" && Array.isArray(answer.answers)) {
          const labeledAnswers = answer.answers.map((item) => {
            if (!isRecord(item) || typeof item.option_id !== "string") return item;
            const question = questions.find((question: unknown) =>
              isRecord(question) && question.id === item.question_id);
            const option = Array.isArray(question?.options)
              ? question.options.find((option: unknown) => isRecord(option) && option.id === item.option_id)
              : undefined;
            return isRecord(option) && typeof option.label === "string"
              ? { ...item, option_label: option.label }
              : item;
          });
          return answeredResult({ ...answer, answers: labeledAnswers });
        }
        return answeredResult(answer);
      } catch (error) {
        // Aborting the peer request only rejects the local promise; the
        // Python-side wait needs an explicit cleanup nudge.
        waits.markCancelled(requestId);
        void peer.request("input.cancel", { request_id: requestId }).catch(() => {});
        throw error;
      } finally {
        waits.end(requestId);
      }
    },
  };
}

export function inputPendingError(): Error & { code: string } {
  return Object.assign(new Error("input_pending: Answer or cancel the pending question first."), {
    code: "input_pending",
  });
}

function answeredResult(answer: JsonObject): {
  content: Array<{ type: "text"; text: string }>;
  details: JsonObject;
} {
  return {
    content: [{ type: "text", text: JSON.stringify(answer) }],
    details: { user_input: answer },
  };
}
