import {
  createAssistantMessageEventStream,
  type Api,
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Model,
  type SimpleStreamOptions,
} from "@earendil-works/pi-ai";

export function modelIdleTimeoutMs(env: NodeJS.ProcessEnv = process.env): number {
  const raw = env.PYROMIND_PI_MODEL_IDLE_TIMEOUT_MS?.trim();
  const value = raw ? Number(raw) : 120_000;
  if (!Number.isSafeInteger(value) || value < 0 || value > 2_147_483_647) {
    throw new Error("PYROMIND_PI_MODEL_IDLE_TIMEOUT_MS must be an integer between 0 and 2147483647");
  }
  return value;
}

/** Bound model generation, not tool execution or the subsequent human wait. */
export function modelStreamWithIdleTimeout(
  model: Model<Api>,
  options: SimpleStreamOptions | undefined,
  open: (options: SimpleStreamOptions) => AssistantMessageEventStream,
  timeoutMs = modelIdleTimeoutMs(),
): AssistantMessageEventStream {
  if (timeoutMs === 0) return open(options ?? {});
  const output = createAssistantMessageEventStream();
  const controller = new AbortController();
  const signal = options?.signal
    ? AbortSignal.any([options.signal, controller.signal])
    : controller.signal;
  void (async () => {
    let partial: AssistantMessage = {
      role: "assistant", content: [], api: model.api, provider: model.provider, model: model.id,
      timestamp: Date.now(), stopReason: "error",
      usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
    };
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
    try {
      for await (const event of open({ ...options, signal })) {
        if ("partial" in event) partial = event.partial;
        if ((event.type === "text_delta" || event.type === "thinking_delta" || event.type === "toolcall_delta") && event.delta) {
          timer.refresh();
        }
        if (timedOut && !options?.signal?.aborted) {
          throw new Error(`Model stream idle timeout: no generation progress for ${timeoutMs}ms`);
        }
        output.push(event);
      }
    } catch (error) {
      const aborted = options?.signal?.aborted === true;
      output.push({
        type: "error", reason: aborted ? "aborted" : "error",
        error: { ...partial, stopReason: aborted ? "aborted" : "error",
          errorMessage: timedOut && !aborted
            ? `Model stream idle timeout: no generation progress for ${timeoutMs}ms`
            : error instanceof Error ? error.message : String(error) },
      });
    } finally {
      clearTimeout(timer);
      output.end();
    }
  })();
  return output;
}
