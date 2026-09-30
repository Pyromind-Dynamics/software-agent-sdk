import type { ExecutionAccess } from "./workflow-execution.js";
import { isRecord, type JsonObject } from "./protocol.js";

export type ModelJudge = (request: JsonObject, signal: AbortSignal) => Promise<string>;

export async function serveJudge(access: ExecutionAccess, directory: string,
  model: ModelJudge, signal: AbortSignal): Promise<void> {
  if (!access.readFile) throw new Error("independent file read unavailable");
  let request: JsonObject | undefined;
  const deadline = Date.now() + 120_000;
  while (!signal.aborted && Date.now() < deadline) {
    try {
      const data = await access.readFile(`${directory}/request.json`, 262144);
      const value: unknown = JSON.parse(data.toString("utf8"));
      if (!isRecord(value) || typeof value.request_id !== "string" ||
          typeof value.criterion !== "string" || typeof value.evidence !== "string") throw new Error("invalid judge request");
      request = { request_id: value.request_id, criterion: value.criterion, evidence: value.evidence };
      break;
    } catch (error) {
      const missing = error instanceof Error && (("code" in error && error.code === "ENOENT") || ("status" in error && error.status === 404));
      if (!missing) throw error;
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
  }
  if (!request || signal.aborted) return;
  const remaining = Math.max(1, deadline - Date.now());
  const boundedSignal = AbortSignal.any([signal, AbortSignal.timeout(remaining)]);
  let response: JsonObject;
  try {
    const text = await model(request, boundedSignal);
    if (Buffer.byteLength(text) > 262144) throw new Error("judge output exceeds size limit");
    const verdict: unknown = JSON.parse(text);
    if (!isRecord(verdict) || typeof verdict.passed !== "boolean" ||
        typeof verdict.reason !== "string" || !verdict.reason.trim()) throw new Error("invalid judge verdict");
    response = { request_id: request.request_id!, verdict: { passed: verdict.passed, reason: verdict.reason } };
  } catch (error) {
    response = { request_id: request.request_id!, error: error instanceof Error ? error.message : "judge failed" };
  }
  if (signal.aborted) return;
  await access.writeFile(`${directory}/response.json`, Buffer.from(JSON.stringify(response)));
  await access.writeFile(`${directory}/response.ready`, Buffer.from("ready"));
}
