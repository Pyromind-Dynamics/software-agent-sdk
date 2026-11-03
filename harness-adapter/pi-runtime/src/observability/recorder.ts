import { createHash } from "node:crypto";
import { appendFile, mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { TraceFlags, type SpanContext } from "@opentelemetry/api";
import type { LangfuseGeneration, LangfuseSpan, LangfuseTool } from "@langfuse/tracing";
import { startObservation } from "@langfuse/tracing";

const SECRET_FIELD = /^(authorization|cookie|set-cookie|api[-_]?key|.*secret.*|.*password.*|(?:access|refresh|auth)[-_]?token|token|thinkingSignature)$/i;
const DEFAULT_LIMIT = 20_000;
const MAX_CONTENT_BYTES = 2 * 1024 * 1024;
const MAX_RUN_BYTES = 32 * 1024 * 1024;
const MAX_PENDING = 64;

function previewLimit(): number {
  const configured = Number(process.env.PI_LANGFUSE_MAX_CHARS ?? DEFAULT_LIMIT);
  return Number.isSafeInteger(configured) && configured > 0 ? configured : DEFAULT_LIMIT;
}

export function observationRunKind(): "scripted_test" | "live" {
  return process.env.PYROMIND_OBSERVABILITY_RUN_KIND === "scripted_test" ? "scripted_test" : "live";
}

function structuredExcerpt(value: unknown, budget: { remaining: number }): unknown {
  if (budget.remaining <= 0) return "[omitted: display budget]";
  if (typeof value === "string") {
    const text = value.slice(0, Math.max(0, budget.remaining));
    budget.remaining -= text.length;
    return text.length < value.length ? `${text}… [truncated]` : text;
  }
  if (Array.isArray(value)) {
    const result: unknown[] = [];
    for (const entry of value) {
      if (budget.remaining <= 0) { result.push("[remaining items omitted]"); break; }
      budget.remaining -= 2;
      result.push(structuredExcerpt(entry, budget));
    }
    return result;
  }
  if (value && typeof value === "object") {
    const result: Record<string, unknown> = {};
    for (const [key, entry] of Object.entries(value)) {
      if (budget.remaining <= key.length + 4) { result._omitted = "remaining fields exceed display budget"; break; }
      budget.remaining -= key.length + 4;
      result[key] = structuredExcerpt(entry, budget);
    }
    return result;
  }
  budget.remaining -= JSON.stringify(value)?.length ?? 4;
  return value;
}

export function redact(value: unknown, secrets: readonly string[] = []): unknown {
  if (typeof value === "string") {
    let text = value.replace(/\b(?:[sp]k-lf-|sk-or-v1-|sk-)[\w-]+\b/g, "[REDACTED]");
    for (const secret of secrets) if (secret.length >= 8) text = text.split(secret).join("[REDACTED]");
    // Only credential-shaped Bearer values, not prose such as "Bearer token
    // for the gateway" in a tool description.
    text = text.replace(/\b(Bearer|Basic)\s+[A-Za-z0-9._~+/-]{16,}=*/gi, "$1 [REDACTED]");
    text = text.replace(/(^|[\r\n])(\s*(?:authorization|proxy-authorization|cookie|set-cookie)\s*:\s*)[^\r\n]+/gi, "$1$2[REDACTED]");
    return text.replace(/("(?:authorization|cookie|set-cookie|api[-_]?key|password|secret|(?:access|refresh|auth)[-_]?token)"\s*:\s*)"(?:[^"\\]|\\.)*"/gi, '$1"[REDACTED]"');
  }
  if (Array.isArray(value)) return value.map((entry) => redact(entry, secrets));
  if (value && typeof value === "object") return Object.fromEntries(
    Object.entries(value).map(([key, entry]) => [
      key,
      // A schema property such as {"api_key": {"type": "string"}} is a
      // definition, not a credential. Redact scalar values only and recurse
      // into objects so recorded tool definitions stay comparable to the
      // request the provider actually received.
      SECRET_FIELD.test(key) && (entry === null || typeof entry !== "object")
        ? "[REDACTED]"
        : redact(entry, secrets),
    ]),
  );
  return value;
}

export type HookOutcome = "no_operation" | "blocked" | "modified";
type Observation = LangfuseSpan | LangfuseGeneration | LangfuseTool;

export function classifyHookResult(result: unknown): HookOutcome {
  if (result === undefined || result === null) return "no_operation";
  if (typeof result === "object" && "block" in result && result.block === true) return "blocked";
  return "modified";
}

export interface RecorderOptions {
  conversationId: string;
  directory: string;
  secrets: string[];
  snapshot: unknown;
  evidenceBaseUrl?: string;
  readFile?: (path: string, limit: number) => Promise<Buffer>;
}

export class ObservationRecorder {
  root?: LangfuseSpan;
  parentContext?: SpanContext;
  generation?: LangfuseGeneration;
  private readonly tools = new Map<string, LangfuseTool>();
  private readonly hookChecks = new Map<string, unknown[]>();
  private readonly blockedCalls = new Set<string>();
  private readonly snapshotReads = new Set<string>();
  private readonly pending = new Set<Promise<void>>();
  private readonly evidenceWrites = new Map<Observation, Promise<void>[]>();
  private readonly reads = new Map<string, { path: string; text: string; evidence: unknown }>();
  private bytes = 0;
  private runId?: string;
  private gaps = 0;
  private readonly gapReasons = new Set<string>();
  private finished = false;
  private requestCount = 0;
  private readonly hookOutcomes: Record<string, number> = {};
  private readonly fileOperations: Array<{
    operation: string;
    path: string;
    failed: boolean;
    execution_state: "blocked_before_execution" | "execution_returned";
    returned_content_sha256?: string;
    returned_content_bytes?: number;
  }> = [];

  constructor(private readonly options: RecorderOptions) {}

  get conversationId(): string { return this.options.conversationId; }

  begin(runId: string, trace?: unknown): void {
    this.root = undefined;
    this.generation = undefined;
    this.tools.clear();
    this.hookChecks.clear();
    this.blockedCalls.clear();
    this.snapshotReads.clear();
    this.parentContext = undefined;
    if (trace && typeof trace === "object" && "trace_id" in trace && "parent_span_id" in trace &&
      typeof trace.trace_id === "string" && /^[a-f0-9]{32}$/.test(trace.trace_id) && !/^0+$/.test(trace.trace_id) &&
      typeof trace.parent_span_id === "string" && /^[a-f0-9]{16}$/.test(trace.parent_span_id) && !/^0+$/.test(trace.parent_span_id)) {
      this.parentContext = { traceId: trace.trace_id, spanId: trace.parent_span_id, traceFlags: TraceFlags.SAMPLED, isRemote: true };
    }
    this.runId = runId;
    this.bytes = 0;
    this.gaps = 0;
    this.gapReasons.clear();
    this.finished = false;
    this.requestCount = 0;
    for (const key of Object.keys(this.hookOutcomes)) delete this.hookOutcomes[key];
    this.fileOperations.length = 0;
    this.reads.clear();
  }

  safe(value: unknown): unknown { return redact(value, this.options.secrets); }

  preview(value: unknown): unknown {
    const clean = this.safe(value);
    const text = JSON.stringify(clean);
    const limit = previewLimit();
    return text.length > limit ? {
      preview: structuredExcerpt(clean, { remaining: limit }), truncated: true, original_chars: text.length,
      sha256: createHash("sha256").update(text).digest("hex"),
    } : clean;
  }

  attach(root: LangfuseSpan): void {
    this.root = root;
    root.update({ metadata: {
      conversation_id: this.options.conversationId, run_id: this.runId ?? "unassigned",
      instrumentation_version: "pyromind-lf-a2",
      run_kind: observationRunKind(),
      usage_source: observationRunKind() === "scripted_test" ? "scripted_test_values_not_actual_consumption" : "provider_reported_when_available",
      coverage: {
        provider_request: "after_extension_payload_handlers",
        hooks: ["no-progress-guard", "tool-result-status"],
        python_steps: this.parentContext ? "product_turn_boundary" : "not_instrumented", platform_tasks: "not_instrumented",
        subagents: "not_verified", third_party_hooks: "external_behavior_only",
      },
    } });
    this.fact("Harness snapshot", this.options.snapshot);
  }

  toolStarted(id: string, tool: LangfuseTool): void { this.tools.set(id, tool); }
  toolEnded(id: string): void {
    const tool = this.tools.get(id);
    const checks = this.hookChecks.get(id) ?? [];
    if (tool && checks.length) this.storeOn(tool, { hook_checks: checks }, "hook_checks_evidence");
    this.hookChecks.delete(id);
    this.blockedCalls.delete(id);
    this.tools.delete(id);
  }

  hook<T>(name: string, event: string, input: unknown, callId: string, invoke: () => T): T {
    const parent = event === "tool_result" ? this.tools.get(callId) ?? this.root : this.root;
    if (!parent) return invoke();
    const startedAt = new Date();
    const record = (outcome: HookOutcome | "exception", result: unknown) => {
      this.hookOutcomes[`${name}:${event}:${outcome}`] = (this.hookOutcomes[`${name}:${event}:${outcome}`] ?? 0) + 1;
      const detail = this.safe({ name, event, input, outcome, result, tool_call_id: callId,
        started_at: startedAt.toISOString(), duration_ms: Date.now() - startedAt.getTime() });
      const checks = this.hookChecks.get(callId) ?? [];
      checks.push(detail);
      this.hookChecks.set(callId, checks);
      if (outcome === "blocked") this.blockedCalls.add(callId);
      if (outcome === "no_operation") return;
      this.attempt(() => {
        const obs = startObservation(`Hook: ${name}`, {
          input: this.preview({ event, input, tool_call_id: callId }),
          output: this.preview({ outcome, result }),
          ...(outcome === "exception" ? { level: "ERROR" as const, statusMessage: "Hook threw" } : {}),
        }, { asType: "span", startTime: startedAt, parentSpanContext: parent.otelSpan.spanContext() });
        this.storeOn(obs, detail, "hook_evidence");
        this.endObservation(obs);
      });
    };
    try {
      const result = invoke();
      this.attempt(() => record(classifyHookResult(result), result ?? null));
      return result;
    } catch (error) {
      this.attempt(() => record("exception", String(error)));
      throw error;
    }
  }

  request(payload: unknown): void {
    const observation = this.generation;
    if (!observation) return;
    this.requestCount += 1;
    const raw = JSON.stringify(payload);
    const clean = this.safe(payload);
    const serialized = JSON.stringify(clean);
    const hash = createHash("sha256").update(serialized).digest("hex");
    const request = clean && typeof clean === "object" ? clean as Record<string, unknown> : {};
    const { messages, tools, ...parameters } = request;
    const skillContext = [...this.reads.values()].map(read => ({
      path: read.path, read_evidence: read.evidence,
      returned_text_present: serialized.includes(JSON.stringify(read.text).slice(1, -1)),
    }));
    this.attempt(() => observation.update({
      input: this.preview(Array.isArray(messages) ? messages : clean),
      metadata: {
        request_source: "provider_onPayload_after_extensions",
        request_sha256: hash,
        provider_payload_sha256: createHash("sha256").update(raw).digest("hex"),
        redaction_changed: raw !== serialized,
        evidence_url: this.evidenceUrl(hash),
        request_parameters: this.preview(parameters),
        tool_definitions_count: Array.isArray(tools) ? tools.length : 0,
        tool_definitions_source: "complete provider payload in request_evidence",
        skill_context: skillContext,
        skill_context_interpretation: "exact returned text match only; not proof of compliance",
        run_kind: observationRunKind(),
        usage_source: observationRunKind() === "scripted_test" ? "scripted_test_values_not_actual_consumption" : "provider_reported_when_available",
      },
    }));
    this.storeOn(observation, clean, "request_evidence");
  }

  private evidenceUrl(hash: string): string {
    const base = (this.options.evidenceBaseUrl ?? process.env.PYROMIND_OBSERVABILITY_API_BASE ?? "http://localhost:5173").replace(/\/$/, "");
    return `${base}/api/v2/pyromind/conversations/${encodeURIComponent(this.options.conversationId)}/execution-evidence/${hash}`;
  }

  fact(name: string, input: unknown, parent = this.root): void {
    if (!parent) return;
    this.attempt(() => {
      const clean = this.safe(input);
      const obs = parent.startObservation(name, { input: this.preview(clean) });
      this.storeOn(obs, clean, "evidence");
      this.endObservation(obs);
    });
  }

  async captureFile(path: string, phase: "initial" | "final", callId: string): Promise<void> {
    const read = this.options.readFile;
    if (!read || !this.root) return;
    const key = `${callId}:${phase}`;
    if (this.snapshotReads.has(key)) return;
    this.snapshotReads.add(key);
    const snapshot = (value: unknown) => this.fact(`Artifact snapshot: ${phase === "initial" ? "before" : "after"}`, value, this.tools.get(callId) ?? this.root);
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      const limit = 1024 * 1024;
      const buffer = await Promise.race([read(path, limit), new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error("artifact_read_timeout")), 1500);
      })]);
      let content: string;
      try {
        content = new TextDecoder("utf-8", { fatal: true }).decode(buffer);
        if (content.includes("\u0000")) throw new Error("binary_content");
      } catch {
        snapshot({
          path, phase, tool_call_id: callId, exists: true,
          source: "execution-filesystem-read", original_bytes_observed: buffer.length,
          complete: false, content_omitted: "unsupported_binary_or_partial_utf8",
        });
        this.gap("artifact_content_unsupported");
        return;
      }
      snapshot({
        path, phase, tool_call_id: callId, exists: true,
        source: "execution-filesystem-read", content,
        original_bytes_observed: buffer.length, complete: buffer.length < limit,
      });
      if (buffer.length >= limit) this.gap("artifact_read_limit");
    } catch (error) {
      const missing = error instanceof Error && "code" in error && error.code === "ENOENT";
      snapshot({ path, phase, tool_call_id: callId,
        exists: missing ? false : "unknown", source: "execution-filesystem-read",
        error: missing ? "not_found" : "read_failed" });
      if (!missing) this.gap("artifact_read_failed");
    } finally { if (timer) clearTimeout(timer); }
  }

  toolResult(name: string, args: unknown, content: unknown, failed: boolean, callId: string): void {
    const tool = this.tools.get(callId);
    const blocked = this.blockedCalls.has(callId);
    this.attempt(() => tool?.update({ metadata: {
      execution_state: blocked ? "blocked_before_execution" : "execution_returned",
      hook_checks: this.preview(this.hookChecks.get(callId) ?? []),
    } }));
    if (!args || typeof args !== "object") return;
    const record = args as Record<string, unknown>;
    const path = typeof record.path === "string" ? record.path : undefined;
    if (!path || !["read", "write", "edit"].includes(name)) return;
    const blocks = Array.isArray(content) ? content : [];
    const text = blocks.map((block) => block?.type === "text" ? block.text : "").join("\n");
    const cleanText = String(this.safe(text));
    const written = typeof record.content === "string"
      ? record.content
      : typeof record.new_str === "string"
        ? record.new_str
        : typeof record.newText === "string"
          ? record.newText
          : undefined;
    this.fileOperations.push({
      operation: name,
      path,
      failed,
      execution_state: blocked ? "blocked_before_execution" : "execution_returned",
      ...(name === "read" ? {
        returned_content_sha256: createHash("sha256").update(cleanText).digest("hex"),
        returned_content_bytes: Buffer.byteLength(cleanText),
      } : {}),
      ...(name !== "read" && written !== undefined ? {
        written_content_sha256: createHash("sha256").update(written).digest("hex"),
        written_content_bytes: Buffer.byteLength(written),
      } : {}),
    });
    const skill = path.includes(".agents/skills/") || path.endsWith("SKILL.md");
    const evidence = {
      operation: name, path, arguments: args, returned_content: content,
      failed, execution_state: blocked ? "blocked_before_execution" : "execution_returned",
      file_kind: skill ? "skill" : "workspace_file",
      scope: "tool-returned content; read may be a partial excerpt",
      artifact_verification: name === "read" ? "returned_content" : "not_independently_verified",
    };
    if (tool) {
      this.attempt(() => tool.update({ metadata: {
        file_kind: skill ? "skill" : "workspace_file", file_path: path,
        skill_read_state: skill ? blocked ? "blocked" : failed ? "failed" : "returned_content" : undefined,
      } }));
      this.storeOn(tool, evidence, "file_evidence");
    }
    if (skill && name === "read" && !failed) {
      this.reads.set(path, {
        path,
        text: cleanText,
        evidence: {
          tool_call_id: callId,
          returned_content_sha256: createHash("sha256").update(cleanText).digest("hex"),
          reference_source: "file_evidence on the corresponding tool; check its storage status",
        },
      });
    }
  }

  async finish(outcome: {
    status: "completed" | "error" | "cancelled";
    stop_reason?: string;
    had_observed_errors?: boolean;
  } = { status: "completed" }): Promise<void> {
    if (this.finished) return;
    this.finished = true;
    // Settle content saves first so the outcome reports the bytes this turn
    // actually persisted, not only the saves that happened to finish early.
    await this.flush();
    this.fact("Turn outcome", {
      status: outcome.status,
      execution_status: outcome.status,
      had_observed_errors: outcome.had_observed_errors ?? false,
      business_result: "not_evaluated_by_observer",
      run_kind: observationRunKind(),
      ...(outcome.stop_reason ? { stop_reason: outcome.stop_reason } : {}),
      model_requests: this.requestCount,
      hook_outcomes: { ...this.hookOutcomes },
      unassociated_hook_checks: [...this.hookChecks].map(([tool_call_id, checks]) => ({ tool_call_id, checks })),
      file_operations: [...this.fileOperations],
      skills_read: [...this.reads.keys()],
      observation_gaps: this.gaps,
      gap_reasons: [...this.gapReasons],
      saved_bytes: this.bytes,
      scope: {
        turn_boundary: this.parentContext ? "pi execution linked to python product turn; tool internals not instrumented" : "pi agent turn only; python product steps are outside this recorder",
        artifacts: "tool-returned content and bounded independent snapshots of public_data text files passed to read/write/edit; other paths and tool internals are outside scope",
        hooks: ["no-progress-guard", "tool-result-status"],
      },
    });
    await this.flush();
    this.attempt(() => this.root?.update({ metadata: {
      observation_gaps: this.gaps, gap_reasons: [...this.gapReasons], capture_state: this.pending.size ? "upload_or_storage_pending" : this.gaps ? "partial" : "recorded_with_declared_scope",
    } }));
    this.generation = undefined;
    this.tools.clear();
  }

  async flush(): Promise<void> {
    let timer: ReturnType<typeof setTimeout> | undefined;
    try { await Promise.race([Promise.allSettled([...this.pending]), new Promise<void>((resolve) => { timer = setTimeout(resolve, 1500); })]); }
    finally { if (timer) clearTimeout(timer); }
    if (this.pending.size) this.gap("storage_flush_timeout");
  }

  gap(reason: string, error?: unknown): void {
    this.gaps += 1;
    this.gapReasons.add(reason);
    const entry = JSON.stringify(this.safe({ time: new Date().toISOString(), conversation_id: this.options.conversationId,
      run_id: this.runId, reason, error: error === undefined ? undefined : String(error) }));
    process.stderr.write(`[observability] ${entry}\n`);
    void mkdir(this.options.directory, { recursive: true, mode: 0o700 }).then(() =>
      appendFile(join(this.options.directory, "failures.jsonl"), entry + "\n", { mode: 0o600 }),
    ).catch(() => {});
  }

  private attempt(invoke: () => void): void { try { invoke(); } catch (error) { this.gap("record_failed", error); } }

  endObservation(observation: Observation): void {
    const endTime = new Date();
    const writes = this.evidenceWrites.get(observation) ?? [];
    this.evidenceWrites.delete(observation);
    if (!writes.length) { this.attempt(() => observation.end(endTime)); return; }
    const end = Promise.allSettled(writes).then(() => this.attempt(() => observation.end(endTime)));
    this.pending.add(end);
    void end.finally(() => this.pending.delete(end));
  }

  private storeOn(observation: Observation, value: unknown, field: string): void {
    const write = this.schedule(async () => {
      let evidence: unknown;
      try { evidence = await this.save(value); }
      catch (error) {
        this.gap("content_save_failed", error);
        evidence = { status: "not_saved", reason: "content_save_failed" };
      }
      this.attempt(() => observation.update({ metadata: { [field]: evidence } }));
      if (evidence && typeof evidence === "object" && "status" in evidence && evidence.status === "not_saved") {
        this.attempt(() => observation.startObservation("Evidence unavailable", {
          level: "WARNING", input: { field }, output: evidence,
        }).end());
      }
    });
    if (write) {
      const writes = this.evidenceWrites.get(observation) ?? [];
      writes.push(write);
      this.evidenceWrites.set(observation, writes);
    } else {
      this.attempt(() => observation.update({ metadata: {
        [field]: { status: "not_saved", reason: "storage_queue_full" },
      } }));
    }
  }

  private schedule(invoke: () => Promise<void>): Promise<void> | undefined {
    if (this.pending.size >= MAX_PENDING) { this.gap("storage_queue_full"); return; }
    const promise = invoke().catch((error) => this.gap("content_save_failed", error));
    this.pending.add(promise);
    void promise.finally(() => this.pending.delete(promise));
    return promise;
  }

  private async save(value: unknown): Promise<unknown> {
    const text = JSON.stringify(this.safe(value));
    const size = Buffer.byteLength(text);
    const hash = createHash("sha256").update(text).digest("hex");
    if (size > MAX_CONTENT_BYTES || this.bytes + size > MAX_RUN_BYTES) {
      this.gap("content_budget");
      return { status: "not_saved", reason: "content_budget", bytes: size, sha256: hash };
    }
    this.bytes += size;
    await mkdir(this.options.directory, { recursive: true, mode: 0o700 });
    await writeFile(join(this.options.directory, hash + ".json"), text, { mode: 0o600, flag: "wx" }).catch((error: NodeJS.ErrnoException) => { if (error.code !== "EEXIST") throw error; });
    return { status: "saved", sha256: hash, bytes: size, local_reference: `.observability/${hash}.json`,
      url: this.evidenceUrl(hash), preview_limit: previewLimit() };
  }
}
