import { isRecord, type JsonObject } from "./protocol.js";

const MAX_BYTES = 32 * 1024;
const MARKER = "\n[轨迹窗口已截断]";
export function displayText(value: unknown): string {
  if (!isRecord(value)) return "";
  if (typeof value.text === "string") return value.text;
  return Array.isArray(value.content) ? value.content.filter(isRecord)
    .filter(block => block.type === "text" && typeof block.text === "string")
    .map(block => block.text).join("\n") : "";
}

/** Strip only native metadata before the transcript body, never interpret tool text. */
export function transcriptWindow(text: string) {
  const lines = text.split(/\r?\n/);
  const body = lines.findIndex(line => /^(?:Live transcript tail|Result transcript tail|Transcript tail|Session transcript tail|Recent output from status.json)/.test(line));
  const unavailable = /Transcript unavailable|no readable activity|no transcript|no recovered final output/i.test(text);
  const outputOnly = lines.some(line => /^Result transcript tail:/.test(line));
  // Unknown native formats fail closed instead of exposing metadata or session internals.
  if (body < 0) return { text: "", availability: "unavailable", truncated: false, outputOnly };
  let truncated = /truncated/i.test(lines.slice(body).join("\n"));
  let safe = lines.slice(body).join("\n").replace(/^([^\n]*tail) from [^\n]*$/m, "$1:" + (truncated ? " [tail truncated]" : ""));
  const bytes = Buffer.from(safe);
  if (bytes.length > MAX_BYTES) {
    let start = bytes.length - (MAX_BYTES - Buffer.byteLength(MARKER));
    while ((bytes[start] & 0xc0) === 0x80) start++;
    safe = bytes.subarray(start).toString("utf8") + MARKER;
    truncated = true;
  }
  return { text: safe, availability: unavailable ? "unavailable" : "available", truncated, outputOnly };
}

/** Presentation only: native plugin owns scheduling, transcripts and recovery. */
export class SubagentDisplay {
  private records = new Map<string, JsonObject>();
  private pending = new Set<string>();
  private settling = new Map<string, Promise<void>>();
  private queried = new Map<string, number>();
  private closed = false;
  constructor(private rpc: (method: string, params: Record<string, unknown>) => Promise<unknown>,
    private progress: (id: string) => void) {}

  settled(id: string) { return this.settling.get(id); }
  value(id: string, key: string) { return this.records.get(id)?.[key]; }
  details(id: string): JsonObject { return { subagent: this.records.get(id) ?? { version: 1, run_id: id } }; }
  update(id: string, values: JsonObject) {
    const next = { version: 1, run_id: id, ...this.records.get(id), ...values };
    const changed = JSON.stringify(next) !== JSON.stringify(this.records.get(id));
    this.records.set(id, next);
    return changed;
  }
  async poll(id: string) {
    if (this.closed || this.pending.has(id) || Date.now() - (this.queried.get(id) ?? 0) < 2000) return;
    this.pending.add(id);
    let settle!: () => void;
    this.settling.set(id, new Promise<void>(resolve => { settle = resolve; }));
    this.queried.set(id, Date.now());
    try {
      const result = await this.rpc("status", { id, view: "transcript", lines: 200 });
      if (this.closed || this.records.get(id)?.ended_at) return;
      const { outputOnly, ...window } = transcriptWindow(displayText(result));
      if (outputOnly) return; // Native completed foreground status contains output, not activity.
      const previous = this.records.get(id)?.trace;
      const comparable = isRecord(previous) ? { ...previous, updated_at: undefined, line_limit: undefined, byte_limit: undefined } : undefined;
      if (JSON.stringify(comparable) === JSON.stringify(window)) return;
      this.update(id, { trace: { ...window, updated_at: new Date().toISOString(), line_limit: 200, byte_limit: MAX_BYTES } });
      this.progress(id);
    } catch { /* Keep the last good window; polling failure is not execution failure. */ }
    finally { this.pending.delete(id); this.settling.delete(id); settle(); }
  }
  close() { this.closed = true; }
}
