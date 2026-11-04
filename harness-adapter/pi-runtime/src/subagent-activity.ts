import { createHash } from "node:crypto";
import { closeSync, existsSync, openSync, readFileSync, readSync, realpathSync, statSync } from "node:fs";
import { join, relative, sep } from "node:path";
import { isRecord, type JsonObject, type JsonValue } from "./protocol.js";

const WINDOW_BYTES = 64 * 1024;
const SOURCE_BYTES = 512 * 1024;

function inside(root: string, path: string): string {
  const resolved = realpathSync(path), local = relative(realpathSync(root), resolved);
  if (!local || local === ".." || local.startsWith(".." + sep) || local.startsWith(sep)) throw Error("Activity artifact belongs to another conversation");
  return resolved;
}

/** Read plugin-owned artifacts only; never parse rendered transcript text. */
export class SubagentActivity {
  constructor(private stateRoot: string, private artifactsRoot: string, private owner: () => string[]) {}

  read(id: string): JsonObject | undefined {
    if (!/^[a-zA-Z0-9-]+$/.test(id)) return;
    try {
      const status = join(this.stateRoot, "async-subagent-runs", id, "status.json");
      let run: Record<string, unknown> | undefined;
      let children: unknown[] = [];
      if (existsSync(status)) {
        const value = JSON.parse(readFileSync(inside(this.stateRoot, status), "utf8"));
        if (isRecord(value) && value.lifecycleArtifactVersion === 3 && value.runId === id) { run = value; children = Array.isArray(value.steps) ? value.steps : []; }
      } else {
        const history = join(this.stateRoot, "async-subagent-results", "foreground-history.json");
        if (!existsSync(history)) return;
        const value = JSON.parse(readFileSync(inside(this.stateRoot, history), "utf8"));
        if (value.version !== 1 || !Array.isArray(value.runs)) return;
        run = value.runs.find((item: unknown) => isRecord(item) && item.runId === id);
        children = run && Array.isArray(run.children) ? run.children : [];
      }
      if (!run || typeof run.sessionId !== "string" || !this.owner().includes(run.sessionId)) return;
      const records: unknown[] = [];
      let truncated = false;
      for (const child of children) {
        if (!isRecord(child) || typeof child.transcriptPath !== "string" || typeof child.sessionFile !== "string") continue;
        // Both the native child session and its public transcript must belong here.
        const sessionFile = inside(join(this.stateRoot, "sessions"), child.sessionFile);
        const path = inside(this.artifactsRoot, child.transcriptPath);
        const size = statSync(path).size, start = Math.max(0, size - SOURCE_BYTES);
        const descriptor = openSync(path, "r");
        const buffer = Buffer.alloc(Math.min(size, SOURCE_BYTES));
        let count: number;
        try { count = readSync(descriptor, buffer, 0, buffer.length, start); } finally { closeSync(descriptor); }
        const rows = buffer.subarray(0, count).toString("utf8").split("\n");
        if (start) { rows.shift(); truncated = true; }
        const scope = createHash("sha256").update(sessionFile).digest("hex").slice(0, 16);
        for (const line of rows) {
          if (!line.trim()) continue;
          try {
            const row = JSON.parse(line);
            if (isRecord(row) && row.version === 1 && row.runId === id) records.push({...row, activityScope: scope});
          } catch { /* The plugin may still be appending the last record. */ }
        }
      }
      return records.length ? activityWindow(records, truncated) : undefined;
    } catch { return; } // Preserve the last good projection during native file rotation.
  }
}

export function activityWindow(records: unknown[], sourceTruncated = false): JsonObject {
  const items = new Map<string, JsonObject>();
  let sequence = 0, truncated = sourceTruncated;
  const bounded = (text: string, limit = 8192) => {
    if (text.length <= limit) return text;
    truncated = true; return text.slice(0, limit) + "\n[内容已截断]";
  };
  const content = (value: unknown): JsonObject[] => {
    if (typeof value === "string") return [{type: "text", text: bounded(value)}];
    return Array.isArray(value) ? value.flatMap(block => isRecord(block) && block.type === "text" && typeof block.text === "string" ? [{type: "text", text: bounded(block.text)}] : []) : [];
  };
  for (const raw of records) {
    if (!isRecord(raw)) continue;
    if (raw.recordType === "truncated") { truncated = true; continue; }
    const scope = String(raw.activityScope ?? raw.runId ?? "activity");
    const tool = (id: string, name: string) => {
      const key = `${scope}:tool:${id}`;
      if (!items.has(key)) items.set(key, {kind: "operation", item_id: key, started_seq: ++sequence, name, category: "tool", status: "running", arguments: {}, output: []});
      return items.get(key)!;
    };
    const args = (value: unknown): JsonValue => {
      if (!isRecord(value)) return {};
      const encoded = JSON.stringify(value);
      if (encoded.length <= 8192) return value as JsonObject;
      truncated = true; return bounded(encoded);
    };
    const message = raw.message;
    if (raw.recordType === "message" && isRecord(message)) {
      if (["user", "assistant"].includes(String(message.role))) {
        const blocks = typeof message.content === "string" ? [{type: "text", text: message.content}] : Array.isArray(message.content) ? message.content : [];
        const identity = createHash("sha256").update(JSON.stringify([scope, message.timestamp ?? raw.timestamp, message.content])).digest("hex").slice(0, 24);
        blocks.forEach((block, index) => {
          if (!isRecord(block)) return;
          if (block.type === "text" && typeof block.text === "string" && block.text.trim()) {
            const key = `${scope}:message:${identity}:${index}`;
            items.set(key, {kind: "message", item_id: key, started_seq: ++sequence, role: String(message.role), content: content([block]), status: "completed"});
          } else if (message.role === "assistant" && block.type === "toolCall" && typeof block.id === "string" && typeof block.name === "string") tool(block.id, block.name).arguments = args(block.arguments);
        });
      } else if (message.role === "toolResult" && typeof message.toolCallId === "string") {
        const item = tool(message.toolCallId, String(message.toolName ?? "tool"));
        item.output = content(message.content); item.status = message.isError ? "failed" : "completed"; item.completed_seq = ++sequence;
      }
    } else if (raw.recordType === "tool_start" && typeof raw.toolCallId === "string" && typeof raw.toolName === "string") {
      const item = tool(raw.toolCallId, raw.toolName);
      if (typeof raw.argsPayload === "string") {
        try { item.arguments = args(JSON.parse(raw.argsPayload)); } catch { item.arguments = bounded(raw.argsPayload); truncated = true; }
      }
    } else if (raw.recordType === "tool_end" && typeof raw.toolCallId === "string") {
      const item = tool(raw.toolCallId, String(raw.toolName ?? "tool"));
      item.status = raw.isError ? "failed" : "completed"; item.completed_seq = ++sequence;
    }
  }
  const window = [...items.values()];
  while (window.length > 128 || Buffer.byteLength(JSON.stringify(window)) > WINDOW_BYTES) { window.shift(); truncated = true; }
  return {items: window, truncated};
}
