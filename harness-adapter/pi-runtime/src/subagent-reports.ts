import { constants } from "node:fs";
import { open, realpath } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { isRecord } from "./protocol.js";

const RUN_ID = "[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}";
const REPORT_NAME = new RegExp(`^(${RUN_ID})_(search|general_purpose)(?:_0)?_output\\.md$`);
const TARGET = new RegExp(`public_data/subagents/(${RUN_ID})/report\\.md`, "g");
const MAX_REPORT_BYTES = 16 * 1024 * 1024;
const escapeRegex = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

function strings(value: unknown): string[] {
  if (typeof value === "string") return [value];
  if (Array.isArray(value)) return value.flatMap(strings);
  if (isRecord(value)) return Object.values(value).flatMap(strings);
  return [];
}
function replace<T>(value: T, source: string, target: string): T {
  if (typeof value === "string") return value.split(source).join(target) as T;
  if (Array.isArray(value)) return value.map(item => replace(item, source, target)) as T;
  if (isRecord(value)) return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, replace(item, source, target)])) as T;
  return value;
}

/** Materializes native final reports through the same coordinated, policy-bound write as the parent. */
export class SubagentReports {
  private readonly completed = new Set<string>();
  private readonly pending = new Map<string, Promise<void>>();
  private readonly failures = new Map<string, string>();
  private readonly root: string;

  constructor(workspace: string, private readonly parentId: () => string, private readonly writer: AgentTool) {
    this.root = join(workspace, "pi", "subagent-artifacts");
  }

  beginTurn() { this.failures.clear(); }

  async transform<C, D>(content: C, details: D, signal?: AbortSignal): Promise<{ content: C; details: D }> {
    const original = content;
    const text = strings([content, details]).join("\n");
    const sources = new Set<string>();
    const pattern = new RegExp(`${escapeRegex(this.root)}/(${RUN_ID})_(?:search|general_purpose)(?:_0)?_output\\.md`, "g");
    for (const match of text.matchAll(pattern)) sources.add(match[0]);
    // Successful tool results are persisted with workspace references. Reconstruct from
    // native metadata after restart, without a second durable report registry.
    const ids = new Set([...text.matchAll(TARGET)].map(match => match[1]));
    if (isRecord(details) && Array.isArray(details.runs) && strings(content).some(value => /preview truncated|Full output:|Output saved to:/.test(value))) {
      for (const run of details.runs) if (isRecord(run) && typeof run.runId === "string" && new RegExp(`^${RUN_ID}$`).test(run.runId)) ids.add(run.runId);
    }
    for (const id of ids) {
      for (const agent of ["search", "general_purpose"]) for (const suffix of ["_0", ""]) {
        const path = join(this.root, `${id}_${agent}${suffix}_output.md`);
        try { await this.metadata(path); sources.add(path); } catch { /* Other native filename variants need not exist. */ }
      }
    }
    for (const source of sources) {
      const id = REPORT_NAME.exec(basename(source))![1];
      const target = `public_data/subagents/${id}/report.md`;
      if (!strings(content).some(value => value.includes(source) || value.includes(target) || value.includes(`run ID: ${id}`) || /preview truncated/.test(value))) continue;
      try {
        await this.sync(source, target, signal);
        content = replace(content, source, target);
        details = replace(details, source, target);
        for (const value of strings(content)) {
          for (const notice of value.matchAll(new RegExp(`\\[子任务已完成，但完整报告同步失败；run ID: ${id}。[^\\]]*\\]`, "g"))) {
            content = replace(content, notice[0], `Full output: ${target}. Read it with read.`);
          }
        }
        if (!strings(content).some(value => value.includes(target)) && strings(original).some(value => /preview truncated/.test(value))) {
          content = this.append(content, `Full output: ${target}. Read it with read.`);
        }
      } catch (error) {
        const reason = this.failures.get(source) ?? (error instanceof Error ? error.message : "write failed");
        const notice = `[子任务已完成，但完整报告同步失败；run ID: ${id}。${reason}。不要通过 terminal 查找或重新执行子任务。]`;
        content = replace(replace(content, source, notice), target, notice);
        if (!strings(content).some(value => value.includes(notice))) content = this.append(content, notice);
        // Keep native references in non-model details so a later turn can retry.
      }
    }
    return { content, details };
  }

  private append<C>(content: C, text: string): C {
    if (typeof content === "string") return `${content}\n\n${text}` as C;
    if (Array.isArray(content)) return [...content, { type: "text", text }] as C;
    return content;
  }

  private async readFile(path: string, limit: number): Promise<string> {
    if (resolve(path) !== path || (await realpath(path)) !== path || (await realpath(this.root)) !== this.root) throw new Error("报告路径校验失败");
    const file = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    try {
      const stat = await file.stat();
      if (!stat.isFile() || stat.size > limit) throw new Error("报告不是普通文件或超过大小限制");
      const bytes = Buffer.alloc(limit + 1);
      let count = 0;
      while (count < bytes.length) {
        const read = await file.read(bytes, count, bytes.length - count, null);
        if (!read.bytesRead) break;
        count += read.bytesRead;
      }
      if (count > limit) throw new Error("报告超过大小限制");
      return bytes.subarray(0, count).toString("utf8");
    } finally { await file.close(); }
  }

  private async metadata(source: string) {
    const match = REPORT_NAME.exec(basename(source));
    if (!match || join(this.root, basename(source)) !== source) throw new Error("报告路径校验失败");
    const meta = JSON.parse(await this.readFile(source.replace(/_output\.md$/, "_meta.json"), 1024 * 1024));
    if (!isRecord(meta) || meta.runId !== match[1] || meta.agent !== match[2] || meta.parentSessionId !== this.parentId()) throw new Error("报告不属于当前会话的原生运行记录");
    return meta;
  }

  private async sync(source: string, target: string, signal?: AbortSignal) {
    signal?.throwIfAborted();
    if (this.completed.has(source)) return;
    if (this.failures.has(source)) throw new Error(this.failures.get(source));
    const active = this.pending.get(source);
    if (active) return active;
    const work = (async () => {
      try {
        await this.metadata(source);
        const content = await this.readFile(source, MAX_REPORT_BYTES);
        const result = await this.writer.execute(`subagent-report:${basename(source)}`, { path: target, content }, signal);
        if (isRecord(result) && result.isError === true) throw new Error("文件写入失败");
        this.completed.add(source);
      } catch (error) {
        const code = (error as NodeJS.ErrnoException).code;
        const reason = code === "ENOENT" ? "原始报告已清理或不可用" : "报告校验或工作区写入失败，后续轮次可重试";
        this.failures.set(source, reason);
        throw error;
      } finally { this.pending.delete(source); }
    })();
    this.pending.set(source, work);
    return work;
  }
}
