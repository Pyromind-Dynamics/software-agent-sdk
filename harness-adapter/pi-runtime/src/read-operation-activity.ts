import { readFileSync } from "node:fs";
import { join } from "node:path";
import { SubagentActivity } from "./subagent-activity.js";

// Read-only recovery entrypoint. No agent session, tools, model or platform calls.
const input = JSON.parse(readFileSync(0, "utf8"));
if (typeof input.workspace !== "string" || !Array.isArray(input.ids) || input.ids.length > 32) throw Error("Invalid activity request");
const piRoot = join(input.workspace, "pi");
const parent = join(piRoot, "session.jsonl");
const header = JSON.parse(readFileSync(parent, "utf8").split("\n", 1)[0]);
if (header.type !== "session" || typeof header.id !== "string") throw Error("Invalid parent session");
const reader = new SubagentActivity(join(piRoot, "subagents"), join(piRoot, "subagent-artifacts"), () => [parent, header.id]);
process.stdout.write(JSON.stringify(Object.fromEntries(input.ids.flatMap((id: unknown) => {
  if (typeof id !== "string") return [];
  const activity = reader.read(id);
  return activity ? [[id, activity]] : [];
}))));
