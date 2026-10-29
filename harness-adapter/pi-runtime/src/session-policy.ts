import { realpathSync } from "node:fs";
import { isAbsolute, resolve } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { inside } from "./workspace-policy.js";

export function capabilityCeiling(allowed: string[] | undefined) {
  return (pi: ExtensionAPI) => {
    if (!allowed) return;
    const ceiling = new Set(allowed);
    const activate = () => pi.setActiveTools(pi.getAllTools().map(tool => tool.name).filter(name => ceiling.has(name)));
    pi.on("session_start", activate);
    pi.on("before_agent_start", activate);
    pi.on("tool_call", event => {
      if (!ceiling.has(event.toolName)) return { block: true, reason: "Tool is not enabled for this conversation" };
    });
  };
}

export function restrictSkillReads(
  tool: AgentTool, workspace: string, resourceRoots: string[], allowedRoots: string[] | undefined,
  skillsDirectory?: string,
): AgentTool {
  if (!allowedRoots || !["read", "grep", "find", "ls"].includes(tool.name)) return tool;
  return { ...tool, execute: async (id, args: any, signal, update) => {
    if (typeof args.path === "string") {
      const path = args.path.replaceAll("\\", "/");
      const candidate = path.startsWith(".agents/skills/") && skillsDirectory
        ? resolve(skillsDirectory, path.slice(".agents/skills/".length))
        : path === ".agents/skills" && skillsDirectory ? skillsDirectory
        : isAbsolute(path) ? path : resolve(workspace, path);
      const canonical = realpathSync(candidate);
      if (resourceRoots.some(root => inside(canonical, root)) && !allowedRoots.some(root => inside(canonical, root))) {
        throw new Error("Skill is not enabled for this conversation");
      }
    }
    return tool.execute(id, args, signal, update);
  } };
}
