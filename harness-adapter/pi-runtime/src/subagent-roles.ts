import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { parseFrontmatter } from "@earendil-works/pi-coding-agent";

export interface SubagentRoleDefinition {
  name: string;
  description: string;
  prompt: string;
  tools: string[];
  skills?: string[];
  skillPaths?: string[];
}

const builtAgents = new URL("./agents/", import.meta.url);
const bundledAgents = existsSync(builtAgents) ? builtAgents : new URL("../agents/", import.meta.url);

export function readBundledSubagentRole(name: string): string {
  return readFileSync(new URL(`${name}.md`, bundledAgents), "utf8");
}

export function parseSubagentRoles(value: unknown): SubagentRoleDefinition[] | undefined {
  if (value === undefined || value === null) return undefined;
  if (!Array.isArray(value) || value.length > 16) throw new Error("Invalid delegated roles");
  const names = new Set<string>();
  return value.map(role => {
    if (!role || typeof role !== "object"
      || Object.keys(role).some(key => !["name", "tools", "skills"].includes(key))
      || typeof role.name !== "string" || !/^[a-z][a-z0-9_]{0,63}$/.test(role.name)
      || names.has(role.name) || !Array.isArray(role.tools)
      || role.tools.some((name: unknown) => typeof name !== "string" || !/^[a-z][a-z0-9_]*$/.test(name))) {
      throw new Error("Invalid delegated role definition");
    }
    if (role.skills !== undefined && (!Array.isArray(role.skills) || role.skills.some((name: unknown) => typeof name !== "string" || !/^[a-z][a-z0-9_-]*$/.test(name)))) throw new Error("Invalid delegated role skills");
    const { frontmatter, body } = parseFrontmatter(readBundledSubagentRole(role.name));
    const description = frontmatter.description;
    if (frontmatter.name !== role.name || !body || typeof description !== "string"
      || !description.trim() || /[\r\n]/.test(description)) {
      throw new Error(`Invalid bundled role: ${role.name}`);
    }
    names.add(role.name);
    return { name: role.name, description, prompt: body, tools: [...new Set<string>(role.tools)], skills: role.skills ? [...role.skills] : [] };
  });
}

export function writeSubagentRoles(agentDir: string, definitions: SubagentRoleDefinition[], childModule: string): void {
  const extensionDir = join(agentDir, "role-extensions");
  mkdirSync(extensionDir, { recursive: true, mode: 0o700 });
  for (const role of definitions) {
    const extension = join(extensionDir, `${role.name}.mjs`);
    writeFileSync(extension, `import { installChildHost } from ${JSON.stringify(pathToFileURL(childModule).href)};\nexport default (pi) => installChildHost(pi, ${JSON.stringify(role.name)});\n`, { mode: 0o600 });
    writeFileSync(join(agentDir, "agents", `${role.name}.md`), [
      "---", `name: ${role.name}`, "advertise: true", `description: ${JSON.stringify(role.description)}`,
      `tools: ${role.tools.join(", ")}`, `extensions: ${extension}`,
      ...(role.skills?.length ? [`skills: ${role.skills.join(", ")}`, `skillPath: ${(role.skillPaths ?? []).join(", ")}`] : []),
      "inheritProjectContext: false", "inheritGlobalContext: false", "inheritSkills: false", "---", role.prompt,
    ].join("\n"), { mode: 0o600 });
  }
}
