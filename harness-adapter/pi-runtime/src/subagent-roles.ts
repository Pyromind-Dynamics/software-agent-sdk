import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

export interface SubagentRoleDefinition {
  name: string;
  description: string;
  prompt: string;
  tools: string[];
  skills?: string[];
  skillPaths?: string[];
}

export function parseSubagentRoles(value: unknown): SubagentRoleDefinition[] | undefined {
  if (value === undefined || value === null) return undefined;
  if (!Array.isArray(value) || value.length > 16) throw new Error("Invalid delegated roles");
  const names = new Set<string>();
  return value.map(role => {
    if (!role || typeof role !== "object" || !/^[a-z][a-z0-9_]{0,63}$/.test(role.name)
      || names.has(role.name) || typeof role.description !== "string" || /[\r\n]/.test(role.description)
      || typeof role.prompt !== "string" || !Array.isArray(role.tools)
      || role.tools.some((name: unknown) => typeof name !== "string" || !/^[a-z][a-z0-9_]*$/.test(name))) {
      throw new Error("Invalid delegated role definition");
    }
    if (role.skills !== undefined && (!Array.isArray(role.skills) || role.skills.some((name: unknown) => typeof name !== "string" || !/^[a-z][a-z0-9_-]*$/.test(name)))) throw new Error("Invalid delegated role skills");
    names.add(role.name);
    return { name: role.name, description: role.description, prompt: role.prompt, tools: [...new Set<string>(role.tools)], skills: role.skills ? [...role.skills] : [] };
  });
}

export function writeSubagentRoles(agentDir: string, definitions: SubagentRoleDefinition[], childModule: string): void {
  const extensionDir = join(agentDir, "role-extensions");
  mkdirSync(extensionDir, { recursive: true, mode: 0o700 });
  for (const role of definitions) {
    const extension = join(extensionDir, `${role.name}.mjs`);
    writeFileSync(extension, `import { installChildHost } from ${JSON.stringify(pathToFileURL(childModule).href)};\nexport default (pi) => installChildHost(pi, ${JSON.stringify(role.name)});\n`, { mode: 0o600 });
    writeFileSync(join(agentDir, "agents", `${role.name}.md`), [
      "---", `name: ${role.name}`, "advertise: false", `description: ${JSON.stringify(role.description)}`,
      `tools: ${role.tools.join(", ")}`, `extensions: ${extension}`,
      ...(role.skills?.length ? [`skills: ${role.skills.join(", ")}`, `skillPath: ${(role.skillPaths ?? []).join(", ")}`] : []),
      "inheritProjectContext: false", "inheritGlobalContext: false", "inheritSkills: false", "---", role.prompt,
    ].join("\n"), { mode: 0o600 });
  }
}
