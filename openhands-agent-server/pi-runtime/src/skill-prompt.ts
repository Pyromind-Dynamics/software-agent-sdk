import {
  formatSkillsForSystemPrompt,
  loadSkills,
  type ExecutionEnv,
} from "@earendil-works/pi-agent-core";

export async function extendSystemPromptWithSkills(
  basePrompt: string,
  env: ExecutionEnv,
  skillDirs: string[],
): Promise<string> {
  if (skillDirs.length === 0) return basePrompt;

  const { skills, diagnostics } = await loadSkills(env, skillDirs);
  if (diagnostics.length > 0) {
    const summary = diagnostics
      .map((diagnostic) => `${diagnostic.path}: ${diagnostic.message}`)
      .join("; ");
    throw new Error(`Failed to load configured Pi skills: ${summary}`);
  }
  if (skills.length === 0) {
    throw new Error("Configured Pi skill directories did not contain any valid skills");
  }

  const skillPrompt = formatSkillsForSystemPrompt(skills);
  return skillPrompt ? `${basePrompt.trimEnd()}\n\n${skillPrompt}` : basePrompt;
}
