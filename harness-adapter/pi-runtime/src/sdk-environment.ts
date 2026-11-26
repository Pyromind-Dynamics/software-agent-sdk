import { realpath } from "node:fs/promises";
import { dirname, isAbsolute } from "node:path";
import { isRecord } from "./protocol.js";
import type { JsonlRpcPeer } from "./rpc-peer.js";

const REQUIRED_ENVIRONMENT_NAMES = [
  "PYROMIND_API_KEY",
  "PYROMIND_CLUSTER",
  "PYROMIND_BASE_URL",
  "PYROMIND_SDK_PYTHON",
] as const;
const ENVIRONMENT_NAMES = [
  ...REQUIRED_ENVIRONMENT_NAMES,
  "PYROMIND_SANDBOX_DEFAULT_IMAGE",
  "PYROMIND_TASK_ASSOCIATIONS",
  "PYROMIND_TASK_CONTEXTS",
  "PYROMIND_PROJECT_EXECUTIONS",
  "PYROMIND_DEBUG_STATE_PATH",
  "LLM_BASE_URL",
  "LLM_AUTH_TOKEN",
  "LLM_MODEL",
] as const;

export async function platformSdkEnvironment(
  peer: JsonlRpcPeer,
  signal?: AbortSignal,
): Promise<Record<string, string>> {
  const result = await peer.request("platform.sdk_environment", {}, signal);
  if (!isRecord(result)) throw new Error("Invalid platform SDK environment");
  if (Object.keys(result).length === 0) return {};
  const environment: Record<string, string> = {};
  for (const [name, value] of Object.entries(result)) {
    if (!ENVIRONMENT_NAMES.some((allowed) => allowed === name) ||
        typeof value !== "string" || !value.trim() || value.includes("\0")) {
      throw new Error("Invalid platform SDK environment");
    }
    environment[name] = value;
  }
  if (!isAbsolute(environment.PYROMIND_SDK_PYTHON ?? "")) {
    throw new Error("Invalid platform SDK interpreter path");
  }
  if (Object.keys(environment).length === 1) return environment;
  if (REQUIRED_ENVIRONMENT_NAMES.some((name) => !environment[name]) ||
      (environment.PYROMIND_DEBUG_STATE_PATH !== undefined &&
       !isAbsolute(environment.PYROMIND_DEBUG_STATE_PATH))) {
    throw new Error("Incomplete platform SDK environment");
  }
  let endpoint: URL;
  try { endpoint = new URL(environment.PYROMIND_BASE_URL); }
  catch { throw new Error("Invalid platform SDK endpoint"); }
  if (!["http:", "https:"].includes(endpoint.protocol) || endpoint.username || endpoint.password) {
    throw new Error("Invalid platform SDK endpoint");
  }
  return environment;
}

export function platformSdkNetworkDomains(environment?: NodeJS.ProcessEnv): string[] {
  return environment?.PYROMIND_API_KEY && environment.PYROMIND_BASE_URL
    ? [new URL(environment.PYROMIND_BASE_URL).hostname] : [];
}

export async function platformSdkReadRoots(environment?: NodeJS.ProcessEnv): Promise<string[]> {
  const executable = environment?.PYROMIND_API_KEY && environment.PYROMIND_SDK_PYTHON;
  if (!executable) return [];
  return [dirname(dirname(executable)), dirname(dirname(await realpath(executable)))];
}
