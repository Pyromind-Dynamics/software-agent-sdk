import { provideGenomeHost } from "@agentgenome/pi-extension";
import { createEventBus, type DefaultResourceLoader } from "@earendil-works/pi-coding-agent";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { JsonlRpcPeer } from "./rpc-peer.js";

/** Host capability only. Pi's package loader instantiates the extension itself. */
export function genomeHostAccess(peer: JsonlRpcPeer, enabled: boolean) {
  const eventBus = createEventBus();
  const binding = provideGenomeHost(eventBus, (method, payload, signal) => {
    if (!enabled) return Promise.reject(new Error("Historical experiences are disabled in this session"));
    return peer.request(method, payload, signal);
  });
  const packageRoot = dirname(fileURLToPath(import.meta.resolve("@agentgenome/pi-extension")));
  return {
    eventBus,
    additionalExtensionPaths: enabled ? [packageRoot] : [],
    assertLoaded(loader: DefaultResourceLoader) {
      if (!enabled) return;
      const { extensions, errors } = loader.getExtensions();
      if (errors.length) throw new Error(`Pi extension loading failed: ${JSON.stringify(errors)}`);
      if (!binding.discovered || !extensions.some((extension) => extension.tools.has("genome_run"))) {
        throw new Error("AgentGenome plugin did not connect to the SDK execution host; local execution is disabled");
      }
    },
  };
}
