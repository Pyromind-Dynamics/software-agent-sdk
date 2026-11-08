// pi-subagents registers its own work and undelivered wakes through this host contract.
const key = Symbol.for("@agegr/pi-web/session-liveness/v1");

interface Provider {
  sessionId: string;
  sessionFile?: string;
  isActive(): boolean;
}

interface Registry {
  version: 1;
  register(provider: Provider): () => void;
  hasActiveProvider(session: { sessionId: string; sessionFile?: string }): boolean;
}

const globals = globalThis as typeof globalThis & { [key]?: Registry };

export function subagentLiveness(): Registry {
  if (globals[key]) return globals[key];
  const providers = new Set<Provider>();
  return globals[key] = {
    version: 1,
    register(provider) {
      providers.add(provider);
      return () => { providers.delete(provider); };
    },
    hasActiveProvider({ sessionId, sessionFile }) {
      return [...providers].some((provider) =>
        provider.sessionId === sessionId &&
        (!provider.sessionFile || provider.sessionFile === sessionFile) &&
        provider.isActive());
    },
  };
}
