import { type Api, clampThinkingLevel, InMemoryCredentialStore, type Model, type Models } from "@earendil-works/pi-ai";
import { stream as streamModel, streamSimple as streamModelSimple } from "@earendil-works/pi-ai/compat";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import { modelIdleTimeoutMs, modelStreamWithIdleTimeout } from "./model-stream.js";

export interface PiModelConfig {
  provider: string;
  modelId: string;
  apiKey: string;
  baseUrl?: string;
  api?: "openai-completions" | "openai-responses";
  contextWindow?: number;
}

export async function createPiModelRuntime(config: PiModelConfig): Promise<{
  modelRuntime: ModelRuntime;
  model: Model<Api>;
}> {
  const outputLimit = configuredOutputLimit();
  const credentials = new InMemoryCredentialStore();
  await credentials.modify(config.provider, async () => ({ type: "api_key", key: config.apiKey }));
  const modelRuntime = await ModelRuntime.create({ credentials, modelsPath: null, refreshOnCreate: false });
  let model = resolveModel(modelRuntime, config);
  if (outputLimit !== undefined) {
    model = { ...model, maxTokens: Math.min(model.maxTokens, outputLimit) };
  }

  if ((config.baseUrl && !isOfficialOpenAIBaseUrl(config.baseUrl)) ||
      (model.api === "openai-completions" && deepSeekModelFamily(model.id))) {
    const idleTimeoutMs = modelIdleTimeoutMs();
    modelRuntime.registerProvider(config.provider, {
      name: config.provider,
      baseUrl: model.baseUrl,
      api: model.api,
      streamSimple: (requestModel, context, options) =>
        modelStreamWithIdleTimeout(requestModel, options, (guardedOptions) => {
          if (requestModel.api !== "openai-completions") {
            return streamModelSimple(requestModel, context, guardedOptions);
          }
          const reasoning = guardedOptions.reasoning
            ? clampThinkingLevel(requestModel, guardedOptions.reasoning)
            : undefined;
          return streamModel(requestModel, context, {
            ...guardedOptions,
            reasoningEffort: reasoning === "off" ? undefined : reasoning,
          });
        }, idleTimeoutMs),
      models: [{
        ...model,
        input: [...model.input],
      }],
    });
    model = modelRuntime.getModel(config.provider, config.modelId) ?? model;
  }
  await modelRuntime.setRuntimeApiKey(config.provider, config.apiKey);
  const stream = modelRuntime.streamSimple.bind(modelRuntime);
  modelRuntime.streamSimple = (requestModel, context, options) => {
    const deepSeek = requestModel.api === "openai-completions" && deepSeekModelFamily(requestModel.id);
    const requested = options?.maxTokens ?? outputLimit ?? (deepSeek ? 256_000 : undefined);
    if (requested === undefined) return stream(requestModel, context, options);
    return stream(requestModel, context, {
      // Pi explicitly passes reasoning: undefined when the session selects off.
      ...(deepSeek ? { reasoning: "high" as const } : {}),
      ...options,
      maxTokens: Math.min(requested, requestModel.maxTokens, outputLimit ?? requested),
    });
  };
  return { modelRuntime, model };
}

function configuredOutputLimit(): number | undefined {
  const raw = process.env.PYROMIND_PI_MAX_OUTPUT_TOKENS?.trim();
  if (!raw) return undefined;
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error("PYROMIND_PI_MAX_OUTPUT_TOKENS must be a positive safe integer");
  }
  return value;
}

export function deepSeekModelFamily(id: string): string | undefined {
  const match = /^(?:deepseek\/)?deepseek-(flash|v4(?:\.1)?-flash|v4-pro)(?:-\d{4,8})?$/i.exec(id);
  return match?.[1]?.toLowerCase();
}

export function resolveModel(
  models: Models,
  config: Omit<PiModelConfig, "apiKey">,
): Model<Api> {
  const catalog = models.getModel(config.provider, config.modelId);
  const family = deepSeekModelFamily(config.modelId);
  if (family && (config.api ?? "openai-completions") === "openai-completions") {
    const baseUrl = config.baseUrl ?? catalog?.baseUrl ??
      (config.provider === "deepseek" ? "https://api.deepseek.com" : undefined);
    if (baseUrl && !isOfficialOpenAIBaseUrl(baseUrl) &&
        (config.api || config.baseUrl || !catalog || catalog.api === "openai-completions")) {
      return deepSeekModel(models, config, family, baseUrl, catalog);
    }
  }
  if (catalog) {
    const api = config.api ?? (
      config.baseUrl && !isOfficialOpenAIBaseUrl(config.baseUrl)
        ? "openai-completions"
        : catalog.api
    );
    return {
      ...catalog,
      api,
      ...(config.baseUrl ? { baseUrl: config.baseUrl } : {}),
      ...(config.contextWindow ? { contextWindow: config.contextWindow } : {}),
    };
  }
  if (!config.baseUrl) throw new Error(`unknown Pi model: ${config.provider}/${config.modelId}`);
  const template = models.getModels(config.provider)[0];
  if (isOfficialOpenAIBaseUrl(config.baseUrl) && !template) {
    throw new Error(`unknown Pi model: ${config.provider}/${config.modelId}`);
  }
  const api = config.api ?? (
    isOfficialOpenAIBaseUrl(config.baseUrl) ? template!.api : "openai-completions"
  );
  const contextWindow = config.contextWindow ?? 128_000;
  return {
    api,
    provider: config.provider,
    id: config.modelId,
    name: config.modelId,
    baseUrl: config.baseUrl,
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow,
    maxTokens: contextWindow,
  };
}

function deepSeekModel(
  models: Models,
  config: Omit<PiModelConfig, "apiKey">,
  family: string,
  baseUrl: string,
  catalog: Model<Api> | undefined,
): Model<"openai-completions"> {
  const flash41 = family === "flash" || family === "v4.1-flash";
  const nativeId = flash41 ? "deepseek-flash" : `deepseek-${family}`;
  const native = models.getModel("deepseek", nativeId);
  const openRouter = config.provider === "openrouter";
  const routeModel = openRouter ? models.getModel("openrouter", `deepseek/deepseek-${family}`) : undefined;
  const metadata = catalog ?? routeModel ?? native;
  const contextWindow = config.contextWindow ?? metadata?.contextWindow ?? 1_000_000;
  const guideCost = family === "v4-pro"
    ? { input: 1.74, output: 3.48, cacheRead: 0.145, cacheWrite: 0 }
    : { input: 0.14, output: 0.28, cacheRead: 0.028, cacheWrite: 0 };
  return {
    ...metadata,
    provider: config.provider,
    id: config.modelId,
    name: metadata?.name ?? config.modelId,
    api: "openai-completions",
    baseUrl,
    reasoning: true,
    input: metadata?.input ?? ["text"],
    cost: (openRouter || flash41 ? metadata?.cost : undefined) ?? guideCost,
    contextWindow,
    maxTokens: Math.min(contextWindow, metadata?.maxTokens ?? 384_000),
    thinkingLevelMap: (openRouter ? (catalog ?? routeModel)?.thinkingLevelMap : undefined) ?? {
      minimal: null, low: flash41 ? "low" : null, medium: null,
      high: "high", xhigh: openRouter ? "xhigh" : "max", max: openRouter ? "xhigh" : "max",
    },
    compat: {
      ...(catalog ?? routeModel)?.compat,
      supportsDeveloperRole: false,
      supportsStore: false,
      maxTokensField: "max_tokens",
      requiresReasoningContentOnAssistantMessages: true,
      thinkingFormat: openRouter ? "openrouter" : "deepseek",
    },
  };
}

function isOfficialOpenAIBaseUrl(value: string): boolean {
  try {
    return new URL(value).hostname.toLowerCase() === "api.openai.com";
  } catch {
    return false;
  }
}
