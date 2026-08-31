import { type Api, InMemoryCredentialStore, type Model, type Models } from "@earendil-works/pi-ai";
import { stream as streamModel, streamSimple as streamModelSimple } from "@earendil-works/pi-ai/compat";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";

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
  const credentials = new InMemoryCredentialStore();
  await credentials.modify(config.provider, async () => ({ type: "api_key", key: config.apiKey }));
  const modelRuntime = await ModelRuntime.create({ credentials, modelsPath: null, refreshOnCreate: false });
  let model = resolveModel(modelRuntime, config);

  if (config.baseUrl && !isOfficialOpenAIBaseUrl(config.baseUrl)) {
    modelRuntime.registerProvider(config.provider, {
      name: config.provider,
      baseUrl: model.baseUrl,
      api: model.api,
      streamSimple: (requestModel, context, options) =>
        requestModel.api === "openai-completions"
          ? streamModel(
            requestModel,
            context,
            // Do not derive max_tokens from context usage for custom Chat
            // Completions servers.
            options ? { ...options } : undefined,
          )
          : streamModelSimple(requestModel, context, options),
      models: [{
        id: model.id,
        name: model.name,
        api: model.api,
        baseUrl: model.baseUrl,
        reasoning: model.reasoning,
        input: [...model.input],
        cost: model.cost,
        contextWindow: model.contextWindow,
        maxTokens: model.maxTokens,
        compat: model.compat,
      }],
    });
    model = modelRuntime.getModel(config.provider, config.modelId) ?? model;
  }
  await modelRuntime.setRuntimeApiKey(config.provider, config.apiKey);
  return { modelRuntime, model };
}

export function resolveModel(
  models: Models,
  config: Omit<PiModelConfig, "apiKey">,
): Model<Api> {
  const catalog = models.getModel(config.provider, config.modelId);
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

function isOfficialOpenAIBaseUrl(value: string): boolean {
  try {
    return new URL(value).hostname.toLowerCase() === "api.openai.com";
  } catch {
    return false;
  }
}
