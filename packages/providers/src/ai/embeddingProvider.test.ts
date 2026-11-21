import { afterEach, describe, expect, it, vi } from "vitest";

const doEmbed = vi.hoisted(() => vi.fn());

vi.mock("@ai-sdk/openai-compatible", () => ({
  createOpenAICompatible: () => ({
    embeddingModel: (modelId: string) => ({ specificationVersion: "v4", provider: "test", modelId, maxEmbeddingsPerCall: 2048, supportsParallelCalls: true, doEmbed }),
  }),
}));

import { createEmbeddingProvider, OpenAiCompatibleEmbeddingProvider } from "./embeddingProvider";

describe("embedding requests", () => {
  afterEach(() => vi.clearAllMocks());

  it("splits a large document into requests the provider accepts, each with its own timeout", async () => {
    doEmbed.mockImplementation(async ({ values }: { values: string[] }) => ({ embeddings: values.map(() => new Array(1536).fill(0.5)), usage: { tokens: values.length }, warnings: [] }));
    const usage = vi.fn();

    const vectors = await new OpenAiCompatibleEmbeddingProvider({ apiKey: "test", timeoutMs: 1234 }).embed(Array.from({ length: 300 }, (_, index) => `chunk ${index}`), usage);

    expect(vectors).toHaveLength(300);
    const requests = doEmbed.mock.calls.map(([options]) => options as { values: string[]; abortSignal: AbortSignal; dimensions?: number });
    expect(requests.map((request) => request.values.length)).toEqual([128, 128, 44]);
    expect(new Set(requests.map((request) => request.abortSignal)).size).toBe(3);
    expect(requests.map((request) => request.dimensions)).toEqual([1536, 1536, 1536]);
    expect(usage).toHaveBeenCalledWith(expect.objectContaining({ inputTokens: 300, model: "text-embedding-3-large" }));
  });

  it("preserves existing small model indexes and gives the new default a distinct fingerprint", async () => {
    doEmbed.mockResolvedValue({ embeddings: [new Array(1536).fill(0.5)], usage: { tokens: 1 }, warnings: [] });
    const fallback = createEmbeddingProvider({ OPENAI_API_KEY: "test", AI_EMBEDDING_MODEL: " " })!;
    const override = createEmbeddingProvider({ OPENAI_API_KEY: "test", AI_EMBEDDING_MODEL: "text-embedding-3-small" })!;
    const usage = vi.fn();

    expect(fallback.fingerprint).not.toBe(override.fingerprint);
    await override.embed(["Greeting"], usage);
    expect(doEmbed.mock.calls[0]![0].dimensions).toBeUndefined();
    expect(usage).toHaveBeenCalledWith(expect.objectContaining({ model: "text-embedding-3-small" }));
  });
});
