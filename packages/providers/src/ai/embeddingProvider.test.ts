import { afterEach, describe, expect, it, vi } from "vitest";

import { createEmbeddingProvider } from "./embeddingProvider";

function mockEmbeddingEndpoint() {
  const requests: { model: string; input: string[]; dimensions?: number }[] = [];
  const signals = new Set<AbortSignal | null | undefined>();
  vi.stubGlobal("fetch", async (_url: RequestInfo | URL, init: RequestInit) => {
    const request = JSON.parse(String(init.body)) as (typeof requests)[number];
    requests.push(request);
    signals.add(init.signal);
    return Response.json({
      data: request.input.map((_, index) => ({ index, embedding: Array(1536).fill(0.5) })),
      usage: { prompt_tokens: request.input.length },
    });
  });
  return { requests, signals };
}

describe("embedding requests", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("requests 1536 dimensions from the default large model, batching chunks with separate timeouts", async () => {
    const { requests, signals } = mockEmbeddingEndpoint();
    const provider = createEmbeddingProvider({ OPENAI_API_KEY: "test" })!;
    const usage = vi.fn();
    const vectors = await provider.embed(Array.from({ length: 300 }, (_, i) => `chunk ${i}`), usage);

    expect(vectors).toHaveLength(300);
    expect(requests.map(({ input }) => input.length)).toEqual([128, 128, 44]);
    expect(signals.size).toBe(3);
    for (const request of requests) expect(request).toMatchObject({ model: "text-embedding-3-large", dimensions: 1536 });
    expect(usage).toHaveBeenCalledWith(expect.objectContaining({ inputTokens: 300, model: "text-embedding-3-large" }));
  });

  it("sends the explicitly configured model without imposing the large model's dimensions", async () => {
    const { requests } = mockEmbeddingEndpoint();
    const environment = { OPENAI_API_KEY: "test", AI_EMBEDDING_MODEL: "text-embedding-3-small" };
    const provider = createEmbeddingProvider(environment)!;
    await provider.embed(["Greeting"]);

    expect(requests[0]).toMatchObject({ model: "text-embedding-3-small", input: ["Greeting"] });
    expect(requests[0]).not.toHaveProperty("dimensions");
    expect(provider.fingerprint).not.toBe(createEmbeddingProvider({ OPENAI_API_KEY: "test" })!.fingerprint);
  });
});
