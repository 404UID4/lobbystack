import { createHash, randomUUID } from "node:crypto";
import { eq, sql } from "drizzle-orm";
import { afterAll, describe, expect, it, vi } from "vitest";

import type { JobEnvelope } from "@lobbystack/contracts";
import { businesses, createDatabaseClient, knowledgeChunks, unitEconomicsEvents, withBusinessTransaction, type Database } from "@lobbystack/db";
import { handleJob } from "./handlers";
import { embedKnowledgeChunks, type KnowledgeEmbeddingProvider } from "./knowledgeEmbeddings";

const testUrl = process.env.LOBBYSTACK_RELIABILITY_TEST_DATABASE_URL;
if (testUrl) {
  const url = new URL(testUrl);
  if (process.env.NODE_ENV === "production" || !["localhost", "127.0.0.1", "[::1]"].includes(url.hostname) || !/test/i.test(url.pathname)) {
    throw new Error("Embedding integration tests require a dedicated local test database.");
  }
}
const client = testUrl ? createDatabaseClient("lobbystack_migrator", { DATABASE_URL: testUrl }) : undefined;
afterAll(() => client?.pool.end());

function vectorFor(text: string, fingerprint: string) {
  const coordinate = createHash("sha256").update(fingerprint + text).digest().readUInt16BE() % 1536;
  return Array.from({ length: 1536 }, (_, i) => Number(i === coordinate));
}

function job(businessId: string, type: JobEnvelope["type"]): JobEnvelope {
  return { jobId: randomUUID(), businessId, type, queue: "bulk", payload: { url: "https://example.invalid" }, trace: {}, idempotencyKey: randomUUID(), scheduled: false };
}

describe.skipIf(!client)("knowledge embedding reuse", () => {
  it("bills only new text, isolates businesses, and pairs each re-embedded vector with its source", async () => {
    const businessId = randomUUID();
    const otherBusiness = randomUUID();
    const rollback = new Error("rollback fixtures");
    try {
      await client!.db.transaction(async (tx) => {
        await tx.insert(businesses).values([businessId, otherBusiness].map((id) => ({
          id, slug: id, name: "Embedding test", timezone: "UTC", businessType: "test", telemetryEnabled: false,
        })));
        await tx.execute(sql`set local session authorization lobbystack_worker`);
        const db = tx as unknown as Database;
        const pages = [
          { url: "https://example.invalid/one", markdown: "# Hours\n# Prices\n# Hours" },
          { url: "https://example.invalid/two", markdown: "# Hours\n# Contact" },
        ];
        let fingerprint = "v1";
        const embed = vi.fn<KnowledgeEmbeddingProvider["embed"]>(async (texts, onUsage) => {
          await onUsage?.({ provider: "test", model: fingerprint, latencyMs: 0, inputTokens: texts.length });
          return texts.map((text) => vectorFor(text, fingerprint));
        });
        const provider = { get fingerprint() { return fingerprint; }, embed };
        const dependencies = { domain: { db }, crawler: { crawl: async () => pages }, embeddings: provider };
        const crawl = (id: string) => handleJob(job(id, "knowledge.crawlWebsite"), dependencies, { isFinalAttempt: false });

        await crawl(businessId);
        expect(embed.mock.calls.map(([texts]) => texts)).toEqual([["# Hours", "# Prices"], ["# Contact"]]);
        embed.mockClear();
        await crawl(businessId);
        expect(embed).not.toHaveBeenCalled();
        pages[0]!.markdown = "# Hours\n# New prices\n# Hours";
        await crawl(businessId);
        expect(embed.mock.calls.map(([texts]) => texts)).toEqual([["# New prices"]]);
        const usage = await withBusinessTransaction(db, { businessId, actorType: "worker" }, (t) =>
          t.select({ tokens: unitEconomicsEvents.tokenUsage }).from(unitEconomicsEvents).where(eq(unitEconomicsEvents.businessId, businessId)));
        expect(usage.map(({ tokens }) => tokens?.inputTokens).sort()).toEqual([1, 1, 2]);

        embed.mockClear();
        await crawl(otherBusiness);
        expect(embed.mock.calls.map(([texts]) => texts)).toEqual([["# Hours", "# New prices"], ["# Contact"]]);

        fingerprint = "v2";
        embed.mockClear();
        await handleJob(job(businessId, "knowledge.reembedBusiness"), dependencies);
        expect(embed.mock.calls.flatMap(([texts]) => texts).sort()).toEqual(["# Contact", "# Hours", "# New prices"]);
        const stored = await withBusinessTransaction(db, { businessId, actorType: "worker" }, (t) =>
          t.select().from(knowledgeChunks).where(eq(knowledgeChunks.businessId, businessId)));
        expect(stored).toHaveLength(5);
        for (const row of stored) {
          expect(row.embeddingFingerprint).toBe("v2");
          expect(row.embedding).toEqual(vectorFor(row.content, "v2"));
        }
        throw rollback;
      });
    } catch (error) {
      if (error !== rollback) throw error;
    }
  });
});

it("deduplicates within a batch even without a fingerprint and rejects incomplete results", async () => {
  const embed = vi.fn<KnowledgeEmbeddingProvider["embed"]>().mockResolvedValueOnce([[1], [2]]).mockResolvedValueOnce([]);
  const db = {} as Database;
  await expect(embedKnowledgeChunks(db, randomUUID(), { embed }, ["a", "b", "a"])).resolves.toEqual([[1], [2], [1]]);
  expect(embed.mock.calls[0]?.[0]).toEqual(["a", "b"]);
  await expect(embedKnowledgeChunks(db, randomUUID(), { embed }, ["a"])).rejects.toThrow("incomplete result");
});
