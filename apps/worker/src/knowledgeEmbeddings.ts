import { and, eq, isNotNull, sql } from "drizzle-orm";

import { knowledgeChunks, withBusinessTransaction, type Database } from "@lobbystack/db";
import { hashContent, type DurableAiUsage } from "@lobbystack/domain";

export type KnowledgeEmbeddingProvider = {
  fingerprint?: string;
  embed(values: string[], onUsage?: (usage: DurableAiUsage) => Promise<void> | void): Promise<number[][]>;
};

export async function embedKnowledgeChunks(
  db: Database,
  businessId: string,
  provider: KnowledgeEmbeddingProvider,
  chunks: string[],
  onUsage?: (usage: DurableAiUsage) => Promise<void> | void,
): Promise<number[][]> {
  const unique = [...new Set(chunks)];
  const vectors = new Map<string, number[]>();
  const fingerprint = provider.fingerprint;
  if (fingerprint && unique.length) {
    const hashes = unique.map(hashContent);
    const stored = await withBusinessTransaction(db, { businessId, actorType: "worker" }, (tx) =>
      tx.selectDistinctOn([knowledgeChunks.contentHash], { content: knowledgeChunks.content, embedding: knowledgeChunks.embedding })
        .from(knowledgeChunks)
        .where(and(
          eq(knowledgeChunks.businessId, businessId),
          eq(knowledgeChunks.embeddingFingerprint, fingerprint),
          eq(knowledgeChunks.embeddingStatus, "completed"),
          isNotNull(knowledgeChunks.embedding),
          sql`${knowledgeChunks.contentHash} = ANY(${sql.param(hashes)}::varchar[])`,
        )),
    );
    for (const row of stored) {
      if (row.embedding) vectors.set(row.content, row.embedding);
    }
  }

  const missing = unique.filter((text) => !vectors.has(text));
  if (missing.length) {
    const fresh = await provider.embed(missing, onUsage);
    if (fresh.length !== missing.length) throw new Error("Knowledge embedding provider returned an incomplete result.");
    missing.forEach((text, i) => vectors.set(text, fresh[i]!));
  }
  return chunks.map((text) => vectors.get(text)!);
}
