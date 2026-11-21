import type OpenAI from "openai";

export const RATE = 24_000;
export const CALLER_MODEL = "gpt-4o-mini-tts";
export const CALLER_VOICE = "alloy";

function mulaw(sample: number): number {
  const sign = sample < 0 ? 0x80 : 0;
  const magnitude = Math.min(Math.abs(sample), 32_635) + 0x84;
  let exponent = 7;
  while (exponent > 0 && !(magnitude & (0x4000 >> (7 - exponent)))) exponent -= 1;
  return ~(sign | (exponent << 4) | ((magnitude >> (exponent + 3)) & 0x0f)) & 0xff;
}

function unmulaw(byte: number): number {
  const value = ~byte & 0xff;
  const magnitude = ((((value & 0x0f) << 3) + 0x84) << ((value & 0x70) >> 4)) - 0x84;
  return value & 0x80 ? -magnitude : magnitude;
}

// Round-trip through 8 kHz μ-law to simulate a phone line.
function phoneLine(pcm: Buffer): Buffer {
  const narrow: number[] = [];
  for (let index = 0; index + 6 <= pcm.length; index += 6) {
    narrow.push(unmulaw(mulaw((pcm.readInt16LE(index) + pcm.readInt16LE(index + 2) + pcm.readInt16LE(index + 4)) / 3)));
  }
  const out = Buffer.alloc(narrow.length * 6);
  narrow.forEach((sample, index) => {
    const next = narrow[index + 1] ?? sample;
    for (let step = 0; step < 3; step += 1) out.writeInt16LE(Math.round(sample + ((next - sample) * step) / 3), (index * 3 + step) * 2);
  });
  return out;
}

export async function callerSpeech(client: OpenAI, text: string, browser: boolean): Promise<Buffer> {
  const response = await client.audio.speech.create({ model: CALLER_MODEL, voice: CALLER_VOICE, input: text, response_format: "pcm" });
  const pcm = Buffer.from(await response.arrayBuffer());
  return browser ? pcm : phoneLine(pcm);
}

export function callerWave(pcm: Buffer): Buffer {
  const header = Buffer.alloc(44);
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + pcm.length, 4);
  header.write("WAVEfmt ", 8);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(RATE, 24);
  header.writeUInt32LE(2 * RATE, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36);
  header.writeUInt32LE(pcm.length, 40);
  return Buffer.concat([header, pcm]);
}
