import type OpenAI from "openai";
import { describe, expect, it, vi } from "vitest";

import { callerSpeech, callerWave } from "./caller-audio";

describe("scripted caller audio", () => {
  it("uses the full-band PCM for browser calls and applies the phone-line filter for phone calls", async () => {
    const pcm = new Int16Array([0, 4_000, 0, 0, -4_000, 0]);
    const create = vi.fn().mockResolvedValue({ arrayBuffer: async () => pcm.buffer });
    const client = { audio: { speech: { create } } } as unknown as OpenAI;

    const browser = await callerSpeech(client, "Hello", true);
    const phone = await callerSpeech(client, "Hello", false);

    expect(create).toHaveBeenCalledWith({ model: "gpt-4o-mini-tts", voice: "alloy", input: "Hello", response_format: "pcm" });
    expect(browser).toEqual(Buffer.from(pcm.buffer));
    expect(phone).toHaveLength(browser.length);
    expect(phone).not.toEqual(browser);
  });

  it("writes a mono 24 kHz PCM WAV without changing the samples", () => {
    const pcm = Buffer.from([0xff, 0x7f, 0x00, 0x80]);
    const wav = callerWave(pcm);

    expect(wav.toString("ascii", 0, 4)).toBe("RIFF");
    expect(wav.readUInt32LE(4)).toBe(wav.length - 8);
    expect(wav.toString("ascii", 8, 16)).toBe("WAVEfmt ");
    expect(wav.readUInt16LE(20)).toBe(1);
    expect(wav.readUInt16LE(22)).toBe(1);
    expect(wav.readUInt32LE(24)).toBe(24_000);
    expect(wav.readUInt16LE(34)).toBe(16);
    expect(wav.toString("ascii", 36, 40)).toBe("data");
    expect(wav.readUInt32LE(40)).toBe(pcm.length);
    expect(wav.subarray(44)).toEqual(pcm);
  });
});
