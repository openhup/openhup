/**
 * Speech primitives: the two engines behind `controller.ts`.
 *
 * Browser path (default): the Web Speech API. Nothing leaves the device, and the wake word is
 * matched locally in JavaScript before any command is processed.
 *
 * Server path (opt-in): a MediaRecorder clip is POSTed to `/voice/transcribe`, and TTS audio is
 * fetched from `/voice/synthesize`. Used only when the operator configured a remote provider.
 */

import type { VoiceConfig } from '$lib/api/client';

const BASE = '/api/v1';

type SpeechRecognition = {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: (() => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
};

type SpeechRecognitionEvent = {
  results: { length: number; [index: number]: { [index: number]: { transcript: string } } };
};

function recognitionConstructor(): (new () => SpeechRecognition) | null {
  if (typeof window === 'undefined') return null;
  const w = window as unknown as Record<string, unknown>;
  return (w.SpeechRecognition ?? w.webkitSpeechRecognition) as (new () => SpeechRecognition) | null;
}

function makeRecognizer(config: VoiceConfig, continuous: boolean): SpeechRecognition | null {
  const Ctor = recognitionConstructor();
  if (!Ctor) return null;
  const recognition = new Ctor();
  recognition.lang = config.language || 'en';
  recognition.continuous = continuous;
  recognition.interimResults = continuous;
  return recognition;
}

/** One-shot recognition: speak a command, resolve with the transcript (empty on any failure). */
export function listenOnce(config: VoiceConfig): Promise<string> {
  return new Promise((resolve) => {
    const recognition = makeRecognizer(config, false);
    if (!recognition) {
      resolve('');
      return;
    }
    let transcript = '';
    recognition.onresult = (event) => {
      transcript = event.results[0]?.[0]?.transcript ?? '';
    };
    recognition.onerror = () => resolve(transcript);
    recognition.onend = () => resolve(transcript);
    recognition.start();
  });
}

/**
 * Continuous recognition that watches for the wake word, matched locally. Returns a stop handle,
 * or null when the browser cannot do it. `onWake` is called exactly once per match.
 */
export function startWakeWord(
  config: VoiceConfig,
  onWake: () => void
): (() => void) | null {
  const recognition = makeRecognizer(config, true);
  if (!recognition) return null;

  const wanted = (config.wake_word || 'hey openhup').toLowerCase();
  let fired = false;
  recognition.onresult = (event) => {
    if (fired) return;
    let heard = '';
    for (let i = 0; i < event.results.length; i += 1) {
      heard += `${event.results[i]?.[0]?.transcript ?? ''} `;
    }
    if (heard.toLowerCase().includes(wanted)) {
      fired = true;
      stop();
      onWake();
    }
  };
  recognition.onerror = () => stop();

  function stop() {
    try {
      recognition?.stop();
    } catch {
      /* already stopped */
    }
  }
  recognition.start();
  return stop;
}

/** Browser TTS. Resolves when speech ends (or immediately when unsupported). */
export function speakBrowser(text: string, voiceName?: string): Promise<void> {
  return new Promise((resolve) => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
      resolve();
      return;
    }
    const utterance = new SpeechSynthesisUtterance(text);
    if (voiceName) {
      const match = window.speechSynthesis
        .getVoices()
        .find((voice) => voice.name === voiceName || voice.lang === voiceName);
      if (match) utterance.voice = match;
    }
    utterance.onend = () => resolve();
    utterance.onerror = () => resolve();
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  });
}

/** Record a short clip and send it to the server for transcription. */
export async function recordAndTranscribe(config: VoiceConfig): Promise<string> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const mimeType = pickMimeType();
  const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  const chunks: Blob[] = [];
  recorder.ondataavailable = (event) => {
    if (event.data.size) chunks.push(event.data);
  };
  recorder.start();

  await new Promise<void>((resolve) => {
    recorder.onstop = () => resolve();
    setTimeout(() => {
      if (recorder.state !== 'inactive') recorder.stop();
    }, 4_000);
  });
  stream.getTracks().forEach((track) => track.stop());

  const blob = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' });
  const response = await fetch(`${BASE}/voice/transcribe`, {
    method: 'POST',
    headers: { 'content-type': blob.type || 'audio/webm' },
    body: blob
  });
  if (!response.ok) throw new Error(`transcribe failed: ${response.status}`);
  const body = (await response.json()) as { text: string };
  return body.text ?? '';
}

/** Fetch audio from the server and play it. */
export async function speakServer(text: string): Promise<void> {
  const response = await fetch(`${BASE}/voice/synthesize`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ text })
  });
  if (!response.ok) throw new Error(`synthesize failed: ${response.status}`);
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  await new Promise<void>((resolve) => {
    const audio = new Audio(url);
    audio.onended = () => resolve();
    audio.onerror = () => resolve();
    audio.play().catch(() => resolve());
  });
  URL.revokeObjectURL(url);
}

/** Speak text using whichever engine the config selects. */
export function speak(config: VoiceConfig, text: string): Promise<void> {
  if (config.tts_on_server) return speakServer(text);
  return speakBrowser(text, config.tts_voice);
}

/** Listen for a command using whichever engine the config selects. */
export function listen(config: VoiceConfig): Promise<string> {
  if (config.stt_on_server) return recordAndTranscribe(config);
  return listenOnce(config);
}

function pickMimeType(): string | undefined {
  const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg'];
  if (typeof MediaRecorder === 'undefined') return undefined;
  return candidates.find((type) => MediaRecorder.isTypeSupported(type));
}
