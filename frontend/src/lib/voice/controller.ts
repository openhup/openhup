/**
 * The voice state machine.
 *
 * States: off → idle → (wake-listening | command-listening) → speaking → back to idle/wake-listening.
 *
 * The wake word is matched *locally* in the browser (see `speech.startWakeWord`); nothing is sent
 * to OpenHup or anywhere else until the word is heard, and then only a single recognised command is
 * processed. `pushToTalk` is the same command loop without the wake word, for browsers that cannot
 * do continuous recognition or for people who prefer a button.
 */

import { get, writable } from 'svelte/store';
import { api } from '$lib/api/client';
import { voiceConfig, voicePreferences } from './settings';
import { listen, startWakeWord } from './speech';
import { runCommand } from './commands';

export type VoiceState = 'off' | 'idle' | 'wake-listening' | 'command-listening' | 'speaking';

export const voiceState = writable<VoiceState>('off');

let wakeStop: (() => void) | null = null;
let busy = false;

/** Fetch server truth once; the controller refuses to do anything until this succeeds. */
export async function initVoice(): Promise<void> {
  try {
    voiceConfig.set(await api.voiceConfig());
  } catch {
    voiceConfig.set(null);
  }
  voiceState.set(get(voiceConfig)?.enabled ? 'idle' : 'off');
}

export function isWakeArmed(): boolean {
  return wakeStop !== null;
}

/** Begin listening for the wake word. Requires mic permission, so call from a user gesture. */
export function armWakeWord(): void {
  const config = get(voiceConfig);
  const preferences = get(voicePreferences);
  if (!config || !preferences.enabled || wakeStop) return;

  const stop = startWakeWord(config, () => {
    void handleCommand();
  });
  if (!stop) {
    voiceState.set('idle'); // browser cannot do continuous recognition
    return;
  }
  wakeStop = stop;
  voiceState.set('wake-listening');
}

/** Stop the wake word loop. */
export function disarm(): void {
  wakeStop?.();
  wakeStop = null;
  if (get(voiceState) === 'wake-listening') voiceState.set('idle');
}

/** One button-press command: listen once, then act and speak. No wake word needed. */
export async function pushToTalk(): Promise<void> {
  await handleCommand();
}

async function handleCommand(): Promise<void> {
  if (busy) return;
  const config = get(voiceConfig);
  if (!config) return;
  busy = true;
  try {
    voiceState.set('command-listening');
    const text = await listen(config);
    if (!text.trim()) {
      return; // nothing recognised (or no mic); re-arm in the finally block
    }
    voiceState.set('speaking');
    await runCommand(text);
  } finally {
    busy = false;
    const preferences = get(voicePreferences);
    voiceState.set(
      preferences.enabled && preferences.wakeWordEnabled && wakeStop ? 'wake-listening' : 'idle'
    );
  }
}

/** A soft two-tone chime so "wake word heard, go ahead" is audible without looking. */
export function chime(): void {
  if (typeof window === 'undefined' || !window.AudioContext) return;
  const context = new AudioContext();
  [880, 1320].forEach((frequency, index) => {
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    oscillator.frequency.value = frequency;
    const start = context.currentTime + index * 0.12;
    gain.gain.setValueAtTime(0.0001, start);
    gain.gain.exponentialRampToValueAtTime(0.08, start + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, start + 0.12);
    oscillator.connect(gain).connect(context.destination);
    oscillator.start(start);
    oscillator.stop(start + 0.13);
  });
}
