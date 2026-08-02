/**
 * Voice settings.
 *
 * Two kinds of state, kept separate on purpose:
 *  - `voiceConfig` is server truth (providers, wake word, whether audio leaves the device), fetched
 *    once from `/voice/config`.
 *  - `voicePreferences` is per-device user choice (is the mic armed, should we speak alerts/tasks),
 *    persisted in localStorage because it belongs to this browser, not the household.
 */

import { get, writable } from 'svelte/store';
import type { VoiceConfig } from '$lib/api/client';

export interface VoicePreferences {
  /** Master switch on this device. Mic access still requires the browser's own permission. */
  enabled: boolean;
  /** Listen for the wake word (matched locally) so no button press is needed. */
  wakeWordEnabled: boolean;
  /** Speak alerts as they arrive. Safety first. */
  speakAlerts: boolean;
  /** Speak new tasks (the spoken "nudge"). */
  speakTasks: boolean;
  /** Speak learned-pattern nudges ("the trash usually fills around now"). */
  speakPatterns: boolean;
  /** Speak wins ("the counter stayed clear 3 days"). On by default: progress noticed is the
   *  caring half of the voice, and it is quiet-hours-gated server-side like every nudge. */
  speakWins: boolean;
  /** Per-device declared identity (ADR-016): the member id this device belongs to. Declared by
   *  the person in Settings, never inferred - the server uses it to target tasks and nudges. */
  whoAmI: string | null;
}

const KEY = 'openhup.voice';
const DEFAULTS: VoicePreferences = {
  enabled: false,
  wakeWordEnabled: true,
  speakAlerts: true,
  speakTasks: true,
  speakPatterns: true,
  speakWins: true,
  whoAmI: null
};

function loadPreferences(): VoicePreferences {
  if (typeof localStorage === 'undefined') return DEFAULTS;
  try {
    return { ...DEFAULTS, ...(JSON.parse(localStorage.getItem(KEY) ?? '{}') as Partial<VoicePreferences>) };
  } catch {
    return DEFAULTS;
  }
}

export const voiceConfig = writable<VoiceConfig | null>(null);
export const voicePreferences = writable<VoicePreferences>(loadPreferences());

voicePreferences.subscribe((preferences) => {
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem(KEY, JSON.stringify(preferences));
  }
});

/** Whether the browser exposes the Web Speech API at all (Chrome/Edge/Safari). */
export function browserSpeechSupported(): boolean {
  if (typeof window === 'undefined') return false;
  const w = window as unknown as Record<string, unknown>;
  return Boolean(w.SpeechRecognition || w.webkitSpeechRecognition) && 'speechSynthesis' in window;
}

export function voiceIsUsable(): boolean {
  const config = get(voiceConfig);
  const preferences = get(voicePreferences);
  // Server-side STT/TTS does not need the browser speech API; browser-side does.
  if (!config || !config.enabled) return false;
  if (config.stt_on_server || config.tts_on_server) return true;
  return browserSpeechSupported();
}
