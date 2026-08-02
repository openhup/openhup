/**
 * Turn a transcript into an action.
 *
 * The server decides (deterministically) what a sentence means; this module just ships it the text
 * and then does the client-side parts: speak the reply back, and follow a navigation target. All
 * the "everything" intents funnel through here.
 */

import { get } from 'svelte/store';
import { api, type VoiceCommandResult } from '$lib/api/client';
import { voiceConfig, voicePreferences } from './settings';
import { speak } from './speech';

/**
 * Run one spoken command: send it to the server, speak the reply, and navigate when asked.
 * Returns the structured result so callers (and tests) can inspect it.
 */
export async function runCommand(text: string): Promise<VoiceCommandResult | null> {
  const config = get(voiceConfig);
  if (!config || !text.trim()) return null;

  let result: VoiceCommandResult;
  try {
    // Declared per-device identity: who this device belongs to, passed along as a hint for
    // task targeting. The person told the device; the server never guesses.
    result = await api.voiceCommand(text, get(voicePreferences).whoAmI);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'something went wrong';
    await speak(config, `Sorry, I couldn't do that: ${message}`);
    return null;
  }

  if (result.reply) await speak(config, result.reply);
  if (result.intent === 'navigate' && result.target) {
    window.location.assign(result.target);
  }
  return result;
}
