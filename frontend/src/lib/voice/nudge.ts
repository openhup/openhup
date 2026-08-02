/**
 * Spoken nudges and announcements.
 *
 * The text-to-speech half of "voice". Subscribes to the live event stream and speaks the things a
 * voice assistant should say without being asked: safety alerts (always), and new tasks as a gentle
 * nudge. Both are toggled in settings, and nothing speaks while a spoken command is in flight.
 */

import { get } from 'svelte/store';
import { onEvent } from '$lib/stores/events';
import { voiceState } from './controller';
import { voiceConfig, voicePreferences } from './settings';
import { speak } from './speech';

export function startNudges(): () => void {
  const stopAlerts = onEvent('alert.raised', (event) => {
    void maybeSpeak('speakAlerts', String(event.payload.text ?? ''));
  });
  const stopTasks = onEvent('task.created', (event) => {
    void maybeSpeak('speakTasks', String(event.payload.text ?? ''));
  });
  // Learned patterns speaking unprompted: "the trash usually fills around now".
  // Forward-looking only, governed server-side by each subject skill's own quiet hours,
  // cooldown, and daily cap.
  const stopPatterns = onEvent('system.pattern_nudge', (event) => {
    void maybeSpeak('speakPatterns', String(event.payload.text ?? ''));
  });
  // Wins: the assistant noticing progress, not just problems ("the counter stayed clear 3 days").
  // Forward-facing only, celebrated at most once per milestone server-side.
  const stopWins = onEvent('system.win_note', (event) => {
    void maybeSpeak('speakWins', String(event.payload.text ?? ''));
  });
  return () => {
    stopAlerts();
    stopTasks();
    stopPatterns();
    stopWins();
  };
}

async function maybeSpeak(
  preferenceKey: 'speakAlerts' | 'speakTasks' | 'speakPatterns' | 'speakWins',
  text: string
): Promise<void> {
  if (!text) return;
  const config = get(voiceConfig);
  const preferences = get(voicePreferences);
  if (!config?.enabled || !preferences.enabled || !preferences[preferenceKey]) return;
  if (get(voiceState) === 'command-listening' || get(voiceState) === 'speaking') return;
  await speak(config, text);
}
