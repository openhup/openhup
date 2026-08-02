<script lang="ts">
  import { get } from 'svelte/store';
  import { armWakeWord, disarm, pushToTalk, voiceState } from './controller';
  import { browserSpeechSupported, voiceConfig, voicePreferences } from './settings';

  const usable = $derived(
    !!$voiceConfig?.enabled &&
      ($voiceConfig.stt_on_server || $voiceConfig.tts_on_server || browserSpeechSupported())
  );

  const status = $derived.by(() => {
    switch ($voiceState) {
      case 'wake-listening':
        return `Listening for "${$voiceConfig?.wake_word ?? 'hey openhup'}"`;
      case 'command-listening':
        return 'Listening…';
      case 'speaking':
        return 'Speaking…';
      default:
        return 'Talk';
    }
  });

  const armed = $derived($voiceState === 'wake-listening');

  async function onClick() {
    const state = get(voiceState);
    if (state === 'wake-listening') {
      disarm();
      return;
    }
    if (state === 'command-listening' || state === 'speaking') return;

    // Talk now. After the first talk grants mic permission, stay armed for the wake word if asked.
    await pushToTalk();
    const preferences = get(voicePreferences);
    if (preferences.enabled && preferences.wakeWordEnabled) armWakeWord();
  }
</script>

<button
  class="voice"
  class:armed
  onclick={onClick}
  disabled={!usable}
  aria-pressed={armed}
  title={status}
>
  <svg class="mic" viewBox="0 0 24 24" aria-hidden="true">
    <rect x="9" y="2.5" width="6" height="11" rx="3" />
    <path d="M5.5 11a6.5 6.5 0 0 0 13 0" />
    <line x1="12" y1="17.5" x2="12" y2="21.5" />
  </svg>
  <span class="label">{status}</span>
</button>

<style>
  .voice {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.45rem 0.8rem;
    border-radius: 999px;
    border: 1px solid var(--line);
    background: var(--surface);
    color: var(--text);
  }
  .voice.armed {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 25%, transparent);
  }
  .mic {
    width: 1rem;
    height: 1rem;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.8;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
  .label {
    font-size: 0.85rem;
  }
</style>
