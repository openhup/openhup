<!--
  Settings: personality picker with live preview, and notification channel status.
-->

<script lang="ts">
  import { onMount } from 'svelte';
  import { get } from 'svelte/store';
  import { api, type Member, type MemoryFact, type MemoryPattern, type PersonalityDraw, type WinMilestone } from '$lib/api/client';
  import { armWakeWord, disarm } from '$lib/voice/controller';
  import {
    browserSpeechSupported,
    voiceConfig,
    voicePreferences,
    type VoicePreferences
  } from '$lib/voice/settings';

  let personalities = $state<Record<string, unknown>[]>([]);
  let channels = $state<Record<string, unknown>[]>([]);
  let preview = $state<Record<string, string> | null>(null);
  let previewId = $state<string | null>(null);

  let facts = $state<MemoryFact[]>([]);
  let newFact = $state('');
  let newFactTopic = $state('');
  let patterns = $state<MemoryPattern[]>([]);
  let patternsNote = $state('');

  let draw = $state<PersonalityDraw | null>(null);
  let wins = $state<WinMilestone[]>([]);
  let members = $state<Member[]>([]);
  let identityEnabled = $state(false);

  onMount(async () => {
    personalities = await api.personalities();
    channels = await api.systemInfo().then((info) => {
      const notify = info.notify as Record<string, unknown> | undefined;
      return (notify?.channels as Record<string, unknown>[]) ?? [];
    });
    facts = await api.memoryFacts();
    await loadPatterns();
    await loadDraw();
    await loadWins();
    await loadMembers();
  });

  async function loadMembers() {
    const result = await api.members();
    members = result.members;
    identityEnabled = result.enabled;
  }

  async function deleteMember(id: string) {
    await api.deleteMember(id);
    await loadMembers();
  }

  function setWhoAmI(id: string | null) {
    voicePreferences.update((preferences) => ({ ...preferences, whoAmI: id }));
  }

  async function loadWins() {
    wins = (await api.wins()).wins;
  }

  async function deleteWin(id: string) {
    await api.deleteWin(id);
    await loadWins();
  }

  async function loadDraw() {
    draw = await api.personalityDraw();
  }

  async function drawVoice() {
    await api.drawPersonality();
    await loadDraw();
  }

  async function clearDraw() {
    await api.clearPersonalityDraw();
    await loadDraw();
  }

  async function loadPatterns() {
    const result = await api.memoryPatterns();
    patterns = result.patterns;
    patternsNote = result.note;
  }

  async function dismissPattern(id: string) {
    await api.dismissMemoryPattern(id);
    await loadPatterns();
  }

  async function addFact() {
    const fact = newFact.trim();
    if (!fact) return;
    await api.addMemoryFact(fact, newFactTopic);
    newFact = '';
    newFactTopic = '';
    facts = await api.memoryFacts();
  }

  async function deleteFact(id: string) {
    await api.deleteMemoryFact(id);
    facts = await api.memoryFacts();
  }

  async function showPreview(id: string) {
    previewId = id;
    preview = null;
    preview = await api.previewPersonality(id);
  }

  function toggleVoice(key: keyof VoicePreferences) {
    voicePreferences.update((preferences) => ({ ...preferences, [key]: !preferences[key] }));
    // Re-evaluate whether the wake word should be armed after any of these change.
    const preferences = get(voicePreferences);
    if (preferences.enabled && preferences.wakeWordEnabled) armWakeWord();
    else disarm();
  }

  function providerLabel(value: string): string {
    return value === 'browser' ? 'this browser (on-device)' : value;
  }
</script>

<svelte:head><title>Settings · OpenHup</title></svelte:head>

<h1>Personality</h1>
<div class="gamble">
  <h2>The mystery voice</h2>
  <p class="muted">
    The assistant's personality is one of five voices (friendly, shy, sassy, sarcastic, angry),
    chosen at setup - either picked by you, or drawn as a gamble. It is never announced; you
    discover it by living with it. The docs and config.yaml are the only places the answer is
    written down.
  </p>
  {#if draw?.drawn}
    <p class="gamble-answer">A mystery voice is active. It is not going to tell you which.</p>
    <div class="row actions">
      <button onclick={drawVoice}>Draw a new voice</button>
      <button onclick={clearDraw} class="danger">Use the configured default instead</button>
    </div>
  {:else}
    <p class="muted">No gamble has been drawn; the configured default voice speaks.</p>
    <div class="row actions">
      <button onclick={drawVoice}>Draw a mystery voice</button>
    </div>
  {/if}
</div>
<ul class="list">
  {#each personalities as p (p.id as string)}
    <li>
      <div class="row">
        <div>
          <h2>{p.display_name as string}</h2>
          <p class="muted">{p.description as string}</p>
        </div>
        <button onclick={() => showPreview(p.id as string)}>Preview</button>
      </div>
      {#if previewId === p.id && preview}
        <div class="preview">
          <p>{preview.task}</p>
          <p class="muted">{preview.note}</p>
        </div>
      {/if}
    </li>
  {/each}
</ul>

<h1>Notifications</h1>
<ul class="list">
  {#each channels as channel (channel.id as string)}
    <li>
      <h2>{channel.id as string}</h2>
      <p class="muted">{channel.type as string}</p>
    </li>
  {/each}
</ul>

<h1>Voice</h1>
{#if $voiceConfig}
  <ul class="list">
    <li>
      <div class="row">
        <div>
          <h2>Enable voice on this device</h2>
          <p class="muted">
            Recognition: {providerLabel($voiceConfig.stt_provider)} · Speech: {providerLabel($voiceConfig.tts_provider)}
          </p>
          {#if !browserSpeechSupported() && !$voiceConfig.stt_on_server && !$voiceConfig.tts_on_server}
            <p class="warn">This browser has no speech support, and no server speech provider is configured.</p>
          {/if}
        </div>
        <label class="toggle">
          <input type="checkbox" checked={$voicePreferences.enabled} onchange={() => toggleVoice('enabled')} />
          <span class="sr-only">Enable voice</span>
        </label>
      </div>
    </li>
    <li>
      <div class="row">
        <div>
          <h2>Listen for the wake word</h2>
          <p class="muted">
            Matched locally in the browser before anything is processed: "{$voiceConfig.wake_word}".
          </p>
        </div>
        <label class="toggle">
          <input type="checkbox" checked={$voicePreferences.wakeWordEnabled} onchange={() => toggleVoice('wakeWordEnabled')} />
          <span class="sr-only">Wake word</span>
        </label>
      </div>
    </li>
    <li>
      <div class="row">
        <div>
          <h2>Speak alerts aloud</h2>
          <p class="muted">Safety alerts are spoken as they arrive.</p>
        </div>
        <label class="toggle">
          <input type="checkbox" checked={$voicePreferences.speakAlerts} onchange={() => toggleVoice('speakAlerts')} />
          <span class="sr-only">Speak alerts</span>
        </label>
      </div>
    </li>
    <li>
      <div class="row">
        <div>
          <h2>Speak new tasks (nudges)</h2>
          <p class="muted">A gentle spoken reminder when a new task appears.</p>
        </div>
        <label class="toggle">
          <input type="checkbox" checked={$voicePreferences.speakTasks} onchange={() => toggleVoice('speakTasks')} />
          <span class="sr-only">Speak tasks</span>
        </label>
      </div>
    </li>
    <li>
      <div class="row">
        <div>
          <h2>Speak noticed patterns</h2>
          <p class="muted">
            When a learned pattern says something is about due, e.g. "the trash usually fills
            around now". Obeys each skill's own quiet hours, cooldown, and daily cap, and is
            always forward-looking.
          </p>
        </div>
        <label class="toggle">
          <input type="checkbox" checked={$voicePreferences.speakPatterns} onchange={() => toggleVoice('speakPatterns')} />
          <span class="sr-only">Speak patterns</span>
        </label>
      </div>
    </li>
    <li>
      <div class="row">
        <div>
          <h2>Speak wins</h2>
          <p class="muted">
            When a place stays clear for whole days - "the counter stayed clear 3 days".
            Forward-facing, celebrated at most once per milestone, and never inside quiet hours.
          </p>
        </div>
        <label class="toggle">
          <input type="checkbox" checked={$voicePreferences.speakWins} onchange={() => toggleVoice('speakWins')} />
          <span class="sr-only">Speak wins</span>
        </label>
      </div>
    </li>
  </ul>
{:else}
  <p class="muted">Voice is disabled in the server configuration.</p>
{/if}

<h1>Who lives here</h1>
{#if identityEnabled}
  {#if members.length > 0}
    <ul class="list">
      {#each members as member (member.id)}
        <li>
          <div class="row">
            <div>
              <h2>{member.name}</h2>
              <p class="muted">
                Enrolled {new Date(member.enrolled_at).toLocaleDateString()}
                {#if member.last_seen_at}
                  · last seen {new Date(member.last_seen_at).toLocaleDateString()}
                {/if}
              </p>
            </div>
            <div class="actions">
              <label class="toggle">
                <input
                  type="radio"
                  name="whoami"
                  checked={$voicePreferences.whoAmI === member.id}
                  onchange={() => setWhoAmI(member.id)}
                />
                <span class="sr-only">This device is {member.name}</span>
              </label>
              <button onclick={() => deleteMember(member.id)} class="danger" aria-label="Forget {member.name}">
                Forget
              </button>
            </div>
          </div>
        </li>
      {/each}
    </ul>
    <p class="muted note">
      The radio marks which member this device belongs to, so the assistant can address the right
      person. It is something you tell the device — it is never inferred from your face or voice.
      Forgetting a member deletes their face data and their presence history everywhere.
    </p>
  {:else}
    <p class="muted note">
      Nobody has been remembered yet. The first time the camera sees an unfamiliar face, the
      assistant will ask — in its own voice — whether it may remember them. A "no" is final for
      the day and nothing about the face is stored.
    </p>
  {/if}
  <div class="row actions">
    <button onclick={() => setWhoAmI(null)} class="danger">This device belongs to nobody</button>
  </div>
{:else}
  <p class="muted note">
    Identity is disabled in the server configuration. The assistant treats everyone in the house
    as one household.
  </p>
{/if}

<h1>What the assistant remembers</h1>
{#if facts.length > 0}
  <ul class="list">
    {#each facts as fact (fact.id)}
      <li>
        <div class="row">
          <div>
            <p>{fact.fact}</p>
            {#if fact.topic}
              <p class="muted">topic: {fact.topic}</p>
            {/if}
          </div>
          <button onclick={() => deleteFact(fact.id)} class="danger" aria-label="Forget: {fact.fact}">
            Forget
          </button>
        </div>
      </li>
    {/each}
  </ul>
{:else}
  <p class="muted">Nothing yet. Say "remember that …" to the assistant, or add a fact below.</p>
{/if}
<div class="add-fact">
  <input
    type="text"
    placeholder="e.g. bin day is Tuesday"
    bind:value={newFact}
    onkeydown={(event) => event.key === 'Enter' && addFact()}
  />
  <input type="text" placeholder="topic (optional)" bind:value={newFactTopic} />
  <button onclick={addFact}>Remember</button>
</div>
<p class="muted note">
  Facts are stored on this device only. Nothing leaves your network except a relevant snippet inside
  a prompt sent to your LLM provider, which is gated and logged like every other call. Anything here
  can be deleted, and a forgotten fact is gone.
</p>

<h1>Wins the assistant has noticed</h1>
{#if wins.length > 0}
  <ul class="list">
    {#each wins as win (win.id)}
      <li>
        <div class="row">
          <div>
            <p>{win.summary}</p>
            <p class="muted">
              {win.kind === 'record_clear_days' ? 'longest clear stretch in 90 days' : 'milestone'}
              {#if win.spoken}· spoken aloud{/if}
            </p>
          </div>
          <button onclick={() => deleteWin(win.id)} class="danger" aria-label="Forget: {win.summary}">
            Forget
          </button>
        </div>
      </li>
    {/each}
  </ul>
{:else}
  <p class="muted note">
    Nothing yet. When a place stays clear for whole days - a milestone or the longest stretch in
    90 days - it is celebrated here and, if you have voice on, spoken aloud. Wins are forward-
    facing only: how long things have been good, never how long anything was left.
  </p>
{/if}

<h1>Patterns the assistant has noticed</h1>
{#if patterns.length > 0}
  <ul class="list">
    {#each patterns as pattern (pattern.id)}
      <li>
        <div class="row">
          <div>
            <p>{pattern.summary}</p>
            <p class="muted">
              confidence {Math.round(pattern.confidence * 100)}%
              {#if pattern.evidence.n_episodes}· from {pattern.evidence.n_episodes} episodes{/if}
              {#if !pattern.nudge_eligible}· not confident enough to speak{/if}
            </p>
          </div>
          <button onclick={() => dismissPattern(pattern.id)} class="danger" aria-label="Dismiss: {pattern.summary}">
            Not useful
          </button>
        </div>
      </li>
    {/each}
  </ul>
  {#if patternsNote}
    <p class="muted note">{patternsNote}</p>
  {/if}
{:else}
  <p class="muted note">
    Nothing yet. Patterns come from the episodes the engine already records — a couple of weeks of
    history and the assistant starts noticing things like "the trash usually fills about every 3
    days". Everything here is dismissable, and a dismissed pattern is never learned again.
  </p>
{/if}

<style>
  .list {
    list-style: none;
    padding: 0;
    margin: 0 auto 2rem;
    max-width: 40rem;
    display: grid;
    gap: 1rem;
  }
  .list li {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 0.75rem;
    padding: 1rem 1.25rem;
  }
  h1 {
    font-size: 1.2rem;
    margin: 1.5rem auto 0.75rem;
    max-width: 40rem;
  }
  h2 {
    margin: 0 0 0.25rem;
    font-size: 1.05rem;
  }
  button {
    padding: 0.5rem 0.9rem;
    border-radius: 0.6rem;
    border: 1px solid var(--line);
    background: var(--surface);
  }
  .preview {
    margin-top: 0.75rem;
    padding: 0.75rem;
    border-left: 3px solid var(--accent);
    background: var(--bg);
    font-size: 0.9rem;
  }
  .gamble {
    max-width: 40rem;
    margin: 0 auto 1rem;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 0.75rem;
    padding: 1rem 1.25rem;
  }
  .gamble-answer {
    margin: 0.75rem 0 0;
    font-size: 1.05rem;
  }
  .actions {
    margin-top: 0.75rem;
    flex-wrap: wrap;
  }
  .muted {
    color: var(--muted);
    margin: 0.25rem 0 0;
  }
  .row {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: flex-start;
  }
  .toggle input {
    width: 1.3rem;
    height: 1.3rem;
  }
  .warn {
    color: var(--danger);
    margin: 0.25rem 0 0;
    font-size: 0.85rem;
  }
  .danger {
    color: var(--danger);
    border-color: var(--danger);
    flex-shrink: 0;
  }
  .add-fact {
    max-width: 40rem;
    margin: 0 auto 0.5rem;
    display: flex;
    gap: 0.5rem;
  }
  .add-fact input {
    flex: 1;
    padding: 0.5rem 0.7rem;
    border-radius: 0.6rem;
    border: 1px solid var(--line);
    background: var(--surface);
    color: var(--text);
  }
  .add-fact input + input {
    flex: 0 0 12rem;
  }
  .note {
    max-width: 40rem;
    margin: 0 auto 2rem;
    font-size: 0.85rem;
  }
</style>
