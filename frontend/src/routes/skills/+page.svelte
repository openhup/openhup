<!--
  Skills — list, enable toggle, and the simulate panel.

  The simulate panel is deliberately placed above the enable toggle: dry-running a skill against
  real history before arming it is what stops a badly-tuned threshold from becoming a bad week.
-->

<script lang="ts">
  import { onMount } from 'svelte';
  import { api, ApiError, type SimulationResult, type SkillSummary } from '$lib/api/client';

  let skills = $state<SkillSummary[]>([]);
  let busy = $state<Record<string, boolean>>({});
  let simulating = $state<Record<string, SimulationResult | null>>({});
  let simulateError = $state<Record<string, string>>({});

  async function refresh() {
    skills = await api.skills();
  }

  onMount(refresh);

  async function toggle(skill: SkillSummary) {
    busy[skill.id] = true;
    try {
      await api.updateSkill(skill.id, { enabled: !skill.enabled });
      await refresh();
    } finally {
      busy[skill.id] = false;
    }
  }

  async function simulate(skill: SkillSummary) {
    simulating[skill.id] = null;
    simulateError[skill.id] = '';
    try {
      simulating[skill.id] = await api.simulate(skill.id);
    } catch (err) {
      simulateError[skill.id] = err instanceof ApiError ? err.message : String(err);
    }
  }
</script>

<svelte:head><title>Skills · OpenHup</title></svelte:head>

<ul class="skills">
  {#each skills as skill (skill.id)}
    <li>
      <div class="row">
        <div>
          <h2>{skill.id}</h2>
          <p class="explain">{skill.explanation}</p>
          {#if skill.warnings.length}
            <ul class="warnings">
              {#each skill.warnings as warning}<li>{warning}</li>{/each}
            </ul>
          {/if}
        </div>
        <label class="toggle">
          <input
            type="checkbox"
            checked={skill.enabled}
            disabled={busy[skill.id]}
            onchange={() => toggle(skill)}
          />
          <span class="sr-only">Enabled</span>
        </label>
      </div>

      <button onclick={() => simulate(skill)}>Simulate against history</button>

      {#if simulating[skill.id] !== undefined}
        <div class="simulation">
          {#if simulating[skill.id]}
            <p><strong>{simulating[skill.id]!.verdict}</strong></p>
            {#if simulating[skill.id]!.advice.length}
              <ul>
                {#each simulating[skill.id]!.advice as line}<li>{line}</li>{/each}
              </ul>
            {/if}
          {:else if simulateError[skill.id]}
            <p class="error">{simulateError[skill.id]}</p>
          {:else}
            <p class="muted">Simulating…</p>
          {/if}
        </div>
      {/if}
    </li>
  {/each}
</ul>

<style>
  .skills {
    list-style: none;
    padding: 0;
    margin: 0 auto;
    max-width: 40rem;
    display: grid;
    gap: 1rem;
  }
  .skills li {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 0.75rem;
    padding: 1rem 1.25rem;
  }
  .row {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
  }
  h2 {
    margin: 0 0 0.25rem;
    font-size: 1.15rem;
  }
  .explain {
    margin: 0;
  }
  .warnings {
    margin: 0.5rem 0 0;
    padding-left: 1.2rem;
    font-size: 0.85rem;
    color: var(--muted);
  }
  button {
    padding: 0.6rem 1rem;
    border-radius: 0.6rem;
    border: 1px solid var(--line);
    background: var(--surface);
    margin-top: 0.75rem;
  }
  .toggle input {
    width: 1.3rem;
    height: 1.3rem;
  }
  .simulation {
    margin-top: 0.75rem;
    padding: 0.75rem;
    border-left: 3px solid var(--accent);
    background: var(--bg);
    font-size: 0.9rem;
  }
  .muted {
    color: var(--muted);
    margin: 0;
  }
  .error {
    color: var(--danger);
  }
</style>
