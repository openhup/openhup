<!--
  Habits: goals and the weekly report. No shame metrics, no "missed"/"overdue" anything.
-->

<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '$lib/api/client';

  let goals = $state<Record<string, unknown>[]>([]);
  let report = $state<Record<string, unknown> | null>(null);

  onMount(async () => {
    [goals, report] = await Promise.all([api.goals(), api.weeklyReport()]);
  });
</script>

<svelte:head><title>Habits · OpenHup</title></svelte:head>

{#if report}
  <section class="report">
    <h1>This week</h1>
    <p>{report.plain_summary as string}</p>
  </section>
{/if}

<h1>Goals</h1>
<ul class="goals">
  {#each goals as goal (goal.id as string)}
    <li>
      <h2>{goal.label as string}</h2>
      <p class="muted">{goal.metric as string}</p>
    </li>
  {/each}
</ul>

<style>
  section,
  ul {
    list-style: none;
    padding: 0;
    margin: 0 auto 2rem;
    max-width: 40rem;
  }
  .report {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 0.75rem;
    padding: 1rem 1.25rem;
  }
  h1 {
    font-size: 1.2rem;
    margin: 1.5rem auto 0.75rem;
  }
  .report h1 {
    margin-top: 0;
  }
  .goals {
    display: grid;
    gap: 1rem;
  }
  .goals li {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 0.75rem;
    padding: 1rem 1.25rem;
  }
  h2 {
    margin: 0 0 0.25rem;
    font-size: 1.05rem;
  }
  .muted {
    color: var(--muted);
    margin: 0;
  }
</style>
