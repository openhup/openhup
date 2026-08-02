<!--
  All tasks — backlog mode, for people who prefer seeing everything over single-task focus.

  Deliberately NOT here: any count of outstanding work, and any red for "not done".
-->

<script lang="ts">
  import { onMount } from 'svelte';
  import { api, type Task } from '$lib/api/client';
  import { connect, liveResource } from '$lib/stores/events';

  let tasks = $state<Task[]>([]);
  let loading = $state(true);
  let busy = $state<Record<string, boolean>>({});

  async function refresh(): Promise<Task[]> {
    tasks = await api.tasks('open');
    return tasks;
  }

  onMount(() => {
    loading = false;
    refresh();
    const stop = connect('task');
    const sub = liveResource(refresh, ['task.'], []).subscribe(() => {});
    return () => {
      sub();
      stop();
    };
  });

  async function act(id: string, fn: () => Promise<unknown>) {
    busy[id] = true;
    try {
      await fn();
      await refresh();
    } finally {
      busy[id] = false;
    }
  }
</script>

<svelte:head><title>Tasks · OpenHup</title></svelte:head>

{#if loading}
  <p class="muted">Loading…</p>
{:else if tasks.length === 0}
  <section class="empty">
    <h1>Nothing right now.</h1>
    <p class="muted">OpenHup is watching, and will say something when there is something worth saying.</p>
  </section>
{:else}
  <ul class="tasks">
    {#each tasks as task (task.id)}
      <li>
        <h2>{task.current_text}</h2>
        <p class="muted">{task.anchor_label}</p>
        <div class="actions">
          <button class="primary" disabled={busy[task.id]} onclick={() => act(task.id, () => api.completeTask(task.id))}>
            Done
          </button>
          <button disabled={busy[task.id]} onclick={() => act(task.id, () => api.snoozeTask(task.id, 60))}>
            Later
          </button>
        </div>
      </li>
    {/each}
  </ul>
{/if}

<style>
  .tasks {
    list-style: none;
    padding: 0;
    margin: 0 auto;
    max-width: 34rem;
    display: grid;
    gap: 1rem;
  }
  .tasks li {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 0.75rem;
    padding: 1rem 1.25rem;
  }
  h1 {
    font-size: 1.6rem;
    font-weight: 600;
  }
  h2 {
    margin: 0 0 0.25rem;
    font-size: 1.15rem;
  }
  .actions {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.75rem;
  }
  button {
    padding: 0.6rem 1rem;
    border-radius: 0.6rem;
    border: 1px solid var(--line);
    background: var(--surface);
  }
  .primary {
    background: var(--accent);
    border-color: var(--accent);
    color: white;
    font-weight: 600;
  }
  .empty {
    text-align: center;
    color: var(--muted);
    padding: 3rem 0;
  }
  .muted {
    color: var(--muted);
    margin: 0;
  }
</style>
