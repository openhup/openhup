<!--
  Today — the default view, and the one that embodies the UX rules.

  Three things it deliberately does NOT do:
    1. show a count of outstanding tasks (there is no badge, and the API does not compute one)
    2. fetch the task list at all in single-task-focus mode
    3. use red for "not done" — red means unsafe, and only that

  What it does show: one task, one micro-step, one photograph, and a snooze button as easy to reach
  as complete. Sometimes the right answer is "not now", and making that awkward only teaches people
  to dismiss things instead.
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { api, snapshotUrl, type Alert, type Task } from '$lib/api/client';
  import { connect, liveResource, connection } from '$lib/stores/events';

  let task = $state<Task | null>(null);
  let alerts = $state<Alert[]>([]);
  let health = $state<{ status: string; problems: string[] } | null>(null);
  let busy = $state(false);
  let celebrating = $state<Task | null>(null);

  async function refresh() {
    [task, alerts, health] = await Promise.all([
      api.nextTask(),
      api.alerts('active'),
      api.health()
    ]);
  }

  onMount(() => {
    refresh();
    // Re-fetch on anything that could change the answer, rather than patching local state from
    // event payloads. The data volumes are tiny and this cannot drift.
    const stop = connect('task,alert,system');
    const unsubscribe = liveResource(refresh, ['task.', 'alert.', 'system.'], null).subscribe(
      () => {}
    );
    return () => {
      unsubscribe();
      stop();
    };
  });

  async function complete() {
    if (!task) return;
    busy = true;
    const finished = task;
    try {
      const updated = await api.completeTask(task.id);
      // Show the "after" picture. This is the only moment of reward the loop has, and it is the
      // single most motivating thing the system produces.
      if (updated.after_snapshot || finished.before_snapshot) celebrating = updated;
      setTimeout(() => (celebrating = null), 6000);
      await refresh();
    } finally {
      busy = false;
    }
  }

  async function snooze(minutes: number) {
    if (!task) return;
    busy = true;
    try {
      await api.snoozeTask(task.id, minutes);
      await refresh();
    } finally {
      busy = false;
    }
  }
</script>

<svelte:head><title>Today · OpenHup</title></svelte:head>

<!-- Safety first, visually and structurally. Alerts sit above everything. -->
{#each alerts as alert (alert.id)}
  <section class="alert" role="alert">
    <h2>{alert.plain_text}</h2>
    {#if alert.facts.length}
      <ul>
        {#each alert.facts as fact}<li>{fact}</li>{/each}
      </ul>
    {/if}
    {#if snapshotUrl(alert.snapshot_ref)}
      <img src={snapshotUrl(alert.snapshot_ref)} alt="Snapshot of {alert.anchor_label}" />
    {/if}
    <button onclick={() => api.ackAlert(alert.id).then(refresh)}>Acknowledged</button>
  </section>
{/each}

{#if celebrating}
  <section class="done">
    <h2>{celebrating.anchor_label} is clear.</h2>
    {#if snapshotUrl(celebrating.after_snapshot)}
      <img src={snapshotUrl(celebrating.after_snapshot)} alt="After" />
    {/if}
  </section>
{:else if task}
  <section class="task">
    <!-- The micro-step, not the whole task. One thing. -->
    <h1>{task.current_text}</h1>
    <p class="where">{task.anchor_label}</p>

    {#if snapshotUrl(task.before_snapshot)}
      <!-- Visual anchoring: removes the ambiguity of "clear the counter" and the working-memory
           load of reconstructing what was meant. -->
      <img src={snapshotUrl(task.before_snapshot)} alt="Current state of {task.anchor_label}" />
    {/if}

    {#if task.micro_steps.length > 1}
      <!-- Progress shown as what is DONE, never as how far from complete. -->
      <p class="steps">
        {task.micro_steps.filter((s) => s.done).length} of {task.micro_steps.length} done
      </p>
    {/if}

    {#if task.reopened}
      <p class="note">{task.note}</p>
    {/if}

    <div class="actions">
      <button class="primary" onclick={complete} disabled={busy}>Done</button>
      <!-- As easy to reach as Done, on purpose. -->
      <button onclick={() => snooze(60)} disabled={busy}>Later</button>
      <button onclick={() => snooze(60 * 14)} disabled={busy}>Tomorrow</button>
    </div>

    <details>
      <summary>Not a real task?</summary>
      <p>
        Telling OpenHup this was wrong is the most useful thing you can do — it feeds the threshold
        suggestions for this skill.
      </p>
      <button onclick={() => api.markFalsePositive(task!.id).then(refresh)}>
        That's supposed to be there
      </button>
    </details>
  </section>
{:else}
  <section class="clear">
    <svg class="clear-mark" viewBox="0 0 64 64" aria-hidden="true">
      <path d="M11 31 L32 8 L53 31 Z" />
      <rect x="13" y="30" width="38" height="28" rx="5" />
      <circle cx="32" cy="44" r="9.5" />
      <circle cx="32" cy="44" r="3.75" class="mark-core" />
    </svg>
    <h1>Nothing right now.</h1>
    <p>OpenHup is watching. It will say something when there is something worth saying.</p>
  </section>
{/if}

{#if health?.problems?.length}
  <!-- Surfaced plainly, because a dead camera producing no tasks looks exactly like a tidy house,
       and that confusion is the worst failure mode this system has. -->
  <aside class="problems">
    <h2>Worth a look</h2>
    <ul>
      {#each health.problems as problem}<li>{problem}</li>{/each}
    </ul>
  </aside>
{/if}

{#if $connection === 'closed'}
  <p class="offline">Reconnecting…</p>
{/if}

<style>
  section {
    max-width: 34rem;
    margin: 0 auto 1.5rem;
  }
  h1 {
    font-size: 1.6rem;
    line-height: 1.3;
    font-weight: 600;
    margin: 0 0 0.25rem;
  }
  .where {
    color: var(--muted);
    margin: 0 0 1rem;
  }
  img {
    width: 100%;
    border-radius: 0.75rem;
    display: block;
    margin-bottom: 1rem;
  }
  .actions {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
  }
  button {
    padding: 0.7rem 1.1rem;
    border-radius: 0.6rem;
    border: 1px solid var(--line);
    background: var(--surface);
    font: inherit;
    cursor: pointer;
  }
  .primary {
    background: var(--accent);
    border-color: var(--accent);
    color: white;
    font-weight: 600;
  }
  /* Red is reserved for unsafe. Nothing about an undone chore is red. */
  .alert {
    border-left: 4px solid var(--danger);
    padding-left: 1rem;
  }
  .done {
    text-align: center;
  }
  .clear {
    text-align: center;
    color: var(--muted);
    padding: 3rem 0;
  }
  .clear-mark {
    width: 3rem;
    height: 3rem;
    fill: var(--muted);
    margin-bottom: 0.5rem;
    opacity: 0.6;
  }
  /* Punch the lens core through to the page background. */
  .mark-core {
    fill: var(--bg);
  }
  .steps,
  .note {
    color: var(--muted);
    font-size: 0.9rem;
  }
  .problems {
    max-width: 34rem;
    margin: 2rem auto 0;
    font-size: 0.9rem;
    color: var(--muted);
  }
  details {
    margin-top: 1.5rem;
    font-size: 0.9rem;
    color: var(--muted);
  }
  .offline {
    text-align: center;
    color: var(--muted);
    font-size: 0.85rem;
  }
</style>
