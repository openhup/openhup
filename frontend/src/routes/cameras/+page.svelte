<!--
  Cameras and anchors. The list, plus a note that ROI editing is the main thing to make easy.
-->

<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '$lib/api/client';

  let cameras = $state<Record<string, unknown>[]>([]);
  let anchors = $state<Record<string, unknown>[]>([]);

  onMount(async () => {
    [cameras, anchors] = await Promise.all([api.cameras(), api.anchors()]);
  });
</script>

<svelte:head><title>Cameras · OpenHup</title></svelte:head>

<h1>Cameras</h1>
<ul class="list">
  {#each cameras as camera (camera.id as string)}
    <li>
      <h2>{camera.name as string}</h2>
      <p class="muted">{camera.kind as string}</p>
    </li>
  {/each}
</ul>

<h1>Anchors</h1>
<ul class="list">
  {#each anchors as anchor (anchor.id as string)}
    <li>
      <h2>{anchor.label as string}</h2>
      <p class="muted">{anchor.id as string}</p>
      <button onclick={() => api.captureBaseline(anchor.id as string)}>Capture baseline</button>
    </li>
  {/each}
</ul>

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
    max-width: 40rem;
    margin: 1.5rem auto 0.75rem;
  }
  h2 {
    margin: 0 0 0.25rem;
    font-size: 1.05rem;
  }
  .muted {
    color: var(--muted);
    margin: 0 0 0.5rem;
  }
  button {
    padding: 0.5rem 0.9rem;
    border-radius: 0.6rem;
    border: 1px solid var(--line);
    background: var(--surface);
  }
</style>
