<!--
  Shell. Note the navigation carries no badges or counts — see docs/UX_NEURODIVERGENT.md.
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import VoiceButton from '$lib/voice/VoiceButton.svelte';
  import { disarm, initVoice } from '$lib/voice/controller';
  import { startNudges } from '$lib/voice/nudge';
  import '../app.css';
  let { children } = $props();

  onMount(() => {
    void initVoice();
    const stopNudges = startNudges();
    return () => {
      stopNudges();
      disarm();
    };
  });

  const nav = [
    { href: '/', label: 'Today' },
    { href: '/tasks', label: 'All tasks' },
    { href: '/skills', label: 'Skills' },
    { href: '/cameras', label: 'Cameras' },
    { href: '/metrics', label: 'Habits' },
    { href: '/settings', label: 'Settings' }
  ];
</script>

<div class="app">
  <header>
    <a class="brand" href="/">
      <!-- The mark, inline so it inherits currentColor and themes with the header. -->
      <svg class="brand-mark" viewBox="0 0 64 64" aria-hidden="true">
        <path d="M11 31 L32 8 L53 31 Z" />
        <rect x="13" y="30" width="38" height="28" rx="5" />
        <circle cx="32" cy="44" r="9.5" />
        <circle cx="32" cy="44" r="3.75" class="mark-core" />
      </svg>
      OpenHup
    </a>
    <nav>
      {#each nav as item}
        <a href={item.href} aria-current={$page.url.pathname === item.href ? 'page' : undefined}>
          {item.label}
        </a>
      {/each}
    </nav>
    <div class="spacer"></div>
    <VoiceButton />
  </header>

  <main>{@render children()}</main>
</div>

<style>
  :global(:root) {
    --bg: #fbfaf8;
    --surface: #ffffff;
    --text: #1c1a17;
    --muted: #6b6560;
    --line: #e4e0da;
    --accent: #2f6f4f;
    --danger: #b23b2e;
    color-scheme: light dark;
  }
  @media (prefers-color-scheme: dark) {
    :global(:root) {
      --bg: #171614;
      --surface: #201f1c;
      --text: #ece9e4;
      --muted: #9c948c;
      --line: #322f2b;
      --accent: #7fb896;
      --danger: #e4796a;
    }
  }
  :global(body) {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
    line-height: 1.5;
  }
  /* No animation longer than ~200ms anywhere in a flow used many times a day. */
  :global(*) {
    transition-duration: 150ms;
  }
  @media (prefers-reduced-motion: reduce) {
    :global(*) {
      transition: none !important;
      animation: none !important;
    }
  }
  .app {
    min-height: 100vh;
  }
  header {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 0.9rem 1.25rem;
    border-bottom: 1px solid var(--line);
    background: var(--surface);
    flex-wrap: wrap;
  }
  .brand {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-weight: 700;
    text-decoration: none;
    color: var(--text);
  }
  .brand-mark {
    width: 1.35rem;
    height: 1.35rem;
    fill: currentColor;
    flex: none;
  }
  /* Punch the lens core through to the header background. */
  .mark-core {
    fill: var(--surface);
  }
  nav {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
  }
  nav a {
    color: var(--muted);
    text-decoration: none;
    font-size: 0.95rem;
  }
  nav a[aria-current='page'] {
    color: var(--text);
    font-weight: 600;
  }
  .spacer {
    flex: 1;
  }
  main {
    padding: 1.75rem 1.25rem 4rem;
  }
</style>
