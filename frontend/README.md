# openhup-frontend

SvelteKit + TypeScript, built as a static bundle into `../backend/static` and served by the API. No
SSR, no Node runtime on your server, no CORS (same-origin in dev via a Vite proxy, same-origin in
production by construction).

```sh
pnpm install
pnpm dev          # http://localhost:5173, proxying /api to 127.0.0.1:8080
pnpm build        # → ../backend/static
```

## State of this component

**Functional scaffold.** Every route renders and the app builds, typechecks, and installs from a
committed lockfile. `/` (Today) is the reference for tone; the rest are working first passes rather
than finished features.

| Route | State |
|---|---|
| `/` Today | done — one task, one step, one photograph |
| `/tasks` | done — backlog mode, for people who prefer seeing everything |
| `/skills` | done — list, enable toggle, **the simulate panel** |
| `/cameras` | basic — camera/anchor list, baseline capture (ROI polygon editor still to build) |
| `/metrics` | basic — goals and the weekly report (charts still to build) |
| `/settings` | basic — personality picker with live preview, notification channels |

The highest-value thing to build next is the **ROI polygon editor** on `/cameras`: drawing anchors by
hand in YAML is the worst part of setup, and subregions — which enable spatial micro-tasking — are
effectively unreachable without it.

## Read this before contributing UI

[docs/UX_NEURODIVERGENT.md](../docs/UX_NEURODIVERGENT.md) lists constraints that are not obvious and
that will be pushed back on in review. The short version:

- **No counts of outstanding work.** No badges, no "3 tasks", nothing. The API does not even compute a
  total, deliberately.
- **In single-task-focus mode, do not fetch the list.** Call `/tasks/next` only. A list you cannot see
  cannot overwhelm you; a list fetched and hidden is one refactor away from being shown.
- **Red means unsafe**, and only that. Nothing about an undone chore is red.
- **Progress shows what is done**, never how far from complete.
- **Snooze is as prominent as complete.** Sometimes "not now" is right, and making it awkward teaches
  people to dismiss things instead.
- **No confirmation dialogs on completing a task.** Trust the press.
- **Nothing over ~200 ms** in a flow used many times a day, and honour `prefers-reduced-motion`.
- **Always render `plain_text` for assistive tech.** Personality-rendered `text` is for display;
  `plain_text` is always present and always factual.

## Conventions

- Svelte 5 runes (`$state`, `$props`, `$derived`).
- `src/lib/api/client.ts` is the only place that calls `fetch`.
- `src/lib/stores/events.ts` owns the single WebSocket. Use `liveResource(fetcher, patterns, initial)`
  to make a view refetch on the events that could change it — simpler and more robust than patching
  local state from event payloads, and the data volumes are tiny.
- Snapshots are resolved with `snapshotUrl()`; they are served through the API so authentication
  applies to imagery of someone's home.
- CSS custom properties for theming, with a dark scheme via `prefers-color-scheme`. No CSS-in-JS.

Types for the endpoints in use are hand-written in `client.ts`. `make types` regenerates the full set
from the backend's Pydantic models via JSON Schema — run it after changing a shared model.
