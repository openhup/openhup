/**
 * Live event stream.
 *
 * One WebSocket, reconnecting with backoff, feeding Svelte stores. Topics are filtered server-side,
 * so a phone on mobile data does not receive every observation from every camera.
 */

import { readable, writable, type Readable } from 'svelte/store';

export interface Envelope {
  id: string;
  type: string;
  ts: string;
  payload: Record<string, unknown>;
  skill_id: string | null;
  anchor_id: string | null;
  episode_id: string | null;
  source: string;
}

export type ConnectionState = 'connecting' | 'open' | 'closed';

export const connection = writable<ConnectionState>('connecting');

/** The last 100 events, newest first. Feeds the activity timeline. */
export const recentEvents = writable<Envelope[]>([]);

const listeners = new Map<string, Set<(event: Envelope) => void>>();

/**
 * Subscribe to an event type, or a prefix like `task.` for a whole family.
 * Returns an unsubscribe function.
 */
export function onEvent(pattern: string, handler: (event: Envelope) => void): () => void {
  const set = listeners.get(pattern) ?? new Set();
  set.add(handler);
  listeners.set(pattern, set);
  return () => set.delete(handler);
}

function dispatch(event: Envelope) {
  recentEvents.update((events) => [event, ...events].slice(0, 100));
  for (const [pattern, handlers] of listeners) {
    if (event.type === pattern || event.type.startsWith(pattern)) {
      for (const handler of handlers) handler(event);
    }
  }
}

export function connect(topics = 'task,alert,skill,system'): () => void {
  if (typeof window === 'undefined') return () => {};

  let socket: WebSocket | null = null;
  let closed = false;
  let backoff = 1000;
  let keepalive: ReturnType<typeof setInterval> | null = null;

  const open = () => {
    if (closed) return;
    connection.set('connecting');
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(`${scheme}://${location.host}/api/v1/ws/events?topics=${topics}`);

    socket.onopen = () => {
      connection.set('open');
      backoff = 1000;
      // The server notices a dead client from a failed receive; a ping keeps intermediaries from
      // silently dropping an idle connection.
      keepalive = setInterval(() => socket?.send('ping'), 30_000);
    };

    socket.onmessage = (message) => {
      try {
        const data = JSON.parse(message.data);
        if (data.type && data.type !== 'pong') dispatch(data as Envelope);
      } catch {
        /* a malformed frame is not worth breaking the UI over */
      }
    };

    socket.onclose = () => {
      connection.set('closed');
      if (keepalive) clearInterval(keepalive);
      if (!closed) {
        setTimeout(open, backoff);
        backoff = Math.min(backoff * 2, 30_000);
      }
    };
  };

  open();
  return () => {
    closed = true;
    if (keepalive) clearInterval(keepalive);
    socket?.close();
  };
}

/**
 * A store that refetches when matching events arrive.
 *
 * This is the pattern most views use: fetch once, then re-fetch on the events that could have
 * changed the answer. Simpler and more robust than patching local state from event payloads, and the
 * data volumes here are tiny.
 */
export function liveResource<T>(
  fetcher: () => Promise<T>,
  patterns: string[],
  initial: T
): Readable<T> {
  return readable<T>(initial, (set) => {
    let cancelled = false;
    const refresh = () => {
      fetcher()
        .then((value) => !cancelled && set(value))
        .catch(() => {
          /* leave the previous value on screen rather than flashing an error */
        });
    };
    refresh();
    const unsubscribes = patterns.map((pattern) => onEvent(pattern, refresh));
    return () => {
      cancelled = true;
      for (const unsubscribe of unsubscribes) unsubscribe();
    };
  });
}
