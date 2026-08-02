/**
 * Typed API client.
 *
 * Types here are hand-written for the endpoints the UI uses; the full set is generated from the
 * backend's JSON Schema via `make types`. Same-origin in both dev (proxied) and production (served
 * by the API), so there is no base URL to configure and no CORS.
 */

const BASE = '/api/v1';

export type Urgency = 'info' | 'low' | 'normal' | 'high' | 'critical';

export interface MicroStep {
  index: number;
  text: string;
  done: boolean;
  subregion_id: string | null;
}

export interface Task {
  id: string;
  skill_id: string;
  anchor_id: string;
  anchor_label: string;
  state: string;
  urgency: Urgency;
  /** Personality-rendered wording. */
  text: string;
  /** Tone-free wording. Always present: used by screen readers whatever the personality. */
  plain_text: string;
  text_source: 'llm' | 'template' | 'user';
  /** The current micro-step. In single-task focus this is the only line to show. */
  current_text: string;
  micro_steps: MicroStep[];
  current_step: number;
  progress: number;
  before_snapshot: string | null;
  after_snapshot: string | null;
  created_at: string;
  completed_at: string | null;
  snoozed_until: string | null;
  note: string | null;
  reopened: boolean;
}

export interface Alert {
  id: string;
  anchor_label: string;
  state: string;
  urgency: Urgency;
  text: string;
  plain_text: string;
  facts: string[];
  snapshot_ref: string | null;
  created_at: string;
  acknowledged_at: string | null;
}

export interface SkillSummary {
  id: string;
  enabled: boolean;
  description: string;
  effect_type: string;
  urgency: Urgency;
  anchors: string[];
  /** Plain-language rendering. Some users will only ever read this, never the YAML. */
  explanation: string;
  warnings: string[];
  errors: string[];
  origin: string;
  source_text: string | null;
}

/** One thing the household told the assistant. Everything here is local and deletable. */
export interface MemoryFact {
  id: string;
  fact: string;
  topic: string | null;
  /** Where it came from: voice | settings | api. */
  source: string;
  created_at: string;
}

/** A pattern the assistant learned from the household's own episode history. */
export interface MemoryPattern {
  id: string;
  kind: 'cadence' | 'time_of_day';
  skill_id: string;
  anchor_id: string;
  /** Forward-facing claim, ready to speak. */
  summary: string;
  confidence: number;
  /** The numbers behind the claim, shown on the review screen. */
  evidence: Record<string, unknown>;
  nudge_eligible: boolean;
  last_nudge_at: string | null;
  updated_at: string | null;
}

export interface SystemHealth {
  status: 'ok' | 'degraded';
  cameras: { id: string; enabled: boolean; stale: boolean; last_frame_at: string | null }[];
  bus_connected: boolean;
  llm_available: boolean;
  /** Surfaced prominently: a dead camera must not look like a tidy house. */
  problems: string[];
}

export interface SimulationResult {
  verdict: string;
  advice: string[];
  tasks_created: number;
  alerts_raised: number;
  per_day: number;
  episodes: number;
}

export type SpeechProvider = 'browser' | 'openai' | 'openai_compatible';

export interface VoiceConfig {
  enabled: boolean;
  stt_provider: SpeechProvider;
  tts_provider: SpeechProvider;
  stt_remote: boolean;
  tts_remote: boolean;
  stt_on_server: boolean;
  tts_on_server: boolean;
  wake_word: string;
  language: string;
  tts_voice: string;
}

/** One enrolled household member (ADR-016): a person who said yes to the consent question.
 *  The embedding is the entire biometric surface and never leaves the device. */
export interface Member {
  id: string;
  name: string;
  active: boolean;
  enrolled_at: string;
  last_seen_at: string | null;
  /** Dimension of the stored embedding; 0 means consent was granted without a capture yet. */
  embedding_dim: number;
}

/** The personality gamble (ADR-014): the mystery voice drawn at setup. */
export interface PersonalityDraw {
  /** The drawn personality id, or null when no gamble has happened. */
  drawn: string | null;
  reroll_count: number;
  pool: string[];
  gamble_enabled: boolean;
}

/** One win the assistant noticed (ADR-015): an anchor stayed clear for whole days. */
export interface WinMilestone {
  id: string;
  anchor_id: string;
  kind: 'clear_days' | 'record_clear_days';
  days: number;
  /** Tone-free summary for the review screen. */
  summary: string;
  spoken: boolean;
  achieved_at: string;
}

export interface VoiceCommandResult {
  intent: 'task_command' | 'query' | 'skill_dictation' | 'navigate' | 'memory' | 'unknown';
  reply: string;
  action: string | null;
  task_id: string | null;
  skill: Record<string, unknown> | null;
  explanation: string;
  confidence: number;
  unsupported: string | null;
  problems: string[];
  needs_confirmation: boolean;
  target: string | null;
}

class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly findings?: { code: string; message: string }[]
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { 'content-type': 'application/json', ...(init?.headers ?? {}) }
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    // A 422 from /skills carries every compile finding, so the UI can list them all at once
    // rather than making someone fix one problem per save.
    throw new ApiError(response.status, body.detail ?? body.error ?? response.statusText, body.findings);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const api = {
  // --- tasks ---------------------------------------------------------------------------
  /** The one thing to do now. Single-task-focus mode calls only this. */
  nextTask: () => request<Task | null>('/tasks/next'),
  tasks: (state = 'open') => request<Task[]>(`/tasks?state=${state}`),
  task: (id: string) => request<Task>(`/tasks/${id}`),

  completeTask: (id: string) => patchTask(id, { action: 'complete' }),
  startTask: (id: string) => patchTask(id, { action: 'start' }),
  snoozeTask: (id: string, minutes: number) => patchTask(id, { action: 'snooze', minutes }),
  dismissTask: (id: string, note?: string) => patchTask(id, { action: 'dismiss', note }),
  /** The most valuable feedback the system gets: it drives threshold suggestions. */
  markFalsePositive: (id: string, note?: string) => patchTask(id, { action: 'false_positive', note }),

  // --- alerts --------------------------------------------------------------------------
  alerts: (state?: string) => request<Alert[]>(`/alerts${state ? `?state=${state}` : ''}`),
  ackAlert: (id: string) => request<Alert>(`/alerts/${id}/ack`, { method: 'POST' }),

  // --- skills --------------------------------------------------------------------------
  skills: () => request<SkillSummary[]>('/skills'),
  skill: (id: string) => request<Record<string, unknown>>(`/skills/${id}`),
  parseSkill: (text: string) =>
    request<{
      ok: boolean;
      skill: Record<string, unknown> | null;
      explanation: string;
      confidence: number;
      unsupported: string | null;
      problems: string[];
      heuristic: boolean;
      needs_confirmation: boolean;
    }>('/skills/parse', { method: 'POST', body: JSON.stringify({ text }) }),
  createSkill: (skill: Record<string, unknown>) =>
    request<{ created: string; warnings: string[] }>('/skills', {
      method: 'POST',
      body: JSON.stringify(skill)
    }),
  updateSkill: (id: string, patch: Record<string, unknown>) =>
    request<{ updated: string }>(`/skills/${id}`, { method: 'PATCH', body: JSON.stringify(patch) }),
  /** Dry-run against real history. Show this before letting anyone enable a skill. */
  simulate: (id: string, days = 7) =>
    request<SimulationResult>(`/skills/${id}/simulate?days=${days}`, { method: 'POST' }),

  // --- cameras and anchors -------------------------------------------------------------
  cameras: () => request<Record<string, unknown>[]>('/cameras'),
  anchors: () => request<Record<string, unknown>[]>('/anchors'),
  captureBaseline: (anchorId: string) =>
    request<{ baseline_ref: string }>(`/anchors/${anchorId}/baseline`, { method: 'POST' }),

  // --- insight -------------------------------------------------------------------------
  detectors: () => request<{ detectors: Record<string, unknown>[] }>('/detectors'),
  metricSeries: (metric: string, days = 30) =>
    request<{ points: { ts: string; value: number }[] }>(
      `/metrics/series?metric=${metric}&days=${days}`
    ),
  goals: () => request<Record<string, unknown>[]>('/metrics/goals'),
  weeklyReport: () => request<Record<string, unknown>>('/metrics/report/weekly'),
  personalities: () => request<Record<string, unknown>[]>('/personalities'),
  previewPersonality: (id: string) =>
    request<Record<string, string>>(`/personalities/${id}/preview`, { method: 'POST' }),

  systemInfo: () => request<Record<string, unknown>>('/system/info'),
  health: () => request<SystemHealth>('/system/health'),
  llmUsage: () => request<Record<string, unknown>>('/system/llm-usage'),

  // --- voice --------------------------------------------------------------------------
  voiceConfig: () => request<VoiceConfig>('/voice/config'),
  voiceCommand: (text: string, speaker?: string | null) =>
    request<VoiceCommandResult>('/voice/command', {
      method: 'POST',
      body: JSON.stringify({ text, speaker: speaker ?? null })
    }),

  // --- members (ADR-016: consent-gated identity) -----------------------------------
  /** Everyone who consented to be remembered. Reviewable and deletable, like facts. */
  members: () => request<{ members: Member[]; enabled: boolean }>('/members'),
  /** Record a consent answer. "yes" hands off to enrollment; "no" stops the re-ask. */
  answerConsent: (anchorId: string, answer: 'yes' | 'no', name?: string) =>
    request<{ reply: string }>('/members/consent', {
      method: 'POST',
      body: JSON.stringify({ anchor_id: anchorId, answer, name: name ?? null })
    }),
  /** Forget a member: their embedding and presence history go with them. */
  deleteMember: (id: string) => request<void>(`/members/${id}`, { method: 'DELETE' }),

  // --- personality -------------------------------------------------------------------
  /** The state of the gamble. `drawn` is the mystery - the UI hides it behind a Reveal. */
  personalityDraw: () => request<PersonalityDraw>('/personality/draw'),
  /** Draw, or re-draw, the mystery voice. */
  drawPersonality: () =>
    request<PersonalityDraw>('/personality/draw', { method: 'POST' }),
  /** Stop the gamble: the configured default speaks again. */
  clearPersonalityDraw: () => request<void>('/personality/draw', { method: 'DELETE' }),
  /** Wins the assistant has noticed, for the review screen. */
  wins: () => request<{ wins: WinMilestone[] }>('/personality/wins'),
  /** Forget a win. A deleted win is gone; the same milestone can be celebrated again. */
  deleteWin: (id: string) => request<void>(`/personality/wins/${id}`, { method: 'DELETE' }),

  // --- memory -------------------------------------------------------------------------
  /** Everything the assistant knows, for the review screen. */
  memoryFacts: () => request<MemoryFact[]>('/memory'),
  addMemoryFact: (fact: string, topic?: string) =>
    request<{ created: string }>('/memory', {
      method: 'POST',
      body: JSON.stringify({ fact, topic: topic?.trim() ? topic.trim() : null })
    }),
  deleteMemoryFact: (id: string) => request<void>(`/memory/${id}`, { method: 'DELETE' }),
  /** Learned patterns, freshly recomputed with their evidence. */
  memoryPatterns: () => request<{ patterns: MemoryPattern[]; note: string }>('/memory/patterns'),
  /** Say a learned pattern is not useful: it is never surfaced or nudged again. */
  dismissMemoryPattern: (id: string) =>
    request<void>(`/memory/patterns/${id}`, { method: 'DELETE' })
};

function patchTask(id: string, body: Record<string, unknown>) {
  return request<Task>(`/tasks/${id}`, { method: 'PATCH', body: JSON.stringify(body) });
}

/** Resolve a `snap://` reference to a URL. Snapshots go through the API so auth applies. */
export function snapshotUrl(ref: string | null): string | null {
  if (!ref) return null;
  return `${BASE}/snapshots/${ref.replace(/^snap:\/\//, '')}`;
}

export { ApiError };
