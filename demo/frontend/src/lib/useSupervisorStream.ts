import { useEffect, useState } from "react";

/** One piece of per-predicate evidence attached to an alert (M2 sentiment / M3 rule / lexical). */
export interface SupervisorEvidence {
  predicate_type: string;
  matched_text?: string;
  score?: number;
  threshold?: number;
}

/** An evidence-backed alert emitted by the shipped GET /supervisor/stream (M0 + M2 + M3). */
export interface SupervisorAlert {
  alert_id: string;
  conversation_id: string;
  window_id: string;
  rule_name: string;
  evidence: SupervisorEvidence[];
}

export interface SupervisorStreamState {
  /** All alerts, newest first (the inbox). */
  alerts: SupervisorAlert[];
  /** Latest alert per conversation (the active-call list), keyed by conversation_id. */
  activeCalls: Record<string, SupervisorAlert>;
}

/**
 * Subscribe to the supervisor SSE stream and expose live alert state.
 *
 * The React analog of chatwoot's ActionCable connector (connect → event → state): a native
 * `EventSource` parses `alert` events into an inbox + a per-conversation active-call map. No runtime
 * dependency; `EventSource` handles auto-reconnect. The default URL goes through the demo's `/api`
 * Vite proxy, which rewrites to the backend `/supervisor/stream`.
 */
export function useSupervisorStream(url = "/api/supervisor/stream"): SupervisorStreamState {
  const [alerts, setAlerts] = useState<SupervisorAlert[]>([]);
  const [activeCalls, setActiveCalls] = useState<Record<string, SupervisorAlert>>({});

  useEffect(() => {
    const source = new EventSource(url);
    const onAlert = (event: MessageEvent) => {
      const alert = JSON.parse(event.data) as SupervisorAlert;
      setAlerts((prev) => [alert, ...prev]);
      setActiveCalls((prev) => ({ ...prev, [alert.conversation_id]: alert }));
    };
    source.addEventListener("alert", onAlert as EventListener);
    return () => {
      source.removeEventListener("alert", onAlert as EventListener);
      source.close();
    };
  }, [url]);

  return { alerts, activeCalls };
}
