import { AlertTriangle, PhoneCall } from "lucide-react";

import {
  type SupervisorAlert,
  type SupervisorEvidence,
  useSupervisorStream,
} from "@/lib/useSupervisorStream";

const RULE_COLOR: Record<string, string> = {
  cancellation: "bg-red-100 text-red-700",
  escalation: "bg-amber-100 text-amber-700",
  cancellation_risk: "bg-red-100 text-red-700",
};

/** Lightweight evidence chip — the alert evidence (EvidenceItem: sentiment / rule predicate) differs
 * from the search PredicateEvidence shape used by EvidenceBadge, so it gets its own compact renderer. */
function EvidenceChip({ ev }: { ev: SupervisorEvidence }) {
  const isSentiment = ev.predicate_type === "sentiment";
  const color = isSentiment
    ? ev.matched_text === "negative"
      ? "bg-red-50 text-red-600"
      : "bg-emerald-50 text-emerald-600"
    : "bg-slate-100 text-slate-600";
  return (
    <span className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium ${color}`}>
      {ev.predicate_type}
      {ev.matched_text && <span className="font-normal opacity-75">&quot;{ev.matched_text}&quot;</span>}
    </span>
  );
}

function AlertRow({ alert }: { alert: SupervisorAlert }) {
  const color = RULE_COLOR[alert.rule_name] ?? "bg-slate-100 text-slate-700";
  return (
    <li className="flex flex-col gap-1 rounded-lg border border-slate-200 p-3">
      <div className="flex items-center gap-2">
        <AlertTriangle className="h-4 w-4 text-amber-500" aria-hidden />
        <span className={`rounded px-1.5 py-0.5 text-xs font-semibold ${color}`}>{alert.rule_name}</span>
        <span className="text-xs text-slate-500">conversa {alert.conversation_id}</span>
      </div>
      {alert.evidence.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {alert.evidence.map((ev, i) => (
            <EvidenceChip key={i} ev={ev} />
          ))}
        </div>
      )}
    </li>
  );
}

/** Supervisor live-monitoring surface (M4): active-call list + alert inbox, fed by the SSE stream. */
export function SupervisorPage() {
  const { alerts, activeCalls } = useSupervisorStream();
  const calls = Object.values(activeCalls);

  return (
    <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
      <section aria-label="Chamadas ativas" className="md:col-span-1">
        <h2 className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-700">
          <PhoneCall className="h-4 w-4" aria-hidden /> Chamadas ativas ({calls.length})
        </h2>
        {calls.length === 0 ? (
          <p className="text-sm text-slate-400">Nenhuma chamada crítica no momento.</p>
        ) : (
          <ul className="flex flex-col gap-2">
            {calls.map((a) => (
              <li key={a.conversation_id} className="rounded-lg border border-slate-200 p-2 text-sm">
                <span className="font-medium">{a.conversation_id}</span>
                <span className="ml-2 text-xs text-slate-500">{a.rule_name}</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section aria-label="Alertas ao vivo" className="md:col-span-2">
        <h2 className="mb-2 text-sm font-semibold text-slate-700">Alertas ao vivo ({alerts.length})</h2>
        {alerts.length === 0 ? (
          <p className="text-sm text-slate-400">Aguardando alertas…</p>
        ) : (
          <ul className="flex flex-col gap-2">
            {alerts.map((a) => (
              <AlertRow key={a.alert_id} alert={a} />
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
