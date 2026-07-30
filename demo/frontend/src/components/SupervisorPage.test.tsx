import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { SupervisorPage } from "./SupervisorPage";

class MockEventSource {
  static last: MockEventSource | null = null;
  listeners: Record<string, ((e: MessageEvent) => void)[]> = {};
  constructor(public url: string) {
    MockEventSource.last = this;
  }
  addEventListener(type: string, cb: (e: MessageEvent) => void) {
    (this.listeners[type] ??= []).push(cb);
  }
  removeEventListener() {}
  close() {}
  emit(type: string, data: unknown) {
    for (const cb of this.listeners[type] ?? []) cb({ data: JSON.stringify(data) } as MessageEvent);
  }
}

describe("SupervisorPage", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("renders an incoming alert with its rule name and evidence", () => {
    vi.stubGlobal("EventSource", MockEventSource);
    render(<SupervisorPage />);

    expect(screen.getByText(/Aguardando alertas/)).toBeInTheDocument();

    act(() =>
      MockEventSource.last!.emit("alert", {
        alert_id: "a1",
        conversation_id: "conv_42",
        window_id: "w1",
        rule_name: "cancellation",
        evidence: [
          { predicate_type: "regex", matched_text: "quero cancelar" },
          { predicate_type: "sentiment", matched_text: "negative", score: 1.2 },
        ],
      }),
    );

    // DoD: the alert (rule name) + its evidence render in the inbox, and the active call appears.
    expect(screen.getAllByText("cancellation").length).toBeGreaterThan(0);
    expect(screen.getByText(/quero cancelar/)).toBeInTheDocument();
    expect(screen.getByText(/negative/)).toBeInTheDocument();
    expect(screen.getAllByText(/conv_42/).length).toBeGreaterThan(0); // active-call list + alert row
    expect(screen.getByText(/Chamadas ativas \(1\)/)).toBeInTheDocument();
  });
});
