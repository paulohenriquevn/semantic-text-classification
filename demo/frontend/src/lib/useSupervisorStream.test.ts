import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { type SupervisorAlert, useSupervisorStream } from "./useSupervisorStream";

/** A controllable EventSource stand-in: the test dispatches `alert` events into it. */
class MockEventSource {
  static last: MockEventSource | null = null;
  listeners: Record<string, ((e: MessageEvent) => void)[]> = {};
  closed = false;
  constructor(public url: string) {
    MockEventSource.last = this;
  }
  addEventListener(type: string, cb: (e: MessageEvent) => void) {
    (this.listeners[type] ??= []).push(cb);
  }
  removeEventListener(type: string, cb: (e: MessageEvent) => void) {
    this.listeners[type] = (this.listeners[type] ?? []).filter((f) => f !== cb);
  }
  close() {
    this.closed = true;
  }
  emit(type: string, data: unknown) {
    for (const cb of this.listeners[type] ?? []) cb({ data: JSON.stringify(data) } as MessageEvent);
  }
}

function alert(id: string, conv: string, rule: string): SupervisorAlert {
  return { alert_id: id, conversation_id: conv, window_id: "w", rule_name: rule, evidence: [] };
}

describe("useSupervisorStream", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("collects dispatched alerts newest-first", () => {
    vi.stubGlobal("EventSource", MockEventSource);
    const { result } = renderHook(() => useSupervisorStream());
    act(() => MockEventSource.last!.emit("alert", alert("a1", "conv_1", "cancellation")));
    act(() => MockEventSource.last!.emit("alert", alert("a2", "conv_2", "escalation")));
    expect(result.current.alerts).toHaveLength(2);
    expect(result.current.alerts[0].alert_id).toBe("a2"); // newest first
  });

  it("tracks the latest alert per conversation (active calls)", () => {
    vi.stubGlobal("EventSource", MockEventSource);
    const { result } = renderHook(() => useSupervisorStream());
    act(() => MockEventSource.last!.emit("alert", alert("a1", "conv_1", "cancellation")));
    act(() => MockEventSource.last!.emit("alert", alert("a2", "conv_1", "escalation")));
    expect(Object.keys(result.current.activeCalls)).toEqual(["conv_1"]);
    expect(result.current.activeCalls["conv_1"].rule_name).toBe("escalation"); // latest wins
  });

  it("closes the stream on unmount", () => {
    vi.stubGlobal("EventSource", MockEventSource);
    const { unmount } = renderHook(() => useSupervisorStream());
    const src = MockEventSource.last!;
    unmount();
    expect(src.closed).toBe(true);
  });
});
