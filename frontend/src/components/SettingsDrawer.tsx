import { useEffect, useState } from "react";
import {
  getApiKey, setApiKey, getCancer, setCancer, CANCERS,
  maskKey, type CancerId,
} from "../settings";

/**
 * SettingsDrawer holds the two operator controls the SaaS layer needs:
 *   1. X-API-Key (masked, stored in localStorage, sent on every request)
 *   2. Cancer selector (breast / nsclc) — persisted, drives which panels
 *      render + which `?cancer=` value the /v1/case/full call carries.
 *
 * Kept as a right-hand drawer instead of an always-visible bar because
 * (a) most of the UI is a research surface where auth is often OFF, and
 * (b) the API key is sensitive — the drawer keeps it out of screenshots
 * unless the operator explicitly opens it.
 *
 * The drawer opens automatically on 401 (App.tsx wires `installApiHooks`
 * with an on401 that flips `open` → true here via `openOnAuthFail`).
 */
interface Props {
  open: boolean;
  onClose: () => void;
  onCancerChange: (c: CancerId) => void;
  cancers: Record<string, { state: string; case_full: boolean }>;
  lastRequestId: string;
}

export function SettingsDrawer(props: Props) {
  const [key, setKey] = useState(getApiKey());
  const [cancer, setCancerLocal] = useState<CancerId>(getCancer());
  const [showFull, setShowFull] = useState(false);

  useEffect(() => {
    if (props.open) {
      // Refresh from localStorage every time we open — another tab
      // might have edited it.
      setKey(getApiKey());
      setCancerLocal(getCancer());
    }
  }, [props.open]);

  function saveKey() {
    setApiKey(key.trim());
  }
  function clearKey() {
    setApiKey("");
    setKey("");
  }
  function pickCancer(c: CancerId) {
    setCancerLocal(c);
    setCancer(c);
    props.onCancerChange(c);
  }

  if (!props.open) return null;

  return (
    <div
      role="dialog"
      aria-label="Settings drawer"
      style={{
        position: "fixed",
        top: 0, right: 0, bottom: 0,
        width: 380, maxWidth: "100%",
        background: "var(--bg, #111)",
        color: "var(--fg, #eee)",
        borderLeft: "1px solid var(--border, #333)",
        boxShadow: "-4px 0 12px rgba(0,0,0,0.4)",
        padding: "1rem",
        overflow: "auto",
        zIndex: 1000,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <h2 style={{ margin: 0 }}>Settings</h2>
        <button onClick={props.onClose} aria-label="Close settings" style={{ background: "transparent", border: "none", color: "inherit", fontSize: "1.5rem", cursor: "pointer" }}>×</button>
      </div>

      <section style={{ marginTop: "1rem" }}>
        <h3 style={{ margin: "0 0 0.25rem 0", fontSize: "0.95rem" }}>API key</h3>
        <p style={{ fontSize: "0.8rem", color: "var(--fg-muted, #999)", margin: "0 0 0.5rem 0" }}>
          Sent as <code>X-API-Key</code>. Get one from your admin via{" "}
          <code>python -m oncology_arbiter.auth.cli issue &lt;tenant&gt;</code>.
          Stored in this browser's localStorage only.
        </p>
        <input
          type={showFull ? "text" : "password"}
          value={key}
          onChange={(e) => setKey(e.target.value)}
          placeholder="oa_live_…"
          aria-label="API key"
          style={{ width: "100%", fontFamily: "Menlo, monospace", fontSize: "0.85rem" }}
        />
        <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.5rem", flexWrap: "wrap" }}>
          <button onClick={saveKey} className="primary">Save</button>
          <button onClick={clearKey}>Clear</button>
          <label style={{ fontSize: "0.75rem", display: "inline-flex", alignItems: "center", gap: "0.25rem" }}>
            <input type="checkbox" checked={showFull} onChange={(e) => setShowFull(e.target.checked)} />
            show full
          </label>
        </div>
        {getApiKey() && !showFull && (
          <div style={{ fontSize: "0.75rem", color: "var(--fg-muted, #999)", marginTop: "0.25rem" }}>
            current: <code>{maskKey(getApiKey())}</code>
          </div>
        )}
      </section>

      <section style={{ marginTop: "1.5rem" }}>
        <h3 style={{ margin: "0 0 0.25rem 0", fontSize: "0.95rem" }}>Cancer</h3>
        <p style={{ fontSize: "0.8rem", color: "var(--fg-muted, #999)", margin: "0 0 0.5rem 0" }}>
          Sent as <code>?cancer=&lt;id&gt;</code> to <code>/v1/case/full</code>.
          Panels rendered below reflect this choice.
        </p>
        <div style={{ display: "grid", gap: "0.35rem" }}>
          {CANCERS.map((c) => {
            const cap = props.cancers?.[c];
            const state = cap?.state ?? "unknown";
            return (
              <label key={c} style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <input
                  type="radio"
                  name="cancer"
                  value={c}
                  checked={cancer === c}
                  onChange={() => pickCancer(c)}
                />
                <span style={{ fontWeight: 600 }}>{c}</span>
                <span className={`pill ${state}`} style={{ fontSize: "0.7rem" }}>{state}</span>
              </label>
            );
          })}
        </div>
      </section>

      <section style={{ marginTop: "1.5rem" }}>
        <h3 style={{ margin: "0 0 0.25rem 0", fontSize: "0.95rem" }}>Last X-Request-Id</h3>
        <div style={{ fontFamily: "Menlo, monospace", fontSize: "0.8rem", wordBreak: "break-all" }}>
          {props.lastRequestId || <span style={{ color: "var(--fg-muted, #999)" }}>(none yet)</span>}
        </div>
      </section>
    </div>
  );
}
