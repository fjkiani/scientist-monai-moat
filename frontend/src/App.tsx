import { useEffect, useMemo, useState } from "react";
import { ScreeningTab } from "./tabs/ScreeningTab";
import { BiopsyTab } from "./tabs/BiopsyTab";
import { TherapyTab } from "./tabs/TherapyTab";
import { CaseViewTab } from "./tabs/CaseViewTab";
import { NsclcTab } from "./tabs/NsclcTab";
import { DemoSamplesTab } from "./tabs/DemoSamplesTab";
import {
  getHealth, listModelCards, installApiHooks,
  type HealthResponseWithCancers, type ModelCardsIndex,
} from "./api";
import {
  getApiKey, getCancer, setCancer as persistCancer, setLastRequestId,
  type CancerId,
} from "./settings";
import { SettingsDrawer } from "./components/SettingsDrawer";
import { HonestyPills } from "./components/HonestyPills";
import { DemoModePlaceholder } from "./components/DemoModePlaceholder";
import { ColdStartBanner } from "./components/ColdStartBanner";

type Tab = "demo" | "screening" | "biopsy" | "therapy" | "case" | "cards" | "nsclc";

export function App() {
  // v0.2.2 / v0.3.0-alpha: Landing tab is context-dependent —
  // - if the deployment is a DEMO_MODE showcase, land on the Demo Samples
  //   tab so a visitor immediately sees what the pipeline produces on real
  //   data. Individual tabs still work but render a read-only placeholder.
  // - otherwise land on Case View (v0.2.2 behaviour), which bundles
  //   screening → biopsy → therapy → co-scientist for a first-time user.
  const [tab, setTab] = useState<Tab>("case");
  const [health, setHealth] = useState<HealthResponseWithCancers | null>(null);
  const [cards, setCards] = useState<ModelCardsIndex | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [cancer, setCancerState] = useState<CancerId>(getCancer());
  const [lastRequestId, setLastRequestIdState] = useState<string>("");
  // Cold-start banner state: unavailable=true while any api.ts fetch is
  // in a "backend down / dyno warming" state. Auto-clears when the next
  // successful response arrives.
  const [backendUnavailable, setBackendUnavailable] = useState(false);
  const [backendUnavailableReason, setBackendUnavailableReason] = useState("");

  // Wire the api.ts hooks once on mount. `apiKey` is read live from
  // localStorage so the operator can rotate keys mid-session without a
  // reload. `on401` opens the drawer.
  useEffect(() => {
    installApiHooks({
      apiKey: () => getApiKey(),
      onRequestId: (rid) => {
        setLastRequestIdState(rid);
        setLastRequestId(rid);
      },
      on401: () => {
        setDrawerOpen(true);
      },
      // Cold-start hooks — surface a warming-up banner when fetches
      // fail with network error or 5xx (Render free-tier spin-down).
      onBackendUnavailable: (reason) => {
        setBackendUnavailableReason(reason);
        setBackendUnavailable(true);
      },
      onBackendRecovered: () => {
        setBackendUnavailable(false);
      },
    });
  }, []);

  useEffect(() => {
    // Arm the cold-start banner if /health hasn't answered in 2 s —
    // typical for a Render free-tier dyno that just spun back up.
    // The banner auto-dismisses when the next fetch succeeds.
    let armed = false;
    const t = window.setTimeout(() => {
      armed = true;
      setBackendUnavailableReason("initial /health probe > 2 s");
      setBackendUnavailable(true);
    }, 2000);

    getHealth().then((h) => {
      window.clearTimeout(t);
      if (armed) setBackendUnavailable(false);
      setHealth(h);
      // If the API reports DEMO_MODE, switch landing tab to Demo Samples
      // exactly once. Anything the operator does after that overrides.
      if (h?.demo_mode) setTab("demo");
    }).catch(() => {
      window.clearTimeout(t);
      setHealth(null);
    });
    listModelCards().then(setCards).catch(() => setCards(null));
  }, []);

  const demoMode = !!health?.demo_mode;
  const contactUrl = health?.contact_url ?? "https://crispro.ai/contact";

  const cancerCaps = useMemo(() => health?.cancers ?? {}, [health]);

  // If the operator switches cancer to a non-breast, jump their view to
  // that cancer's tab immediately. Keeps the two selectors (drawer +
  // header pill) in sync.
  function handleCancerChange(c: CancerId) {
    setCancerState(c);
    persistCancer(c);
    if (c === "nsclc") setTab("nsclc");
    if (c === "breast" && tab === "nsclc") setTab("case");
  }

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "1.5rem" }}>
      <header style={{ marginBottom: "1rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: "1rem" }}>
          <div>
            <h1 style={{ margin: 0 }}>Oncology Arbiter</h1>
            <div style={{ color: "var(--fg-muted)", fontSize: "0.8rem" }}>
              Multi-cancer decision-support · Research Use Only (not for diagnosis)
              {health && <span> · v{health.version} · {health.status}</span>}
            </div>
          </div>
          <nav className="tabs">
            {demoMode && (
              <button className={tab === "demo" ? "active" : ""} onClick={() => setTab("demo")} data-testid="tab-demo-samples">Demo samples</button>
            )}
            <button className={tab === "screening" ? "active" : ""} onClick={() => setTab("screening")}>Screening</button>
            <button className={tab === "biopsy" ? "active" : ""} onClick={() => setTab("biopsy")}>Biopsy</button>
            <button className={tab === "therapy" ? "active" : ""} onClick={() => setTab("therapy")}>Therapy</button>
            <button className={tab === "case" ? "active" : ""} onClick={() => setTab("case")}>Case view</button>
            <button className={tab === "nsclc" ? "active" : ""} onClick={() => setTab("nsclc")}>NSCLC</button>
            <button className={tab === "cards" ? "active" : ""} onClick={() => setTab("cards")}>Model cards</button>
            <button
              onClick={() => setDrawerOpen(true)}
              title="API key, cancer selector, last request id"
              aria-label="Open settings"
            >
              ⚙︎ Settings
            </button>
          </nav>
        </div>
        <div style={{ marginTop: "0.5rem", display: "flex", flexWrap: "wrap", gap: "0.5rem", alignItems: "center" }}>
          <span className="pill" style={{ fontWeight: 600 }}>cancer: {cancer}</span>
          {lastRequestId && (
            <span className="pill" style={{ fontFamily: "Menlo, monospace", fontSize: "0.7rem" }}>
              req: {lastRequestId}
            </span>
          )}
          {health && (
            <HonestyPills
              models={health.models_loaded || {}}
              demoMode={demoMode}
              contactUrl={contactUrl}
            />
          )}
        </div>
      </header>

      {/* Cold-start banner: warms visitors that a Render free-tier
          demo is spinning up. Auto-dismisses on the next successful
          fetch. See ColdStartBanner.tsx for the state machine. */}
      <ColdStartBanner unavailable={backendUnavailable} reason={backendUnavailableReason} />

      {/* Demo Samples tab: only rendered when server reports demo_mode.
          It is the primary landing surface for public demo deployments. */}
      {tab === "demo" && demoMode && <DemoSamplesTab />}

      {/* Live workflow tabs. In DEMO_MODE these render a read-only
          placeholder that routes visitors to the contact page and the
          Demo Samples tab (which has real pre-computed outputs). This
          matches the API behaviour where POSTs return 403. */}
      {tab === "screening" && (
        demoMode
          ? <DemoModePlaceholder tabLabel="Screening" contactUrl={contactUrl} demoSampleKind="screening" onOpenSamplesTab={() => setTab("demo")} />
          : <ScreeningTab />
      )}
      {tab === "biopsy" && (
        demoMode
          ? <DemoModePlaceholder tabLabel="Biopsy" contactUrl={contactUrl} demoSampleKind="biopsy" onOpenSamplesTab={() => setTab("demo")} />
          : <BiopsyTab />
      )}
      {tab === "therapy" && (
        demoMode
          ? <DemoModePlaceholder tabLabel="Therapy" contactUrl={contactUrl} demoSampleKind="case_full" onOpenSamplesTab={() => setTab("demo")} />
          : <TherapyTab />
      )}
      {tab === "case" && (
        demoMode
          ? <DemoModePlaceholder tabLabel="Case view" contactUrl={contactUrl} demoSampleKind="case_full" onOpenSamplesTab={() => setTab("demo")} />
          : <CaseViewTab />
      )}
      {tab === "nsclc" && (
        demoMode
          ? <DemoModePlaceholder tabLabel="NSCLC" contactUrl={contactUrl} demoSampleKind="nsclc" onOpenSamplesTab={() => setTab("demo")} />
          : <NsclcTab />
      )}
      {tab === "cards" && (
        <div className="card">
          <h2>Model cards</h2>
          <p style={{ fontSize: "0.85rem", color: "var(--fg-muted)" }}>
            Every stage model ships an RUO-disclaimered model card. Cards flag
            proxy paths, synthetic weights, and gated repos so a reader cannot
            confuse a heuristic result with a live inference.
          </p>
          {!cards && <div style={{ color: "var(--fg-muted)" }}>Loading…</div>}
          {cards && cards.cards.length === 0 && <div>No cards found.</div>}
          {cards && (
            <table style={{ width: "100%", fontSize: "0.85rem" }}>
              <thead><tr>
                <th style={{ textAlign: "left" }}>slug</th>
                <th style={{ textAlign: "left" }}>title</th>
                <th style={{ textAlign: "right" }}>bytes</th>
                <th>markers</th>
              </tr></thead>
              <tbody>
                {cards.cards.map((c) => (
                  <tr key={c.slug}>
                    <td>
                      <a href={`/v1/artifacts/docs/${c.slug}.md`} target="_blank" rel="noreferrer"
                         style={{ color: "var(--accent)" }}>{c.slug}</a>
                    </td>
                    <td>{c.title}</td>
                    <td style={{ textAlign: "right", fontFamily: "Menlo, monospace" }}>{c.n_bytes}</td>
                    <td style={{ fontSize: "0.75rem" }}>
                      {Object.entries(c.honesty_markers).filter(([, v]) => v).map(([k]) => (
                        <span key={k} className="pill" style={{ marginRight: "0.25rem" }}>{k}</span>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      <footer style={{ marginTop: "2rem", color: "var(--fg-muted)", fontSize: "0.75rem", borderTop: "1px solid var(--border)", paddingTop: "0.75rem" }}>
        Research use only. Not a diagnostic device. See model cards for provenance,
        weights loading state, and known failure modes.
      </footer>

      <SettingsDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        onCancerChange={handleCancerChange}
        cancers={cancerCaps as Record<string, { state: string; case_full: boolean }>}
        lastRequestId={lastRequestId}
      />
    </div>
  );
}

export default App;
