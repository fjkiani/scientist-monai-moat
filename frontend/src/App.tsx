import { useEffect, useState } from "react";
import { ScreeningTab } from "./tabs/ScreeningTab";
import { BiopsyTab } from "./tabs/BiopsyTab";
import { TherapyTab } from "./tabs/TherapyTab";
import { CaseViewTab } from "./tabs/CaseViewTab";
import { getHealth, listModelCards, type HealthResponse, type ModelCardsIndex } from "./api";

type Tab = "screening" | "biopsy" | "therapy" | "case" | "cards";

export function App() {
  const [tab, setTab] = useState<Tab>("screening");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [cards, setCards] = useState<ModelCardsIndex | null>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
    listModelCards().then(setCards).catch(() => setCards(null));
  }, []);

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "1.5rem" }}>
      <header style={{ marginBottom: "1rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: "1rem" }}>
          <div>
            <h1 style={{ margin: 0 }}>Oncology Arbiter</h1>
            <div style={{ color: "var(--fg-muted)", fontSize: "0.8rem" }}>
              Breast cancer decision-support · Research Use Only (not for diagnosis)
              {health && <span> · v{health.version} · {health.status}</span>}
            </div>
          </div>
          <nav className="tabs">
            <button className={tab === "screening" ? "active" : ""} onClick={() => setTab("screening")}>Screening</button>
            <button className={tab === "biopsy" ? "active" : ""} onClick={() => setTab("biopsy")}>Biopsy</button>
            <button className={tab === "therapy" ? "active" : ""} onClick={() => setTab("therapy")}>Therapy</button>
            <button className={tab === "case" ? "active" : ""} onClick={() => setTab("case")}>Case view</button>
            <button className={tab === "cards" ? "active" : ""} onClick={() => setTab("cards")}>Model cards</button>
          </nav>
        </div>
        {health && Object.keys(health.models_loaded || {}).length > 0 && (
          <div style={{ marginTop: "0.5rem" }}>
            {Object.entries(health.models_loaded).map(([k, v]) => (
              <span key={k} className={`pill ${v}`}>{k}: {v}</span>
            ))}
          </div>
        )}
      </header>

      {tab === "screening" && <ScreeningTab />}
      {tab === "biopsy" && <BiopsyTab />}
      {tab === "therapy" && <TherapyTab />}
      {tab === "case" && <CaseViewTab />}
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
    </div>
  );
}

export default App;
