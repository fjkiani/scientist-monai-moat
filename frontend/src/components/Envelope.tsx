import type { Envelope, ArbiterScore } from "../api";

interface EnvelopeViewProps {
  env: Envelope;
  arbiter?: ArbiterScore | null;
}

export function ProvenancePill({ state, name }: { state: string; name?: string | null }) {
  return (
    <div style={{ marginBottom: "0.75rem" }}>
      <span className={`pill ${state}`}>{state}</span>
      {name && <span className="pill">{name}</span>}
    </div>
  );
}

export function WarningsList({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) return null;
  return (
    <div style={{ marginBottom: "0.75rem" }}>
      <strong style={{ fontSize: "0.85rem" }}>Honesty warnings ({warnings.length})</strong>
      {warnings.map((w, i) => (
        <div className="warning" key={i}>{w}</div>
      ))}
    </div>
  );
}

export function EvidenceList({ evidence }: { evidence: Envelope["evidence"] }) {
  if (!evidence.length) return null;
  return (
    <div style={{ marginBottom: "0.75rem" }}>
      <strong style={{ fontSize: "0.85rem" }}>Evidence ({evidence.length})</strong>
      {evidence.map((e, i) => (
        <div className="evidence-item" key={i}>
          <a href={e.url} target="_blank" rel="noopener noreferrer">{e.url}</a>
          <div style={{ fontSize: "0.8rem", color: "var(--fg-muted)" }}>
            {e.source}: {e.quoted_text}
          </div>
        </div>
      ))}
    </div>
  );
}

export function ArbiterScoreView({ arb }: { arb: ArbiterScore | null | undefined }) {
  if (!arb) return null;
  const bucketColor: Record<string, string> = { LOW: "#75A025", MID: "#FF9400", HIGH: "#E9ED4C" };
  return (
    <div className="card">
      <h2>L3 arbiter · {arb.model_name}</h2>
      <div style={{ display: "flex", gap: "1rem", alignItems: "center", flexWrap: "wrap" }}>
        <div>
          <div style={{ color: "var(--fg-muted)", fontSize: "0.75rem" }}>P(positive)</div>
          <div style={{ fontSize: "1.5rem", fontWeight: 600 }}>{(arb.p_positive * 100).toFixed(1)}%</div>
        </div>
        <div>
          <div style={{ color: "var(--fg-muted)", fontSize: "0.75rem" }}>Risk bucket</div>
          <div style={{
            background: bucketColor[arb.risk_bucket] || "gray",
            color: arb.risk_bucket === "HIGH" ? "black" : "white",
            padding: "0.25rem 0.75rem", borderRadius: "4px", fontWeight: 600,
          }}>{arb.risk_bucket}</div>
        </div>
        <div>
          <div style={{ color: "var(--fg-muted)", fontSize: "0.75rem" }}>Driving feature</div>
          <div>{arb.driving_feature} ({arb.driving_feature_contribution.toFixed(3)})</div>
        </div>
        <div>
          <div style={{ color: "var(--fg-muted)", fontSize: "0.75rem" }}>n_training</div>
          <div>{arb.n_training} <span style={{ color: "var(--fg-muted)" }}>({arb.model_state})</span></div>
        </div>
      </div>
      <div style={{ marginTop: "0.75rem", fontSize: "0.85rem" }}>
        <strong>Recommendation:</strong> {arb.recommendation}
      </div>
      <details style={{ marginTop: "0.5rem", fontSize: "0.8rem" }}>
        <summary>Term contributions</summary>
        <pre>{JSON.stringify(arb.term_contributions, null, 2)}</pre>
      </details>
    </div>
  );
}

export function EnvelopeCard({ env, arbiter }: EnvelopeViewProps) {
  return (
    <>
      <div className="card">
        <h2>Envelope · request {env.provenance.request_id}</h2>
        <ProvenancePill state={env.provenance.model_state} name={env.provenance.model_name} />
        <WarningsList warnings={env.warnings} />
        <EvidenceList evidence={env.evidence} />
        <details style={{ fontSize: "0.8rem", color: "var(--fg-muted)" }}>
          <summary>Disclaimer & caveat</summary>
          <div style={{ marginTop: "0.5rem" }}>{env.disclaimer}</div>
          <div style={{ marginTop: "0.5rem" }}>{env.caveat}</div>
        </details>
      </div>
      <ArbiterScoreView arb={arbiter} />
    </>
  );
}
