import { useState } from "react";
import { reasonTherapy, type TherapyResponse } from "../api";
import { EnvelopeCard } from "../components/Envelope";

const NCCN_BREAST_URL = "https://www.nccn.org/professionals/physician_gls/pdf/breast.pdf";

export function TherapyTab() {
  const [er, setEr] = useState<"positive" | "negative" | "unknown">("positive");
  const [pr, setPr] = useState<"positive" | "negative" | "unknown">("positive");
  const [her2, setHer2] = useState<"positive" | "negative" | "equivocal" | "unknown">("negative");
  const [grade, setGrade] = useState<1 | 2 | 3>(2);
  const [subtype, setSubtype] = useState<"IDC" | "DCIS" | "benign" | "">("IDC");
  const [confidence, setConfidence] = useState<number>(0.72);
  const [stage, setStage] = useState<string>("T1N0M0");
  const [menopausal, setMenopausal] = useState<"pre" | "post" | "peri" | "unknown">("post");
  const [age, setAge] = useState<number>(58);

  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<TherapyResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function submit() {
    setBusy(true); setErr(null); setResult(null);
    try {
      const boolMap = (v: "positive" | "negative" | "unknown") =>
        v === "positive" ? true : v === "negative" ? false : null;
      const her2Map = her2 === "unknown" ? null : her2;
      const body = {
        biopsy_output: {
          disclaimer: "",
          caveat: "",
          provenance: { model_state: "loaded_biopsy_probe", request_id: "ui-supplied" },
          honesty_gate: { seen_urls_count: 0, evidence_kept: 0, evidence_dropped: 0 },
          evidence: [],
          warnings: [],
          subtype_prediction: subtype || null,
          confidence,
          grade,
          receptor_panel: {
            er_positive: boolMap(er),
            pr_positive: boolMap(pr),
            her2_status: her2Map,
            ki67_percent: null,
          },
        },
        patient_context: {
          age,
          menopausal_status: menopausal,
          prior_therapies: [],
          comorbidities: [],
          genomic_markers: { stage },
        },
      };
      const r = await reasonTherapy(body);
      setResult(r);
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  }

  return (
    <>
      <div
        className="card"
        data-testid="therapy-caveat-banner"
        style={{
          background: "#fef3c7",
          borderLeft: "4px solid #f59e0b",
          color: "#78350f",
        }}
      >
        <div style={{ fontWeight: 600, fontSize: "0.95rem" }}>
          ⚠ Research use only — nccn-lite-v0 proxy rules engine
        </div>
        <div style={{ fontSize: "0.85rem", marginTop: "0.35rem" }}>
          This tab maps ER/PR/HER2/grade to a small fixed table of published NCCN
          Guideline sections. It is NOT a trained ML model, NOT a live TxGemma agent,
          and NOT a certified clinical decision support tool. It does not reason about
          individual comorbidities, drug interactions, prior therapy, or trial
          enrollment. Do not use for treatment decisions. Every recommendation links
          back to its NCCN section — verify against the full guideline (
          <a href={NCCN_BREAST_URL} target="_blank" rel="noreferrer"
             style={{ color: "#78350f", textDecoration: "underline" }}>
            NCCN Breast Cancer Guidelines PDF
          </a>
          ) with a certified breast oncologist before any clinical action.
        </div>
      </div>

      <div className="card">
        <h2>Therapy · NCCN-lite rules + TxGemma (gated)</h2>
        <p style={{ fontSize: "0.85rem", color: "var(--fg-muted)" }}>
          Precedence: TxGemma-9b-chat if HAI-DEF unlocks the repo → 5-branch NCCN-lite
          rules proxy → PLACEHOLDER. The proxy honesty warning surfaces on every
          non-gated response.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.75rem" }}>
          <div>
            <label>ER</label>
            <select value={er} onChange={(e) => setEr(e.target.value as any)}>
              <option>positive</option><option>negative</option><option>unknown</option>
            </select>
          </div>
          <div>
            <label>PR</label>
            <select value={pr} onChange={(e) => setPr(e.target.value as any)}>
              <option>positive</option><option>negative</option><option>unknown</option>
            </select>
          </div>
          <div>
            <label>HER2</label>
            <select value={her2} onChange={(e) => setHer2(e.target.value as any)}>
              <option>positive</option><option>negative</option>
              <option>equivocal</option><option>unknown</option>
            </select>
          </div>
          <div>
            <label>Grade</label>
            <select value={grade} onChange={(e) => setGrade(Number(e.target.value) as any)}>
              <option value={1}>1</option><option value={2}>2</option><option value={3}>3</option>
            </select>
          </div>
          <div>
            <label>Subtype</label>
            <select value={subtype} onChange={(e) => setSubtype(e.target.value as any)}>
              <option value="IDC">IDC</option>
              <option value="DCIS">DCIS</option>
              <option value="benign">benign</option>
              <option value="">unknown</option>
            </select>
          </div>
          <div>
            <label>Subtype confidence</label>
            <input type="number" step="0.05" min={0} max={1} value={confidence}
                   onChange={(e) => setConfidence(Number(e.target.value))} />
          </div>
          <div>
            <label>Stage (T/N/M)</label>
            <input value={stage} onChange={(e) => setStage(e.target.value)} placeholder="T1N0M0" />
          </div>
          <div>
            <label>Menopausal status</label>
            <select value={menopausal} onChange={(e) => setMenopausal(e.target.value as any)}>
              <option>pre</option><option>post</option><option>peri</option><option>unknown</option>
            </select>
          </div>
          <div>
            <label>Age</label>
            <input type="number" min={18} max={120} value={age}
                   onChange={(e) => setAge(Number(e.target.value))} />
          </div>
        </div>

        <button className="primary" onClick={submit} disabled={busy} style={{ marginTop: "0.75rem" }}>
          {busy ? "Reasoning…" : "Recommend therapy"}
        </button>
        {err && <div className="warning" style={{ marginTop: "0.75rem" }}>{err}</div>}
      </div>

      {result && (
        <>
          <div className="card">
            <div style={{
              display: "flex",
              flexWrap: "wrap",
              gap: "0.5rem",
              alignItems: "center",
              marginBottom: "0.5rem",
            }} data-testid="therapy-provenance-bar">
              <span className="pill" style={{ background: "#eef2ff", color: "#3730a3" }}>
                model: {result.provenance?.model_name ?? result.provenance?.model_state ?? "unknown"}
              </span>
              {result.branch_id && (
                <span className="pill" style={{ background: "#ecfdf5", color: "#065f46" }}>
                  branch: {result.branch_id}
                </span>
              )}
              {result.rules_sha256 && (
                <span className="pill" style={{ background: "#f5f5f4", color: "#57534e", fontFamily: "ui-monospace, monospace" }}>
                  rules sha256: {result.rules_sha256.slice(0, 12)}…
                </span>
              )}
            </div>
            <h2>Recommended options ({result.recommended_options.length})</h2>
            {result.recommended_options.length === 0 && (
              <div style={{ color: "var(--fg-muted)" }}>No options returned (placeholder or gated).</div>
            )}
            {result.recommended_options.map((opt, i) => (
              <div key={i} style={{
                borderTop: i === 0 ? "none" : "1px solid var(--border)",
                paddingTop: i === 0 ? 0 : "0.5rem",
                marginTop: "0.5rem",
              }}>
                <div style={{ fontWeight: 600 }}>{opt.regimen}</div>
                <div style={{ fontSize: "0.75rem", color: "var(--fg-muted)" }}>
                  <span className="pill">line {opt.line_of_therapy}</span>
                  {opt.evidence && opt.evidence.length > 0 && (
                    <span style={{ marginLeft: "0.5rem" }} data-testid={`nccn-links-${i}`}>
                      {opt.evidence.map((ev, j) => (
                        <a
                          key={j}
                          href={ev.url || NCCN_BREAST_URL}
                          target="_blank"
                          rel="noreferrer"
                          style={{
                            color: "var(--accent)",
                            marginRight: "0.5rem",
                            textDecoration: "underline",
                          }}
                          title={ev.source || "NCCN Breast Cancer Guidelines"}
                        >
                          {ev.quoted_text || ev.source || "NCCN evidence"}
                        </a>
                      ))}
                    </span>
                  )}
                </div>
                <div style={{ fontSize: "0.85rem", marginTop: "0.25rem" }}>{opt.rationale}</div>
                {opt.contraindications.length > 0 && (
                  <div style={{ fontSize: "0.8rem", color: "var(--warn)", marginTop: "0.25rem" }}>
                    Contraindications: {opt.contraindications.join(", ")}
                  </div>
                )}
              </div>
            ))}
            {result.not_recommended.length > 0 && (
              <>
                <h3 style={{ marginTop: "1rem" }}>Not recommended</h3>
                <ul>
                  {result.not_recommended.map((opt, i) => (
                    <li key={i}><strong>{opt.regimen}</strong> — {opt.rationale}</li>
                  ))}
                </ul>
              </>
            )}
          </div>
          <EnvelopeCard env={result} arbiter={result.arbiter_score} />
        </>
      )}
    </>
  );
}
