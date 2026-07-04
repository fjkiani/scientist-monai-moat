import { useState } from "react";
import { runCaseFull, type FullCaseResponse } from "../api";
import { EnvelopeCard } from "../components/Envelope";

/**
 * NSCLC panel — shape-only placeholder that hits
 * `/v1/case/full?cancer=nsclc`. Today the backend returns a placeholder
 * envelope (no screening/biopsy/therapy sub-stages, one warning that
 * flags the placeholder status).
 *
 * When worker-2 lands the LIDC-IDRI pipeline this panel gains:
 *   - CT DICOM series uploader (multiple .dcm files, one series)
 *   - MONAI nodule heuristic overlay
 *   - NSCLC arbiter score (T/N/M, EGFR/KRAS/ALK, PD-L1)
 *   - NCCN-NSCLC recommended regimens
 *
 * Until then this tab exists on purpose: it proves the cancer selector,
 * the `?cancer=` query param, and the honesty envelope all round-trip on
 * a non-breast route.
 */
export function NsclcTab() {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<FullCaseResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function submit() {
    setBusy(true); setErr(null); setResult(null);
    try {
      // Empty body — the placeholder branch doesn't consume screening/biopsy inputs yet.
      const r = await runCaseFull("nsclc", {});
      setResult(r);
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  }

  return (
    <>
      <div className="card">
        <h2>NSCLC · placeholder (LIDC-IDRI pipeline landing from worker-2)</h2>
        <p style={{ fontSize: "0.85rem", color: "var(--fg-muted)" }}>
          Cancer track: <code>nsclc</code>. Sends <code>POST /v1/case/full?cancer=nsclc</code>.
          This panel is intentionally minimal — it proves the cancer selector,
          the query param, and the honesty envelope round-trip on a non-breast
          route. Real inputs (CT DICOM series, EGFR/PD-L1 fields) arrive when
          the LIDC pipeline lands.
        </p>

        <button className="primary" onClick={submit} disabled={busy} style={{ marginTop: "0.5rem" }}>
          {busy ? "Running…" : "Run NSCLC placeholder"}
        </button>
        {err && <div className="warning" style={{ marginTop: "0.75rem" }}>{err}</div>}
      </div>

      {result && (
        <>
          <div className="card">
            <h2>NSCLC response summary</h2>
            <div>
              <span className={`pill ${result.provenance.model_state}`}>
                {result.provenance.model_state}
              </span>
              <span className="pill">
                model: {result.provenance.model_name ?? "—"}
              </span>
            </div>
            <div style={{ marginTop: "0.5rem", fontSize: "0.85rem" }}>
              screening: <code>{result.screening ? "…" : "null"}</code>
              {" · "}
              biopsy: <code>{result.biopsy ? "…" : "null"}</code>
              {" · "}
              therapy: <code>{result.therapy ? "…" : "null"}</code>
              {" · "}
              elo: <code>{result.elo_ranked_hypotheses.length}</code>
            </div>
          </div>
          <EnvelopeCard env={result} />
        </>
      )}
    </>
  );
}
