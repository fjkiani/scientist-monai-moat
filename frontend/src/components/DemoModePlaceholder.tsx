// v0.3.0-alpha: shared placeholder shown on Screening / Biopsy / Therapy /
// NSCLC / Case tabs when the deployment is in DEMO_MODE. The three raw
// endpoints on this public showcase deployment are POST-blocked (HTTP
// 403); the frontend routes anyone who wants to run the API on their own
// data to the contact URL surfaced on /health.
//
// The Demo Samples tab (see tabs/DemoSamplesTab.tsx) exposes the pre-
// computed real outputs so a visitor can still see what a live inference
// looks like end to end.

interface DemoModePlaceholderProps {
  tabLabel: string;                 // e.g. "Screening"
  contactUrl: string;               // /health.contact_url
  demoSampleKind?:                  // link to a matching pre-computed sample
    | "screening" | "biopsy" | "case_full" | "nsclc";
  onOpenSamplesTab?: () => void;    // if provided, wires "View sample" button
}

export function DemoModePlaceholder({
  tabLabel,
  contactUrl,
  demoSampleKind,
  onOpenSamplesTab,
}: DemoModePlaceholderProps) {
  return (
    <div className="card" style={{ borderLeft: "3px solid var(--accent)" }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: "0.75rem", flexWrap: "wrap" }}>
        <h2 style={{ margin: 0 }}>{tabLabel}</h2>
        <span className="pill" style={{ background: "#FF9400", color: "#000" }}>demo deployment · read-only</span>
      </div>

      <p style={{ marginTop: "0.75rem" }}>
        Live inference for {tabLabel.toLowerCase()} is disabled on this public
        showcase. It runs the real Oncology Arbiter v0.3.0-alpha stack —
        MedSigLIP-448 on Modal, ClinicalBERT NER, MONAI LUNA16 RetinaNet,
        NCCN-lite rules, and a Co-Scientist Elo loop — but firing every
        anonymous caller through the real weights would burn GPU credit and
        misrepresent throughput.
      </p>

      <p style={{ marginTop: "0.5rem" }}>
        To see the exact envelope this endpoint returns on real data —
        including the DICOM sha256, latency, and provenance for every
        weight loaded — open the <strong>Demo Samples</strong> tab. Each
        sample is a captured response from our workers with the same
        pipeline you would hit if the endpoint were live.
      </p>

      <div style={{ display: "flex", gap: "0.75rem", marginTop: "1rem", flexWrap: "wrap" }}>
        {onOpenSamplesTab && (
          <button
            className="primary"
            onClick={onOpenSamplesTab}
            data-testid={`demo-placeholder-view-sample-${demoSampleKind ?? "all"}`}
          >
            {demoSampleKind
              ? `View pre-computed ${demoSampleKind} sample →`
              : "View demo samples →"}
          </button>
        )}
        <a
          href={contactUrl}
          target="_blank"
          rel="noreferrer"
          className="button-like"
          style={{
            padding: "0.5rem 0.9rem",
            borderRadius: 4,
            border: "1px solid var(--accent)",
            color: "var(--accent)",
            textDecoration: "none",
            fontWeight: 600,
          }}
          data-testid={`demo-placeholder-contact-${demoSampleKind ?? "all"}`}
        >
          Run on your own data → Contact
        </a>
      </div>

      <details style={{ marginTop: "1rem", fontSize: "0.85rem", color: "var(--fg-muted)" }}>
        <summary>What is running under the hood?</summary>
        <ul style={{ marginTop: "0.5rem" }}>
          <li>Screening: MedSigLIP-448 on Modal (GPU) → CBIS-DDSM logreg probe (test AUC=0.7526 on n=641).</li>
          <li>Biopsy: ClinicalBERT NER + regex fusion. Micro relaxed F1=0.9546, strict F1=0.8733 on the synthetic held-out test set.</li>
          <li>NSCLC: MONAI RetinaNet trained on LUNA16 fold 0 (mAP=0.852, mAR=0.998).</li>
          <li>Therapy: NCCN-lite deterministic rules keyed on receptors + menopausal status.</li>
          <li>Co-Scientist: generate → reflect → rank(Elo) → evolve → rank, with a URL-honesty gate that strips any hypothesis quoting an unseen URL.</li>
        </ul>
      </details>
    </div>
  );
}
