# Tumor Board (CrisPRO Oncology Arbiter) — Frontend Package

**Contract:** `Tumor Board Case-Resolution Contract v1.0.0`
**Regulatory posture:** Research Use Only. Not FDA-cleared, not CE-marked, not for clinical use.
**Data posture:** All three cases are `EXAMPLE-DATA-RUO` fixtures. No real patient data.
**Generated:** 2026-07-18

This directory is a **frontend hand-off** — HTML mockups, wireframe PNGs, and a JSON payload/component spec for the Tumor Board case-resolution view. It is **not** a live frontend. The design system team owns final visual design, tokens, and framework integration.

---

## 1. What's here

```
tumor_board_frontend/
├── README.md                          ← this file
├── contract_snapshot.json             ← contract v1.0.0 (verbatim)
├── quarantine_test_harness.json       ← forbidden substrings, regex, required RUO tokens
├── component_spec.json                ← 10 components, 5-state matrix, palette, layout grid, CSS hooks, invariants
├── acceptance_tests.json              ← T1–T7 harness output (all green)
│
├── case_router/
│   ├── mockup.html                    ← landing / case selector (opens in any browser)
│   └── wireframe.png                  ← schematic wireframe of the router
│
└── cases/
    ├── hgsoc_mbd4_lof/                ← HGSOC + MBD4 p.R468* (loss-of-function)
    ├── tnbc_brca1_mut/                ← TNBC + BRCA1 c.68_69delAG germline
    └── nsclc_kras_g12c/               ← NSCLC + KRAS p.G12C somatic
        ├── l5_output.json             ← L5 arbiter payload (structured output contract)
        ├── mockup.html                ← full case view, 5 state tabs
        └── wireframes/
            ├── happy.png
            ├── empty.png
            ├── loading.png
            ├── degraded.png
            └── quarantined.png
```

## 2. How to open

- **Router / landing:** open `case_router/mockup.html` in a browser. Click a case card to open its mockup.
- **Case view:** open any `cases/<slug>/mockup.html`. Use the state tab bar at the top to switch between happy / empty / loading / degraded / quarantined.
- **Wireframes:** open the PNGs directly. They are grayscale schematic layouts intended as a design-system starting point, not final pixels.

Everything is self-contained: no build step, no external CDN, no JS framework. A single vanilla `tbShowTab()` handler swaps the state pane.

## 3. Per-file purpose

| File | Purpose |
|---|---|
| `contract_snapshot.json` | Frozen contract v1.0.0 the mockups implement. Treat as source of truth. |
| `quarantine_test_harness.json` | Forbidden substrings + regex + required RUO tokens the frontend must never regress on. Wire this into CI. |
| `component_spec.json` | 10 components (`RuoBanner`, `CaseHeader`, `MolecularContext`, `MechanismHypotheses`, `Vulnerabilities`, `TherapyTrialExploration`, `EvidenceTrace`, `SafetyGovernanceFlags`, `MissingDataAndConfidence`, `HumanDecisionFooter`), each with a 5-state UI matrix, CSS hooks, and layout grid. |
| `acceptance_tests.json` | T1–T7 harness results. Rerun after every change. |
| `cases/<slug>/l5_output.json` | The exact JSON shape the frontend should expect from L5. Bind components to these fields. |
| `cases/<slug>/mockup.html` | Reference implementation of the 3-column layout, all 5 states, on that case's payload. |
| `cases/<slug>/wireframes/*.png` | Schematic layouts labeled by block + state. Non-normative visuals. |
| `case_router/mockup.html` | Landing page with 3 case cards. |
| `case_router/wireframe.png` | Schematic of the router. |

## 4. Invariants the frontend team MUST preserve

These are contract-level. Breaking any of them is a regression.

1. **RUO banner is persistent** on every rendered view (case + router). Never hide, never collapse, never move below the fold. Must literally include `"Research Use Only"` and `"Not FDA-cleared"`.
2. **Human Decision Footer** renders on every case view with all four items (`Diagnosis`, `Treatment selection and prescribing`, `Trial enrollment decision`, `Interpretation of RUO/heuristic outputs`).
3. **Every claim-bearing block wrapper carries `data-ruo="true"`.** All 7 output-contract blocks are claim-bearing.
4. **`prohibited_in_output` is enforced textually.** The frontend must never render any string matching the forbidden substrings/regex in `quarantine_test_harness.json`. Specifically:
   - No `GBM ZEB1 as escape marker` / `ZEB1-escape` framing
   - No `LATIFY delta`, no `DDR 0.983` as case figures
   - No treatment directives (`prescribe`, `will respond to`, `should receive`, `patient-specific therapeutic`, etc.)
   - No individual outcome predictions
   - No synthetic lethality as a patient-specific therapeutic fact
5. **Degraded state is triggered by** `ranking_model.model_state == "proxy_co_scientist"` OR `ranking_model.n_training == 0`. Render the yellow "TEMPLATE / rules-lite" chip on `MechanismHypotheses` and `Vulnerabilities` when either is true.
6. **Quarantined state suppresses `therapy_trial_exploration`** (show suppression box, no agents), adds a "QUARANTINE FIRED" chip to `SafetyGovernanceFlags`, and appends `"A therapy claim was suppressed in this run."` to `quarantines_active`. All other blocks fall through to happy content.
7. **Every claim in the payload has non-null `evidence.entity_id` + `evidence.source_file`.** The frontend must render at least one visible evidence pointer per claim (the `EvidenceTrace` block is the canonical surface, but blocks may also inline a pointer).
8. **The 7 output-contract blocks always render (or explicitly render as empty/loading/degraded/quarantined).** Never silently drop a block — that hides a contract violation.

## 5. Payload shape (L5 output contract)

```jsonc
{
  "meta": {
    "contract_version": "1.0.0",
    "example_data_ruo": true,                       // MUST be true in every fixture here
    "pipeline_layers_traversed": ["L1","L2","L3","L4a","L4b","L4c","L5"]
  },
  "ruo_banner": { "text": "...", "regulatory_posture": "..." },
  "case_header": { "case_id": "...", "molecular_summary": "...", "clinical_summary": "..." },
  "blocks": {
    "molecular_context":              { ...variant, ...clinical, evidence: {...} },
    "mechanism_hypotheses":           { ranked: [...], ranking_model: {...} },
    "vulnerabilities_synthetic_lethality": { targets: [...] },
    "therapy_trial_exploration":      { agents: [...], trials: [...] },
    "evidence_trace":                 { entries: [{claim, entity_id, source_file}, ...] },
    "safety_governance_flags":        { flags: [...], quarantines_active: [...] },
    "missing_data_and_confidence":    { unknowns: [...], confidence_bars: [...] }
  },
  "human_decision_footer": {
    "text": "...",
    "remains_human": ["Diagnosis", "Treatment selection and prescribing",
                      "Trial enrollment decision", "Interpretation of RUO/heuristic outputs"]
  }
}
```

Bind components to these paths. Everything the mockups render comes from this shape.

## 6. Swapping example data for real payloads later

When you wire this to the live L5 arbiter output:

1. **Do not remove the RUO banner or footer** — they are contract-level UI, not a demo artifact.
2. **`meta.example_data_ruo` will flip to `false`** for real payloads — this must show a stronger production-data notice, not remove the RUO banner. (RUO is a regulatory posture, not a demo flag.)
3. **Keep `data-ruo="true"`** on every block wrapper regardless of `example_data_ruo`.
4. **Re-run the quarantine harness** in `quarantine_test_harness.json` on every rendered page before shipping. Failing a substring/regex check is a hard block.
5. **Re-run `acceptance_tests.json`** T1–T7 in CI on every payload/mockup change.
6. **Do not add features that predict individual patient outcomes** or emit treatment directives — even if the model output invites it. That crosses the human-decision boundary and violates the contract.

## 7. Design-system TODO for the frontend team

The mockups intentionally use a minimal inline style. Before production:

- Replace inline CSS with your design tokens (colors, spacing, type scale). Palette in `component_spec.json` uses Phylo colors (`#000000, #ECE9E2, #FAF9F3, #E9ED4C, #FF9400, #75A025, #FD9BED, #0279EE`) as a starting point.
- Extract the state-tab bar into a real component (mockup uses a single `tbShowTab` handler purely for hand-off demonstration).
- Add real focus states, keyboard nav, and ARIA labels — the mockup has minimal a11y.
- Add responsive breakpoints. Layout collapses at ≤900 px per `component_spec.json.layout.breakpoint`.
- Add real i18n for the RUO banner text — the exact string is contract-mandated in English; translations must be legally reviewed.

## 8. Acceptance test summary (`acceptance_tests.json`)

| Test | Result | What it checks |
|---|---|---|
| T1 | 3/3 | Every `output_contract.blocks[]` present in every case payload |
| T2 | 3/3 | Every `data-block=` node in mockups also has `data-ruo="true"` |
| T3 | 8/8 | Zero forbidden substrings/regex hits across all HTMLs + payloads + spec |
| T4 | 4/4 | All 4 `remains_human` items present in every case mockup + router |
| T5 | 3/3 | Every claim in every payload has non-null `evidence.entity_id` + `source_file` |
| T6 | 3/3 | Every case × state has exactly 7 `data-block=` nodes |
| T7 | 16/16 | All 16 wireframe PNGs exist and are non-empty; sampled media-checks pass |

Rerun with the harness in `acceptance_tests.json` any time you change a payload, mockup, or component spec.
