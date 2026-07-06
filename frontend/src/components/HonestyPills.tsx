// v0.2.2: Header pill row showing each model slot's live state.
//
// Each pill is a button; clicking opens a popover explaining what the state
// actually means (e.g. "proxy_rules_lite: NCCN-lite rules deterministic
// path; TxGemma weights not loaded"). This turns /health from a wall of
// jargon into a first-time user tour of the honesty system.

import { useEffect, useRef, useState } from "react";
import type { ModelState } from "../api";

// Human-readable explanations. Keys are ModelState values from api.ts.
// Keep in lockstep with schemas.ModelState and _compute_models_loaded().
const STATE_EXPLAIN: Record<string, { title: string; body: string; severity: "loaded" | "proxy" | "template" | "off" }> = {
  placeholder: {
    title: "Placeholder",
    body: "No model is wired for this slot yet. Requests will fall back to the API's degraded path and return an honesty warning.",
    severity: "off",
  },
  loaded: {
    title: "Loaded (in memory)",
    body: "A real model is loaded and performs inference for this slot.",
    severity: "loaded",
  },
  loaded_medsiglip: {
    title: "MedSigLIP (HAI-DEF)",
    body: "Google MedSigLIP-448 image encoder is loaded and used for zero-shot mammography scoring. Requires an accepted HAI-DEF terms-of-use.",
    severity: "loaded",
  },
  loaded_biopsy_probe: {
    title: "Biopsy linear probe (synthetic)",
    body: "MedSigLIP image encoder + a synthetic-trained linear classifier over receptor labels. Numbers should be treated as a proxy, not evidence.",
    severity: "loaded",
  },
  loaded_monai_detector: {
    title: "MONAI detector (trained)",
    body: "A trained MONAI detection model is loaded. Not the current alpha config.",
    severity: "loaded",
  },
  loaded_txgemma: {
    title: "TxGemma (HAI-DEF)",
    body: "Google TxGemma-9B therapy reasoner. Requires an accepted HAI-DEF terms-of-use.",
    severity: "loaded",
  },
  proxy_siglip: {
    title: "General-domain SigLIP proxy",
    body: "Standard SigLIP image encoder (no medical fine-tuning). Ships as an opt-in fallback when HAI-DEF MedSigLIP is not available.",
    severity: "proxy",
  },
  proxy_monai_heuristic: {
    title: "MONAI mask-gradient heuristic",
    body: "Deterministic heuristic over the MONAI backbone — NOT a trained detector. Bounding boxes are illustrative.",
    severity: "proxy",
  },
  proxy_rules_lite: {
    title: "NCCN-lite deterministic rules",
    body: "Hand-written NCCN-lite rules engine (hormone-receptor + HER2 → therapy branch). No LLM in the loop; SHA-pinned rules file.",
    severity: "proxy",
  },
  proxy_lung_heuristic: {
    title: "NSCLC HU-threshold heuristic",
    body: "Simple Hounsfield-unit threshold + connected-components on the lung CT. Placeholder until a real NSCLC detector lands.",
    severity: "proxy",
  },
  proxy_regex_v0: {
    title: "Regex pathology parser",
    body: "Deterministic regex over the pathology report to pull ER/PR/HER2/Ki-67. Stateless code, always available. This is what actually populates the receptor panel today.",
    severity: "proxy",
  },
  proxy_co_scientist: {
    title: "Co-Scientist Elo tournament",
    body: "Deterministic Elo scoring over stage hypotheses. Not an LLM — a scoring function. Ranks candidate hypotheses so the arbiter's decision is auditable.",
    severity: "proxy",
  },
  template: {
    title: "L3 arbiter templates",
    body: "JSON templates on disk with n_training=0. The arbiter chooses a template, it doesn't learn one. Marked template until a fitted logistic replaces the templates.",
    severity: "template",
  },
  loading: { title: "Loading", body: "Model is warming up. Requests may fail with 503.", severity: "off" },
  unavailable: { title: "Unavailable", body: "Model failed to load. Requests return 5xx.", severity: "off" },
  gated: { title: "Gated", body: "HAI-DEF terms-of-use not accepted for this repository. See /v1/model-cards for the auth path.", severity: "off" },
  cached: { title: "Cached", body: "Response served from cache — no inference this request.", severity: "loaded" },
};

const SLOT_LABELS: Record<string, string> = {
  monai_screening: "screening",
  medsiglip_biopsy: "biopsy classifier",
  biopsy_report_parser: "biopsy parser",
  txgemma_therapy: "therapy",
  co_scientist: "co-scientist",
  l3_arbiter: "arbiter",
  nsclc_pipeline: "nsclc",
};

export interface HonestyPillsProps {
  models: Partial<Record<string, ModelState>>;
}

export function HonestyPills({ models }: HonestyPillsProps) {
  const [openSlot, setOpenSlot] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Close popover on outside click.
  useEffect(() => {
    if (!openSlot) return;
    function onDoc(e: MouseEvent) {
      if (!containerRef.current) return;
      if (!containerRef.current.contains(e.target as Node)) {
        setOpenSlot(null);
      }
    }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [openSlot]);

  // Close on Escape.
  useEffect(() => {
    if (!openSlot) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpenSlot(null);
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [openSlot]);

  const entries = Object.entries(models).filter(([, v]) => !!v) as [string, ModelState][];
  if (entries.length === 0) return null;

  return (
    <div ref={containerRef} style={{ position: "relative", display: "inline-flex", flexWrap: "wrap", gap: "0.5rem" }}>
      {entries.map(([slot, state]) => {
        const label = SLOT_LABELS[slot] ?? slot;
        const explain = STATE_EXPLAIN[state] ?? {
          title: state,
          body: `Unknown model state ${state}. This is a frontend/backend drift — please file a bug.`,
          severity: "off" as const,
        };
        const isOpen = openSlot === slot;
        return (
          <div key={slot} style={{ position: "relative" }}>
            <button
              className={`pill ${state}`}
              onClick={() => setOpenSlot(isOpen ? null : slot)}
              aria-expanded={isOpen}
              aria-haspopup="dialog"
              title={`${label}: ${state} — click to inspect`}
              data-testid={`honesty-pill-${slot}`}
            >
              {label}: {state}
            </button>
            {isOpen && (
              <div
                role="dialog"
                aria-label={`${label} state details`}
                className="honesty-popover"
                style={{ top: "calc(100% + 6px)", left: 0 }}
                data-testid={`honesty-popover-${slot}`}
              >
                <h4>
                  {label}{" "}
                  <span className="badge">{state}</span>
                </h4>
                <div style={{ fontWeight: 600, marginBottom: "0.25rem" }}>{explain.title}</div>
                <div style={{ color: "var(--fg-muted)" }}>{explain.body}</div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
