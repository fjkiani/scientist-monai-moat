/**
 * ReceptorPanelForm — the honesty-contract confirmation gate.
 *
 * Given the raw receptor_panel from /v1/biopsy/analyze (proxy_regex_v0), this
 * component:
 *
 *   1. Shows each of the 4 fields (ER, PR, HER2, grade) as a segmented pill
 *      control.
 *   2. Colors each pill by parse_state:
 *        matched       → green   ("regex confident")
 *        ambiguous     → orange  ("regex ambiguous — please confirm")
 *        no_match      → gray    ("regex found nothing — please enter")
 *        user_supplied → blue    ("you edited this")
 *   3. Pre-fills the current parser value where present; the clinician can
 *      change it. Any change flips the pill to user_supplied.
 *   4. Disables the "Confirm receptors → run therapy" button until ALL FOUR
 *      fields have a definitive value (no null).
 *   5. On Confirm, invokes onConfirm(panel, grade) with the confirmed values.
 *
 * This is deliberately strict: even a matched parser output is not treated as
 * clinically valid until the pathologist has clicked Confirm. There is no
 * "trust the parser" shortcut — the whole point of v0.2.1 is that a proxy
 * regex cannot drive therapy silently.
 */

import { useEffect, useMemo, useState } from "react";
import type { ParseStateValue, ReceptorPanel } from "../api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type Her2Value = "positive" | "negative" | "equivocal";

export interface ConfirmedPanel {
  er_positive: boolean;
  pr_positive: boolean;
  her2_status: Her2Value;
  grade: 1 | 2 | 3;
}

export interface ReceptorPanelFormProps {
  /** Backend response panel (proxy_regex_v0). Never mutated. */
  panel: ReceptorPanel;
  /** Grade extracted by the backend (separate field on BiopsyResponse). */
  parsedGrade: number | null;
  /** Called with the user-confirmed panel + grade when Confirm is clicked. */
  onConfirm: (confirmed: ConfirmedPanel) => void;
  /** True while a downstream therapy request is in flight. */
  busy?: boolean;
}

// ---------------------------------------------------------------------------
// Style helpers
// ---------------------------------------------------------------------------

function pillColor(state: ParseStateValue | null | undefined): string {
  switch (state) {
    case "matched":
      return "#16a34a"; // green
    case "ambiguous":
      return "#ea580c"; // orange
    case "no_match":
      return "#64748b"; // gray
    case "user_supplied":
      return "#0279EE"; // brand blue
    default:
      return "#64748b";
  }
}

function pillLabel(state: ParseStateValue | null | undefined): string {
  switch (state) {
    case "matched":
      return "parser: matched";
    case "ambiguous":
      return "parser: ambiguous";
    case "no_match":
      return "parser: no match";
    case "user_supplied":
      return "user-supplied";
    default:
      return "not parsed";
  }
}

const btnBase: React.CSSProperties = {
  padding: "0.35rem 0.75rem",
  border: "1px solid var(--border)",
  background: "white",
  cursor: "pointer",
  fontSize: "0.85rem",
  fontWeight: 500,
};

function segBtn(selected: boolean): React.CSSProperties {
  return {
    ...btnBase,
    background: selected ? "var(--fg)" : "white",
    color: selected ? "white" : "var(--fg)",
    borderColor: selected ? "var(--fg)" : "var(--border)",
  };
}

// ---------------------------------------------------------------------------
// Sub-component: labeled row with a segmented picker + provenance pill
// ---------------------------------------------------------------------------

interface RowProps<T> {
  label: string;
  options: Array<{ value: T; label: string }>;
  value: T | null;
  onChange: (v: T) => void;
  provenance: ParseStateValue | null | undefined;
}

function Row<T extends string | number | boolean>({
  label, options, value, onChange, provenance,
}: RowProps<T>) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "1rem",
                  marginBottom: "0.6rem", flexWrap: "wrap" }}>
      <div style={{ minWidth: "5rem", fontWeight: 600, fontSize: "0.9rem" }}>
        {label}
      </div>
      <div style={{ display: "flex", gap: 0, borderRadius: 6, overflow: "hidden" }}>
        {options.map((opt, i) => (
          <button
            key={String(opt.value)}
            type="button"
            onClick={() => onChange(opt.value)}
            style={{
              ...segBtn(value === opt.value),
              borderLeftWidth: i === 0 ? 1 : 0,
              borderRadius: 0,
            }}
          >
            {opt.label}
          </button>
        ))}
      </div>
      <span
        style={{
          fontSize: "0.7rem",
          padding: "0.15rem 0.5rem",
          borderRadius: 999,
          background: pillColor(provenance),
          color: "white",
          fontWeight: 500,
        }}
        data-testid={`parse-pill-${label.toLowerCase()}`}
      >
        {pillLabel(provenance)}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function ReceptorPanelForm({
  panel, parsedGrade, onConfirm, busy,
}: ReceptorPanelFormProps) {
  // Local editable copy of the panel. Every field carries its own provenance
  // state that starts at whatever the parser said and flips to user_supplied
  // on any user edit.
  const parsed = panel.parse_state ?? {
    er: "no_match", pr: "no_match", her2: "no_match", grade: "no_match",
  };

  const [er, setEr] = useState<boolean | null>(panel.er_positive);
  const [pr, setPr] = useState<boolean | null>(panel.pr_positive);
  const [her2, setHer2] = useState<Her2Value | null>(panel.her2_status);
  const [grade, setGrade] = useState<number | null>(parsedGrade);

  // Provenance can move from (matched|ambiguous|no_match) → user_supplied
  // when the local value diverges from the parser value.
  const [erState, setErState] = useState<ParseStateValue>(parsed.er);
  const [prState, setPrState] = useState<ParseStateValue>(parsed.pr);
  const [her2State, setHer2State] = useState<ParseStateValue>(parsed.her2);
  const [gradeState, setGradeState] = useState<ParseStateValue>(parsed.grade);

  // If the parent re-renders with a fresh panel (e.g. user hits Analyze
  // again with a different report), reset the local state.
  useEffect(() => {
    setEr(panel.er_positive);
    setPr(panel.pr_positive);
    setHer2(panel.her2_status);
    setGrade(parsedGrade);
    setErState(parsed.er);
    setPrState(parsed.pr);
    setHer2State(parsed.her2);
    setGradeState(parsed.grade);
    // We intentionally exclude parsed.* from the deps — panel + parsedGrade
    // are the source of truth, and parsed is derived synchronously above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [panel, parsedGrade]);

  const allFilled = useMemo(
    () => er !== null && pr !== null && her2 !== null && grade !== null,
    [er, pr, her2, grade],
  );

  function handleConfirm() {
    if (!allFilled) return;
    onConfirm({
      er_positive: er!,
      pr_positive: pr!,
      her2_status: her2!,
      grade: grade as 1 | 2 | 3,
    });
  }

  return (
    <div className="card" data-testid="receptor-panel-form">
      <h2>Confirm receptor panel</h2>
      <p style={{ fontSize: "0.85rem", color: "var(--fg-muted)" }}>
        Below is the panel extracted from the report by <code>proxy_regex_v0</code>.
        <strong> This is a regex proxy, not a validated clinical NLP model.</strong>
        {" "}Confirm every field before the therapy branch is called.
      </p>

      <Row<boolean>
        label="ER"
        options={[
          { value: true, label: "Positive" },
          { value: false, label: "Negative" },
        ]}
        value={er}
        onChange={(v) => { setEr(v); setErState("user_supplied"); }}
        provenance={erState}
      />
      <Row<boolean>
        label="PR"
        options={[
          { value: true, label: "Positive" },
          { value: false, label: "Negative" },
        ]}
        value={pr}
        onChange={(v) => { setPr(v); setPrState("user_supplied"); }}
        provenance={prState}
      />
      <Row<Her2Value>
        label="HER2"
        options={[
          { value: "positive", label: "Positive" },
          { value: "negative", label: "Negative" },
          { value: "equivocal", label: "Equivocal" },
        ]}
        value={her2}
        onChange={(v) => { setHer2(v); setHer2State("user_supplied"); }}
        provenance={her2State}
      />
      <Row<number>
        label="Grade"
        options={[
          { value: 1, label: "1" },
          { value: 2, label: "2" },
          { value: 3, label: "3" },
        ]}
        value={grade}
        onChange={(v) => { setGrade(v); setGradeState("user_supplied"); }}
        provenance={gradeState}
      />

      <div style={{ marginTop: "1rem", display: "flex", alignItems: "center",
                    gap: "1rem", flexWrap: "wrap" }}>
        <button
          type="button"
          className="primary"
          disabled={!allFilled || busy}
          onClick={handleConfirm}
          data-testid="confirm-receptors-btn"
        >
          {busy ? "Running therapy…" : "Confirm receptors → run therapy"}
        </button>
        {!allFilled && (
          <span style={{ fontSize: "0.8rem", color: "var(--fg-muted)" }}>
            Fill every field to enable the therapy call.
          </span>
        )}
      </div>
    </div>
  );
}
