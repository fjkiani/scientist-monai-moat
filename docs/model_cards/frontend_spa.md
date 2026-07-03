# Frontend SPA (React + Vite)

**Research Use Only. Not for diagnostic use.**

The oncology-arbiter frontend is a single-page React application served
optionally by the FastAPI backend under `/ui/`. It calls only the documented
`/v1/*` and `/health` endpoints — it holds no clinical logic, no thresholds,
and no risk-tier assignments of its own.

## Design contract

The frontend is a **transparent renderer of server envelopes**. It MUST NOT:

- Coerce a `model_state="placeholder"` response into a "loaded" appearance.
- Drop or hide the RUO `disclaimer` or `caveat` fields.
- Reorder, filter, or aggregate warnings out of the `warnings[]` list.
- Compute its own recommendation or risk bucket — those come from the L3
  arbiter and the therapy engine on the server.
- Persist any patient data client-side (no localStorage, IndexedDB, or
  service worker caching of API payloads).

Every tab renders the raw server envelope (provenance pill, evidence,
warnings, disclaimer+caveat) via the shared `<EnvelopeCard>` component so
these invariants cannot be forgotten in one screen and enforced in another.

## Tabs

- **Screening** (`/v1/screening/analyze`) — DICOM upload; renders findings
  with an in-DOM bbox overlay for any finding carrying
  `location_bbox_normalized` (i.e. MONAI detector hits).
- **Biopsy** (`/v1/biopsy/analyze`) — WSI-image or pathology-report upload;
  renders subtype, grade, receptor panel, confidence.
- **Therapy** (`/v1/therapy/reason`) — receptor + stage + menopausal form;
  submits a synthesized `biopsy_output` envelope built from the form fields
  (the endpoint accepts either a real biopsy response or a hand-built one).
- **Case view** (`/v1/case/full`) — end-to-end chained call across all three
  stages; surfaces the Elo-ranked hypotheses block verbatim.
- **Model cards** — indexes `/v1/model-cards` and links to raw markdown
  via `/v1/artifacts/docs/{slug}.md`.

## Provenance pill color contract

| Model state class in CSS       | Meaning                                | Color         |
|--------------------------------|----------------------------------------|---------------|
| `pill.loaded`, `pill.loaded_medsiglip`, `pill.loaded_biopsy_probe`, `pill.loaded_txgemma`, `pill.loaded_monai_detector` | Live model returned a real inference | Green (`--accent-2`) |
| `pill.proxy_siglip`, `pill.proxy_rules_lite`, `pill.proxy_monai_heuristic` | Proxy / heuristic path, not the real model | Orange (`--warn`) |
| `pill.gated`, `pill.unavailable` | Model refused (HAI-DEF gate or load failure) | Yellow (`--danger`) |
| `pill.placeholder` (unstyled)  | Endpoint returned a stub               | Grey          |

The three color families are chosen so a reader can tell a proxy result
from a live result at a glance without having to read the pill text.

## Static-mount opt-in

Serving the SPA is gated by `ONCOLOGY_ARBITER_SERVE_FRONTEND=1`. When the
flag is unset, `create_app()` does not mount `/ui/*` at all — this keeps
the backend Docker image buildable without a Node toolchain.

## Build reproducibility

- React 18.3.1 / react-dom 18.3.1
- Vite 5.4.10 / @vitejs/plugin-react 4.3.1
- TypeScript 5.6.3 (strict, `noEmit` only — Vite handles the emit)
- Node 18.20.4 (session runtime, not pinned in `package.json.engines` yet)

Build command: `npm --prefix frontend run build`, emits to
`src/oncology_arbiter/api/static/dist/`.

## Failure modes surfaced

| Failure                              | UI behavior                                                        |
|--------------------------------------|--------------------------------------------------------------------|
| Backend not reachable                | Fetch throws; error string rendered in tab-scoped error banner.    |
| `model_state=placeholder`            | Grey pill; envelope card shows disclaimer+caveat unchanged.        |
| `model_state=gated`                  | Yellow pill; warning list renders `<repo>_gated:<level>:<reason>`. |
| Empty `recommended_options`          | "No options returned (placeholder or gated)" copy.                 |
| Missing `location_bbox_normalized`   | Finding shown in table only; no overlay drawn.                     |

## Not implemented in this frontend

- No PACS integration.
- No PHI storage (no local cache).
- No offline mode.
- No streaming — every response is a single HTTP request/response.
- No dark mode (yet).

## Honesty warning (verbatim)

> This frontend is a viewer over model outputs, not a diagnostic device.
> Model states, warnings, and disclaimers are surfaced verbatim from the
> server. Do not use this tool to make clinical decisions.
