import type { ScreeningFinding } from "../api";

interface BboxOverlayProps {
  imageDataUrl: string | null;
  findings: ScreeningFinding[];
  width?: number;
  height?: number;
}

/**
 * Renders an image (or grey placeholder) with normalized bboxes overlaid
 * for any findings that carry `location_bbox_normalized`. Findings without
 * a bbox (SigLIP zero-shot labels) are ignored here — they show up in the
 * findings table separately.
 */
export function BboxOverlay({ imageDataUrl, findings, width = 512, height = 512 }: BboxOverlayProps) {
  const bboxed = findings.filter(
    (f) => f.location_bbox_normalized !== null && f.label.startsWith("monai_heuristic:"),
  );

  return (
    <div style={{
      position: "relative",
      width,
      height,
      border: "1px solid var(--border)",
      background: imageDataUrl ? `#000 url(${imageDataUrl}) center/contain no-repeat` : "#f1f5f9",
    }}>
      {!imageDataUrl && (
        <div style={{
          position: "absolute", inset: 0,
          display: "flex", alignItems: "center", justifyContent: "center",
          color: "var(--fg-muted)", fontSize: "0.85rem",
        }}>
          No image preview (upload a DICOM to see bbox overlays)
        </div>
      )}
      {bboxed.map((f, i) => {
        const [x0, y0, x1, y1] = f.location_bbox_normalized!;
        return (
          <div key={i} style={{
            position: "absolute",
            left: `${x0 * 100}%`,
            top: `${y0 * 100}%`,
            width: `${(x1 - x0) * 100}%`,
            height: `${(y1 - y0) * 100}%`,
            border: "2px solid var(--warn)",
            boxShadow: "0 0 0 1px rgba(0,0,0,0.4)",
            pointerEvents: "none",
          }}>
            <div style={{
              position: "absolute", top: -20, left: 0,
              background: "var(--warn)", color: "black",
              padding: "1px 4px", fontSize: "0.7rem", borderRadius: "2px",
              whiteSpace: "nowrap", fontFamily: "Menlo, monospace",
            }}>
              {f.label.replace("monai_heuristic:", "")} · {(f.score * 100).toFixed(0)}%
            </div>
          </div>
        );
      })}
    </div>
  );
}
