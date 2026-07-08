import { useEffect, useState } from "react";

/**
 * Cold-start banner for Render free-tier deployments.
 *
 * Free-tier web services spin down after 15 min without inbound traffic.
 * On the next request the dyno takes ~1 min to boot. Render shows its
 * OWN loading page during that first-hit boot, so this banner only
 * matters for the mid-session case: a user is already on the SPA when
 * the backend spins down and their next fetch stalls or 502s.
 *
 * Behaviour:
 *   - `unavailable` prop is true → show the banner with a spinner.
 *   - After 6 s of continuous unavailability, escalate the copy to make
 *     it explicit that this is Render cold-start behaviour, not a bug.
 *   - `unavailable` returns to false → hide.
 */
export function ColdStartBanner({
  unavailable,
  reason,
}: {
  unavailable: boolean;
  reason: string;
}) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!unavailable) {
      setElapsed(0);
      return;
    }
    const t0 = Date.now();
    setElapsed(0);
    const id = window.setInterval(() => {
      setElapsed(Math.floor((Date.now() - t0) / 1000));
    }, 1000);
    return () => window.clearInterval(id);
  }, [unavailable]);

  if (!unavailable) return null;

  const escalated = elapsed >= 6;

  return (
    <div
      role="alert"
      style={{
        marginBottom: "1rem",
        padding: "0.75rem 1rem",
        borderLeft: "3px solid #FF9400",
        background: "#fff5e6",
        color: "#000",
        fontSize: "0.9rem",
        borderRadius: 4,
      }}
      data-testid="cold-start-banner"
    >
      <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
        <div className="spinner" style={{
          width: 16, height: 16,
          border: "2px solid #FF9400",
          borderTopColor: "transparent",
          borderRadius: "50%",
          animation: "cs-spin 1s linear infinite",
          flexShrink: 0,
        }} />
        <div>
          <div style={{ fontWeight: 600 }}>
            {escalated ? "Backend is warming up — this can take up to ~60 s." : "Backend is unreachable — retrying…"}
          </div>
          <div style={{ fontSize: "0.8rem", color: "#555", marginTop: "0.2rem" }}>
            {escalated
              ? (
                <>
                  This is a Render free-tier demo. After 15 min of inactivity the
                  service spins down; the next request takes about a minute to
                  boot. Pre-computed samples continue to work as soon as
                  <code style={{ margin: "0 0.2rem" }}>/health</code>
                  responds. Elapsed: {elapsed}s.
                </>
              )
              : (
                <>
                  Waiting {elapsed}s. Reason: <code>{reason}</code>
                </>
              )}
          </div>
        </div>
      </div>
      <style>{`
        @keyframes cs-spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
