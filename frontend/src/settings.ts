/**
 * Client-side settings (API key + cancer selection + last X-Request-Id).
 *
 * Everything here is scoped to `localStorage` — the operator has to paste
 * a key in the drawer, we never persist it server-side, we never bake it
 * into the SPA bundle, and it's easy to blow away by clicking `Clear`.
 *
 * The `apiKey` field is masked in the UI (`•` for every char but the
 * `oa_live_` prefix and the last 4 chars). If a request comes back 401,
 * the drawer opens so the operator can rotate.
 */

export type CancerId = "breast" | "nsclc";

export const CANCERS: CancerId[] = ["breast", "nsclc"];

export const LS_KEY_APIKEY = "oa.apiKey";
export const LS_KEY_CANCER = "oa.cancer";
export const LS_KEY_LAST_REQID = "oa.lastRequestId";

export function getApiKey(): string {
  return (typeof window !== "undefined" && window.localStorage.getItem(LS_KEY_APIKEY)) || "";
}

export function setApiKey(k: string): void {
  if (typeof window === "undefined") return;
  if (!k) window.localStorage.removeItem(LS_KEY_APIKEY);
  else window.localStorage.setItem(LS_KEY_APIKEY, k);
}

export function getCancer(): CancerId {
  if (typeof window === "undefined") return "breast";
  const v = window.localStorage.getItem(LS_KEY_CANCER);
  if (v === "nsclc") return "nsclc";
  return "breast";
}

export function setCancer(c: CancerId): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(LS_KEY_CANCER, c);
}

export function getLastRequestId(): string {
  return (typeof window !== "undefined" && window.localStorage.getItem(LS_KEY_LAST_REQID)) || "";
}

export function setLastRequestId(rid: string): void {
  if (typeof window === "undefined" || !rid) return;
  window.localStorage.setItem(LS_KEY_LAST_REQID, rid);
}

/**
 * Renders a key as `oa_live_c221…6680` — enough to spot check on screen
 * without leaking. Returns an empty string if the key is unset.
 */
export function maskKey(k: string): string {
  if (!k) return "";
  if (k.length <= 12) return k; // shouldn't happen, but stay safe
  const head = k.slice(0, 12);
  const tail = k.slice(-4);
  return `${head}…${tail}`;
}
