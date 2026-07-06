// v0.2.2: safe base64 encode/decode helpers for large DICOM/WSI payloads.
//
// The naïve one-liner
//     btoa(String.fromCharCode(...bytes))
// spreads N million arguments into fromCharCode for a ~14 MB DICOM, which
// blows past V8's callstack limit (~65k args). Chunk by 32 KB so each
// spread stays safe, then concatenate the base64 outputs.

const B64_CHUNK = 0x8000; // 32 KB — well under any engine's arg-spread limit

/** Base64-encode a Uint8Array of arbitrary length. Safe for ≥14 MB inputs. */
export function bytesToBase64Chunked(bytes: Uint8Array): string {
  let out = "";
  for (let i = 0; i < bytes.length; i += B64_CHUNK) {
    const slice = bytes.subarray(i, Math.min(i + B64_CHUNK, bytes.length));
    // fromCharCode.apply is safe: slice.length <= 32k.
    out += String.fromCharCode.apply(
      null, slice as unknown as number[]
    );
  }
  return btoa(out);
}

/** Decode a base64 string into a Uint8Array. atob has no length limit; we
 * just loop char-by-char, which is O(n) and avoids any spread overflow. */
export function base64ToBytes(b64: string): Uint8Array {
  const decoded = atob(b64);
  const bytes = new Uint8Array(decoded.length);
  for (let i = 0; i < decoded.length; i++) bytes[i] = decoded.charCodeAt(i);
  return bytes;
}
