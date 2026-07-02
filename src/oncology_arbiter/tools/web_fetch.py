"""Web fetch — pull URL, clean text, cache by URL hash.

Ported from `Co-Scientist/co_scientist/tools/web_fetch.py`. Config indirection
replaced with module-level constants. Everything security-relevant kept
verbatim: SSRF guard (initial + redirect), stream+cap size, upfront
Content-Length check, User-Agent, redirect ceiling.

- HTML → trafilatura.extract
- PDF (Content-Type or .pdf suffix) → pypdf text extraction
- Cache: `<artifacts_dir>/papers/<sha1(url)>.json` (survives resume)
"""
from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import socket
import time
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from .base import ToolCtx, ToolResult

# Defaults matching Co-Scientist's runtime config, hoisted here since we
# dropped the `cfg.web_fetch` layer.
MAX_BYTES: int = 5 * 1024 * 1024      # 5 MB body cap
TIMEOUT_SECONDS: float = 30.0
USER_AGENT: str = (
    "oncology-arbiter/0.1 (+https://github.com/fjkiani/oncology-arbiter) "
    "python-httpx"
)
MAX_REDIRECTS: int = 5


# --------------------------------------------------------------------------- #
# SSRF guard


def _is_private_ip(host: str) -> bool:
    """Return True if host resolves to a private / loopback / link-local /
    reserved IP. Used to block SSRF against the metadata service and
    intranet targets even when the user-supplied URL passes the scheme check.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        # If we can't resolve, be conservative: refuse.
        return True
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr.split("%")[0])
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return True
    return False


# --------------------------------------------------------------------------- #
# Tool


class WebFetchTool:
    name = "web_fetch"
    description = (
        "Fetch a URL and return its main text content (HTML → cleaned text; PDF → "
        "extracted text). Returns {url, title?, text, content_type, status, bytes}. "
        "Use after pubmed_search / arxiv_search / europe_pmc_search to read the "
        "actual paper. Cached per-session by URL hash."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Absolute http(s) URL."},
            "max_chars": {
                "type": "integer", "minimum": 200, "maximum": 200_000, "default": 30_000,
                "description": "Truncate text to this many characters.",
            },
        },
        "required": ["url"],
    }

    async def call(self, args: dict[str, Any], ctx: ToolCtx) -> ToolResult:
        t0 = time.monotonic()
        url = (args.get("url") or "").strip()
        max_chars = int(args.get("max_chars") or 30_000)
        if not url.startswith(("http://", "https://")):
            return ToolResult(is_error=True, error_message="URL must start with http(s)")

        host = urlsplit(url).hostname or ""
        if not host:
            return ToolResult(is_error=True, error_message="URL has no host")
        if await asyncio.to_thread(_is_private_ip, host):
            return ToolResult(
                is_error=True,
                error_message="URL resolves to a private/loopback address",
            )

        cached = await self._read_cache(ctx, url)
        if cached is not None:
            cached = self._truncate(cached, max_chars)
            return ToolResult(
                content=cached,
                duration_ms=int((time.monotonic() - t0) * 1000),
                result_bytes=len(json.dumps(cached)),
            )

        async def _check_redirect(response: httpx.Response) -> None:
            loc = response.headers.get("location")
            if not loc:
                return
            next_url = (
                httpx.URL(loc)
                if loc.startswith(("http://", "https://"))
                else response.url.join(loc)
            )
            next_host = next_url.host
            if not next_host or await asyncio.to_thread(_is_private_ip, next_host):
                raise httpx.RequestError(
                    "redirect to private/loopback address blocked",
                    request=response.request,
                )

        try:
            async with (
                httpx.AsyncClient(
                    timeout=TIMEOUT_SECONDS,
                    follow_redirects=True,
                    max_redirects=MAX_REDIRECTS,
                    headers={"User-Agent": USER_AGENT},
                    event_hooks={"response": [_check_redirect]},
                ) as client,
                client.stream("GET", url) as r,
            ):
                if r.status_code >= 400:
                    return ToolResult(
                        is_error=True,
                        error_message=f"HTTP {r.status_code}",
                        content={"url": url, "status": r.status_code},
                    )
                cl = r.headers.get("content-length")
                if cl is not None:
                    try:
                        if int(cl) > MAX_BYTES:
                            return ToolResult(
                                is_error=True,
                                error_message=f"response too large ({cl} bytes advertised)",
                            )
                    except ValueError:
                        pass

                chunks: list[bytes] = []
                total = 0
                async for chunk in r.aiter_bytes():
                    total += len(chunk)
                    if total > MAX_BYTES:
                        return ToolResult(
                            is_error=True,
                            error_message=f"response too large (>{MAX_BYTES} bytes, stopped streaming)",
                        )
                    chunks.append(chunk)
                body = b"".join(chunks)
                final_url = str(r.url)
                status = r.status_code
                headers = r.headers
        except httpx.HTTPError as e:
            return ToolResult(is_error=True, error_message=f"fetch failed: {e}")

        ct = (headers.get("Content-Type") or "").lower()
        is_pdf = "application/pdf" in ct or url.lower().endswith(".pdf")
        try:
            if is_pdf:
                text = await asyncio.to_thread(_extract_pdf, body)
                title: str | None = None
            else:
                text, title = await asyncio.to_thread(
                    _extract_html, body.decode("utf-8", errors="replace"), url
                )
        except Exception as e:
            return ToolResult(is_error=True, error_message=f"extraction failed: {e}")

        payload: dict[str, Any] = {
            "url": final_url,
            "title": title,
            "text": text,
            "content_type": ct,
            "status": status,
            "bytes": total,
        }
        await self._write_cache(ctx, url, payload)
        payload = self._truncate(payload, max_chars)
        return ToolResult(
            content=payload,
            duration_ms=int((time.monotonic() - t0) * 1000),
            result_bytes=len(json.dumps(payload)),
        )

    # ----------------------------- cache --------------------------------- #

    def _cache_path(self, ctx: ToolCtx, url: str) -> Path | None:
        if ctx.session_id is None:
            return ctx.artifacts_dir / "papers" / f"{_url_hash(url)}.json"
        return (
            ctx.artifacts_dir / "sessions" / ctx.session_id / "papers"
            / f"{_url_hash(url)}.json"
        )

    async def _read_cache(self, ctx: ToolCtx, url: str) -> dict[str, Any] | None:
        p = self._cache_path(ctx, url)
        if p is None or not p.exists():
            return None

        def _do() -> dict[str, Any]:
            return json.loads(p.read_text())

        return await asyncio.to_thread(_do)

    async def _write_cache(
        self, ctx: ToolCtx, url: str, payload: dict[str, Any]
    ) -> None:
        p = self._cache_path(ctx, url)
        if p is None:
            return

        def _do() -> None:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, default=str, ensure_ascii=False))
            tmp.replace(p)

        await asyncio.to_thread(_do)

    @staticmethod
    def _truncate(payload: dict[str, Any], max_chars: int) -> dict[str, Any]:
        text = payload.get("text") or ""
        if len(text) > max_chars:
            payload = {**payload, "text": text[:max_chars], "truncated": True}
        return payload


# --------------------------------------------------------------------------- #
# Extractors (sync — run via to_thread)


def _extract_html(html: str, url: str) -> tuple[str, str | None]:
    import trafilatura  # type: ignore[import-not-found]

    extracted = trafilatura.extract(
        html, url=url, include_comments=False, include_tables=True
    )
    title = None
    md = trafilatura.metadata.extract_metadata(html)
    if md:
        title = md.title
    return extracted or "", title


def _extract_pdf(data: bytes) -> str:
    import pypdf  # type: ignore[import-not-found]

    reader = pypdf.PdfReader(BytesIO(data))
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n\n".join(parts)


def _url_hash(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()
