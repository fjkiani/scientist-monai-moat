#!/usr/bin/env python3
"""Delivery validator (importable module + CLI).

This file must be importable for `tests/unit/test_oncology_gauntlet.py`.
Implementation lives in `scripts/validate_tonight_delivery.py` to keep the
skill folder tidy, but we load it here and re-export its symbols.
"""

from __future__ import annotations

import runpy
from pathlib import Path

_IMPL = Path(__file__).with_name("scripts") / "validate_tonight_delivery.py"
_ns = runpy.run_path(str(_IMPL), run_name="validate_tonight_delivery_impl")
for k, v in _ns.items():
    # Don't clobber module dunders; in particular, preserve `__name__ == "__main__"`
    # when this file is executed as a script.
    if k.startswith("__"):
        continue
    globals()[k] = v


if __name__ == "__main__":
    raise SystemExit(main())

