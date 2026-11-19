#!/usr/bin/env python3
"""One-shot rebuild ml_8d package from indexed monolith (dev utility)."""
from __future__ import annotations

from pathlib import Path

MONO = Path("/tmp/ml_8d_trial_decoder_monolith.py")
OUT = Path(__file__).resolve().parents[1] / "src/oncology_arbiter/arbiter/ml_8d"

lines = MONO.read_text().splitlines(keepends=True)


def sl(start: int, end: int) -> str:
    return "".join(lines[start - 1 : end])


def write(name: str, content: str) -> None:
    (OUT / name).write_text(content)
    print(f"wrote {name} ({content.count(chr(10))} lines)")


write(
    "constants.py",
    '''"""8D bilinear decoder constants (axes, MoA class names, RUO strings)."""
from __future__ import annotations

'''
    + sl(54, 93),
)

write(
    "paths.py",
    '''"""Filesystem roots for cohort data loaders."""
from __future__ import annotations

from pathlib import Path

'''
    + sl(888, 892),
)

write(
    "patient.py",
    '''"""Signed 8D patient vector construction."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .constants import SIGNED_AXES_8D

'''
    + sl(96, 263),
)

write(
    "moa.py",
    '''"""MoA weight vectors for K-dimensional mechanism columns."""
from __future__ import annotations

import numpy as np

from .constants import MOA_CLASSES_15

'''
    + sl(270, 277)
    + "\n\n\ndef moa_vec(moa_classes: tuple[str, ...] | None = None, **weights: float) -> np.ndarray:\n"
    + '    """Build MoA vector for arbitrary contract column set (default v1 K=15)."""\n'
    + "    classes = moa_classes or MOA_CLASSES_15\n"
    + "    vec = np.zeros(len(classes), dtype=float)\n"
    + "    idx_map = {name: i for i, name in enumerate(classes)}\n"
    + "    for k, v in weights.items():\n"
    + "        if k not in idx_map:\n"
    + '            raise KeyError(f"Unknown MoA class: {k}")\n'
    + "        vec[idx_map[k]] = float(v)\n"
    + "    return vec\n",
)

write(
    "registry.py",
    '''"""Trial registry MoA specs and recognized clinical feature keys."""
from __future__ import annotations

from typing import Any

'''
    + sl(281, 399)
    + "\n"
    + sl(581, 636),
)

write(
    "scoring.py",
    '''"""Signed bilinear projection scoring helpers."""
from __future__ import annotations

import numpy as np

'''
    + sl(406, 428),
)

write(
    "exceptions.py",
    '''"""Bilinear decoder input validation errors."""
from __future__ import annotations

'''
    + sl(431, 437),
)

write(
    "decoder_io.py",
    '''"""Load and validate crispro_8d_bilinear_decoder_v*.json contracts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import MOA_CLASSES_15, SIGNED_AXES_8D
from .exceptions import BilinearDecoderInputError


def contract_moa_classes(data: dict[str, Any]) -> tuple[str, ...]:
    """Resolve MoA column names from v1 (moa_classes_15) or v2+ keys."""
    for key in ("moa_classes_24", "moa_classes", "moa_classes_15"):
        raw = data.get(key)
        if isinstance(raw, list) and raw:
            return tuple(str(x) for x in raw)
    raise BilinearDecoderInputError(
        "contract missing moa class list",
        code="contract_shape_mismatch",
    )


def contract_matrix_k(data: dict[str, Any]) -> int:
    classes = contract_moa_classes(data)
    matrix = data.get("W_MoA_matrix")
    if not isinstance(matrix, list) or len(matrix) != 8:
        raise BilinearDecoderInputError(
            "W_MoA_matrix is not 8 rows",
            code="contract_shape_mismatch",
        )
    k = len(classes)
    if any(not isinstance(row, list) or len(row) != k for row in matrix):
        raise BilinearDecoderInputError(
            f"W_MoA_matrix is not 8x{k}",
            code="contract_shape_mismatch",
        )
    shape = data.get("W_MoA_shape")
    if shape is not None and list(shape) != [8, k]:
        raise BilinearDecoderInputError(
            f"W_MoA_shape {shape!r} does not match 8x{k}",
            code="contract_shape_mismatch",
        )
    return k


def bilinear_contract_path(version: str | None = None) -> Path:
    """Frozen weights under arbiter/models/ (not PATH A registry)."""
    models = Path(__file__).resolve().parent.parent / "models"
    if version in (None, "v1", "1"):
        return models / "crispro_8d_bilinear_decoder_v1.json"
    if version in ("v2", "2"):
        return models / "crispro_8d_bilinear_decoder_v2.json"
    return models / f"crispro_8d_bilinear_decoder_{version}.json"


def load_bilinear_contract(path: Path | None = None) -> dict[str, Any]:
    """Load bilinear decoder JSON. Does not refit W_MoA."""
    contract_path = path or bilinear_contract_path()
    if not contract_path.is_file():
        raise BilinearDecoderInputError(
            f"bilinear contract missing at {contract_path}",
            code="contract_missing",
            missing=[contract_path.name],
        )
    data = json.loads(contract_path.read_text())
    axes = data.get("signed_axes_8d")
    if axes != list(SIGNED_AXES_8D):
        raise BilinearDecoderInputError(
            "contract signed_axes_8d does not match the module constants",
            code="contract_shape_mismatch",
        )
    moa_15 = data.get("moa_classes_15")
    if moa_15 is not None and list(moa_15) != list(MOA_CLASSES_15):
        raise BilinearDecoderInputError(
            "contract moa_classes_15 does not match the module constants",
            code="contract_shape_mismatch",
        )
    contract_matrix_k(data)
    return data
''',
)

# trials.py — score/rank with v1+v2 MoA column support
trials_body = sl(475, 578)
trials_body = trials_body.replace("MOA_CLASSES_15", "moa_classes")
trials_body = trials_body.replace(
    '''    unknown = [str(k) for k in moa.keys() if k not in MOA_CLASSES_15]
    if unknown:
        raise BilinearDecoderInputError(
            f"unknown moa columns: {unknown}",
            code="shape_mismatch",
        )
    try:
        weights = {str(k): float(v) for k, v in moa.items()}
''',
    '''    try:
        weights = {str(k): float(v) for k, v in moa.items()}
''',
)
trials_body = trials_body.replace(
    '    contract = load_bilinear_contract(contract_path)\n    w_moa = np.asarray(contract["W_MoA_matrix"], dtype=float)\n    if w_moa.shape != (8, 15):\n        raise BilinearDecoderInputError(\n            f"W_MoA_matrix shape {w_moa.shape} is not (8, 15)",\n            code="contract_shape_mismatch",\n        )\n    m_vec = _moa_vec(**weights)',
    '''    contract = load_bilinear_contract(contract_path)
    moa_classes = contract_moa_classes(contract)
    w_moa = np.asarray(contract["W_MoA_matrix"], dtype=float)
    if w_moa.shape[0] != 8:
        raise BilinearDecoderInputError(
            f"W_MoA_matrix shape {w_moa.shape} is not (8, K)",
            code="contract_shape_mismatch",
        )
    unknown = [str(key) for key in weights if key not in moa_classes]
    if unknown:
        raise BilinearDecoderInputError(
            f"unknown moa columns: {unknown}",
            code="shape_mismatch",
        )
    m_vec = moa_vec(moa_classes, **weights)''',
)
trials_body = trials_body.replace(
    "bilinear_ml_loto requires moa_weights over the 15 Drug MoA columns",
    "bilinear_ml_loto requires moa_weights over contract MoA columns",
)

rank_body = sl(639, 856)
rank_body = rank_body.replace(
    "    contract = load_bilinear_contract(contract_path)\n    w_moa = np.asarray(contract[\"W_MoA_matrix\"], dtype=float)\n",
    "    contract = load_bilinear_contract(contract_path)\n    moa_classes = contract_moa_classes(contract)\n    w_moa = np.asarray(contract[\"W_MoA_matrix\"], dtype=float)\n",
)
rank_body = rank_body.replace("_moa_vec(**moa_map)", "moa_vec(moa_classes, **moa_map)")

write(
    "trials.py",
    '''"""Single-trial scoring and 42-trial ranking (frozen W_MoA)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .constants import SIGNED_AXES_8D
from .decoder_io import contract_moa_classes, load_bilinear_contract
from .exceptions import BilinearDecoderInputError
from .moa import moa_vec
from .patient import build_signed_8d_patient_vector
from .registry import (
    COMPARATOR_MOA_SPECS,
    RECOGNIZED_CLINICAL_FEATURE_KEYS,
    REGISTRY_42_MOA_SPECS,
)
from .scoring import signed_cosine_similarity, signed_fit_score

'''
    + trials_body
    + "\n\n"
    + rank_body,
)

cohorts_header = '''"""Responder cohort and subgroup contrast loaders."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy import optimize, stats
from sklearn.metrics import roc_auc_score

from .paths import _CASPRO_ROOT, _CRISPR_MAIN_ROOT, _DEV_ROOT, _MOAT_ROOT

'''
write("cohorts.py", cohorts_header + sl(860, 887) + sl(895, 1488))

fit_header = '''"""Fit W_MoA from cohorts (offline; not used on query path)."""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from .constants import MOA_CLASSES_15, SIGNED_AXES_8D
from .cohorts import ResponderCohortBlock, SubgroupContrastBlock, load_all_responder_cohorts, load_published_subgroup_contrasts
from .moa import _moa_vec
from .scoring import signed_fit_batch, signed_fit_score

'''
write("fit.py", fit_header + sl(1490, 1636))

loto_header = '''"""LOTO validation and trial vector decode."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .cohorts import ResponderCohortBlock, SubgroupContrastBlock, load_all_responder_cohorts, load_published_subgroup_contrasts
from .constants import MOA_CLASSES_15, SIGNED_AXES_8D
from .exceptions import BilinearDecoderInputError
from .fit import _bootstrap_auroc_ci, build_biological_anchor_prior_w_moa, fit_bilinear_w_moa
from .moa import _moa_vec
from .patient import build_signed_8d_patient_vector
from .registry import COMPARATOR_MOA_SPECS, REGISTRY_42_MOA_SPECS
from .scoring import signed_fit_batch, signed_fit_score

'''
write("loto.py", loto_header + sl(1638, 1897))

routing_header = '''"""TCGA-COADREAD MSS portfolio routing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from .decoder_io import load_bilinear_contract
from .exceptions import BilinearDecoderInputError
from .moa import _moa_vec
from .patient import build_signed_8d_patient_vector
from .registry import COMPARATOR_MOA_SPECS, REGISTRY_42_MOA_SPECS, RECOGNIZED_CLINICAL_FEATURE_KEYS
from .scoring import signed_fit_batch, signed_fit_score

'''
write("routing.py", routing_header + sl(1899, 2150))

pipeline_header = '''"""Full offline pipeline export (CLI entry)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .decoder_io import load_bilinear_contract
from .loto import decode_all_trials_with_w_moa, run_loto_and_heldout_validation
from .routing import route_485_tcga_coadread_mss_patients

'''
write("pipeline.py", pipeline_header + sl(2152, 2231))

SHIM = Path(__file__).resolve().parents[1] / "src/oncology_arbiter/arbiter/ml_8d_trial_decoder.py"
SHIM.write_text(
    '''"""Backward-compatible shim — import from oncology_arbiter.arbiter.ml_8d."""
from __future__ import annotations

from oncology_arbiter.arbiter.ml_8d import *  # noqa: F403
from oncology_arbiter.arbiter.ml_8d import __all__  # noqa: F401
''',
)
print("shim ok")