"""Load and validate crispro_8d_bilinear_decoder_v*.json contracts."""
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
