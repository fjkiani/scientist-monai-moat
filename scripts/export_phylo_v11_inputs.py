#!/usr/bin/env python3
"""Export Git-safe ovarian DepMap + GSE165897 slice for Phylo v11."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "phylo_v11_inputs"
DEPMAP_CACHE = REPO_ROOT / "oncology-coPilot" / "oncology-backend-minimal" / ".cache" / "depmap"
GSE_DIR = REPO_ROOT / "scripts" / "data_acquisition" / "sae"
CONSTANTS_PATH = (
    REPO_ROOT
    / "oncology-coPilot"
    / "oncology-backend-minimal"
    / "api"
    / "services"
    / "therapy_fit_v4"
    / "resources"
    / "ovarian_two_door_standalone_v3.py"
)

ALPHA_EXTRA_GENES = [
    "BCL2L1", "BCL2", "MCL1", "BCL2A1", "PMAIP1", "BCL2L11", "BAD", "BAX", "BAK1", "CPEB4", "WASHC4",
    "HGS", "VDR", "RXRA", "EPHA2", "SRC", "YES1", "FYN", "LCK", "PTK2", "FERMT2", "ITGAV", "ITGB1",
    "ITGB3", "ITGB5", "RICTOR", "RPTOR", "MTOR", "AKT1", "STAT3", "IL6", "CLPP", "MRPL12", "MRPS15",
    "SMO", "GLI1", "GLI2", "PTCH1", "KITLG", "KIT", "LEPR", "PI16", "FABP4", "LRRC15", "ENG", "FAP",
    "MFAP4", "CXCL10", "CXCL9", "YAP1", "WWTR1", "PF4", "HLA-DRA", "CD74", "CD80", "CD86", "ABCB1",
    "TUBB3", "SLFN11", "PARP1", "PARP2", "RNF144A", "ATR", "CHEK1", "WEE1", "PKMYT1", "CCNE1", "RB1",
    "NF1", "PTEN", "KRAS", "NRAS", "BRAF", "TP53", "MBD4", "BRCA1", "BRCA2", "RAD51C", "RAD51D", "PALB2",
    "RRM1", "RRM2", "TYMS", "DCK", "TOP1", "HDAC1", "HDAC2", "HDAC3", "HDAC6", "HMGCR", "ADRB2", "ATP1A1",
]

MUT_CNA_GENES = [
    "BRCA1", "BRCA2", "TP53", "MBD4", "CCNE1", "NF1", "RB1", "PTEN", "KRAS", "BRAF", "ARID1A", "PIK3CA", "MYC",
]

OVARIAN_LINEAGE = "Ovary/Fallopian Tube"
TARBALL_NAME = "phylo_v11_ovarian_inputs.tar.gz"
MAX_TAR_MB = 10.0
TARGET_TAR_MB = 8.0

# v10 alias copies on disk only — never pack twice (doubles tarball size).
TAR_SKIP_ALIASES = frozenset(
    {
        "ovarian_expression.parquet",
        "ovarian_expression.csv",
        "ovarian_crispr.parquet",
        "ovarian_crispr.csv",
    }
)

# Phylo upload tarball: pack priority set first, then add full-gene parquets if under MAX_TAR_MB.
TAR_MEMBER_PRIORITY = (
    "MANIFEST_SHA256.json",
    "ovarian_expression_key_panel.csv",
    "ovarian_crispr_key_panel.csv",
    "ovarian_mutations_and_cna.csv",
    "gse165897_human_paired_compartments.csv",
    "ovarian_expression_all_genes.parquet",
    "ovarian_crispr_all_genes.parquet",
)


def _load_constants():
    spec = importlib.util.spec_from_file_location("ovarian_two_door_standalone_v3", CONSTANTS_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def build_key_panel_gene_map(constants) -> Dict[str, int]:
    panel: Dict[str, int] = {}
    for d in (
        constants.STRESS_34_CLEAN_GENES,
        constants.PROLIFERATION_106_GENES,
        constants.DDR_PROXY_4_GENES,
        getattr(constants, "EXTRA_GENE_IDS", {}),
    ):
        panel.update(d)
    for g in ALPHA_EXTRA_GENES:
        if g not in panel:
            panel[g] = None  # type: ignore
    for g, eid in list(panel.items()):
        if eid is None:
            panel.pop(g)
    return dict(sorted(panel.items(), key=lambda x: x[0]))


def resolve_expression_columns(columns: pd.Index, gene_map: Dict[str, int]) -> Dict[str, str]:
    """Map Hugo symbol -> OmicsExpression column name."""
    by_symbol: Dict[str, List[str]] = {}
    for col in columns:
        if " (" in col and col.endswith(")"):
            sym = col.split(" (")[0]
            by_symbol.setdefault(sym, []).append(col)
    resolved: Dict[str, str] = {}
    missing: List[str] = []
    for sym, eid in gene_map.items():
        cols = by_symbol.get(sym, [])
        if not cols:
            missing.append(sym)
            continue
        if eid is not None:
            want = f"{sym} ({eid})"
            picked = want if want in cols else cols[0]
        else:
            picked = cols[0]
        resolved[sym] = picked
    if missing:
        print(f"  WARN: {len(missing)} key-panel genes missing from OmicsExpression: {missing[:12]}...")
    return resolved


def resolve_crispr_columns(columns: pd.Index, gene_map: Dict[str, int]) -> Dict[str, str]:
    """Map Hugo symbol -> crispr_gene_effect column (same `GENE (entrez)` pattern as expression)."""
    return resolve_expression_columns(columns, gene_map)


def _write_compact_parquet(df: pd.DataFrame, path: Path) -> float:
    """float32 + zstd for Git-safe DepMap slices."""
    out = df.astype(np.float32, copy=False)
    out.to_parquet(path, compression="zstd", compression_level=9)
    return path.stat().st_size / (1024 * 1024)


def ovarian_model_ids(sample_info: pd.DataFrame) -> List[str]:
    mask = sample_info["OncotreeLineage"] == OVARIAN_LINEAGE
    return sample_info.index[mask].astype(str).tolist()


def first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.is_file():
            return p
    return None


def omics_expression_path() -> Optional[Path]:
    return first_existing(
        [REPO_ROOT / "OmicsExpression.parquet", DEPMAP_CACHE / "OmicsExpression.parquet"]
    )


def required_input_paths() -> List[Tuple[str, Path]]:
    """Inputs that must exist before a full export run."""
    paths: List[Tuple[str, Path]] = [
        ("DepMap sample_info", DEPMAP_CACHE / "sample_info.parquet"),
        ("DepMap crispr_gene_effect", DEPMAP_CACHE / "crispr_gene_effect.parquet"),
        ("DepMap mutations", DEPMAP_CACHE / "mutations.parquet"),
        ("DepMap cna", DEPMAP_CACHE / "cna.parquet"),
        ("GSE165897 UMI counts", GSE_DIR / "GSE165897_UMIcounts_HGSOC.tsv.gz"),
        ("GSE165897 cellInfo", GSE_DIR / "GSE165897_cellInfo_HGSOC.tsv.gz"),
        ("ovarian constants module", CONSTANTS_PATH),
    ]
    expr = omics_expression_path()
    if expr is not None:
        paths.append(("OmicsExpression", expr))
    return paths


def dry_check_required_paths(*, quiet_ok: bool = False) -> int:
    """Verify required inputs on disk; return 0 if all present."""
    missing: List[Tuple[str, Path]] = []
    for label, path in required_input_paths():
        if not path.is_file():
            missing.append((label, path))
    if missing:
        for label, path in missing:
            print(f"MISSING [{label}]: {path}", file=sys.stderr)
        if omics_expression_path() is None:
            print(
                f"MISSING [OmicsExpression]: "
                f"{REPO_ROOT / 'OmicsExpression.parquet'} OR "
                f"{DEPMAP_CACHE / 'OmicsExpression.parquet'}",
                file=sys.stderr,
            )
        return 1
    if not quiet_ok:
        print("dry-check OK (all required inputs present):")
        for label, path in required_input_paths():
            print(f"  OK  {label}: {path}")
    return 0


def export_expression(ov_model_ids: List[str], gene_map: Dict[str, int], si: pd.DataFrame) -> Dict[str, Any]:
    expr_path = omics_expression_path()
    if expr_path is None:
        raise FileNotFoundError(
            f"OmicsExpression.parquet not found at {REPO_ROOT / 'OmicsExpression.parquet'} "
            f"or {DEPMAP_CACHE / 'OmicsExpression.parquet'}"
        )

    expr = pd.read_parquet(expr_path)
    expr.index = expr.index.astype(str)
    ov_ids = [m for m in ov_model_ids if m in expr.index]
    ov_expr = expr.loc[ov_ids].copy()
    ov_expr.index.name = "ModelID"

    all_path = OUT_DIR / "ovarian_expression_all_genes.parquet"
    size_mb = _write_compact_parquet(ov_expr, all_path)
    print(f"  expression all_genes: {ov_expr.shape[0]} models × {ov_expr.shape[1]} genes, {size_mb:.2f} MB")

    col_map = resolve_expression_columns(ov_expr.columns, gene_map)
    key_cols = [col_map[s] for s in sorted(col_map.keys())]
    key_df = ov_expr[key_cols].copy()
    key_df.columns = [c.split(" (")[0] for c in key_cols]
    key_df = key_df.reset_index()
    key_path = OUT_DIR / "ovarian_expression_key_panel.csv"
    key_df.to_csv(key_path, index=False)

    # v10 aliases → all-genes parquet + key-panel csv (documented in MANIFEST)
    shutil.copy2(all_path, OUT_DIR / "ovarian_expression.parquet")
    shutil.copy2(key_path, OUT_DIR / "ovarian_expression.csv")

    donors = si.loc[ov_ids, "PatientID"].nunique() if "PatientID" in si.columns else None
    ref_donors = si.loc[ov_ids].dropna(subset=["PatientID"]).groupby("PatientID").size()
    n_ref = int((ref_donors == 1).sum()) if len(ref_donors) else None
    return {
        "n_expr_models": len(ov_ids),
        "n_expr_genes_all": ov_expr.shape[1],
        "n_key_genes": len(col_map),
        "n_donors": int(donors) if donors is not None else None,
        "n_reference_donors_single_model": n_ref,
        "source": str(expr_path),
    }


def export_crispr(ov_model_ids: List[str], gene_map: Dict[str, int]) -> Dict[str, Any]:
    cr_path = DEPMAP_CACHE / "crispr_gene_effect.parquet"
    if not cr_path.is_file():
        raise FileNotFoundError(cr_path)

    cr = pd.read_parquet(cr_path)
    cr.index = cr.index.astype(str)
    ov_ids = [m for m in ov_model_ids if m in cr.index]
    ov_cr = cr.loc[ov_ids].copy()
    ov_cr.index.name = "ModelID"

    all_path = OUT_DIR / "ovarian_crispr_all_genes.parquet"
    size_mb = _write_compact_parquet(ov_cr, all_path)
    print(f"  crispr all_genes: {ov_cr.shape[0]} models × {ov_cr.shape[1]} genes, {size_mb:.2f} MB")

    col_map = resolve_crispr_columns(ov_cr.columns, gene_map)
    key_df = ov_cr[[col_map[s] for s in sorted(col_map.keys())]].copy()
    key_df.columns = sorted(col_map.keys())
    key_df = key_df.reset_index()
    key_path = OUT_DIR / "ovarian_crispr_key_panel.csv"
    key_df.to_csv(key_path, index=False)

    shutil.copy2(all_path, OUT_DIR / "ovarian_crispr.parquet")
    shutil.copy2(key_path, OUT_DIR / "ovarian_crispr.csv")

    return {"n_crispr_models": len(ov_ids), "n_crispr_genes_all": ov_cr.shape[1], "n_key_genes": len(col_map)}


def export_mutations_cna(ov_model_ids: List[str], gene_map: Dict[str, int]) -> Dict[str, Any]:
    mut_path = DEPMAP_CACHE / "mutations.parquet"
    cna_path = DEPMAP_CACHE / "cna.parquet"
    if not mut_path.is_file():
        raise FileNotFoundError(mut_path)
    if not cna_path.is_file():
        raise FileNotFoundError(cna_path)

    mutations = pd.read_parquet(mut_path, columns=["ModelID", "HugoSymbol", "LikelyLoF", "Hotspot", "VepImpact"])
    mutations = mutations[mutations["ModelID"].isin(ov_model_ids)]

    cna = pd.read_parquet(cna_path)
    cna.index = cna.index.astype(str)

    # Entrez for mut/cna genes from expression column parse or gene_map
    entrez_fallback = {
        "ARID1A": 8289,
        "PIK3CA": 5290,
        "MYC": 4609,
        "TP53": 7157,
        "MBD4": 8930,
        "CCNE1": 898,
        "BRCA1": 672,
        "BRCA2": 675,
        "NF1": 4763,
        "RB1": 5925,
        "PTEN": 5728,
        "KRAS": 3845,
        "BRAF": 673,
    }
    for g in MUT_CNA_GENES:
        if g in gene_map:
            entrez_fallback[g] = gene_map[g]

    rows = []
    for model_id in sorted(set(ov_model_ids) & set(cna.index)):
        row: Dict[str, Any] = {"ModelID": model_id}
        sub = mutations[mutations["ModelID"] == model_id]
        for gene in MUT_CNA_GENES:
            gsub = sub[sub["HugoSymbol"] == gene]
            has_mut = len(gsub) > 0
            has_lof = bool(gsub["LikelyLoF"].any()) if has_mut and "LikelyLoF" in gsub.columns else False
            has_hotspot = bool(gsub["Hotspot"].any()) if has_mut and "Hotspot" in gsub.columns else False
            row[f"{gene}_any_mutation"] = int(has_mut)
            row[f"{gene}_likely_lof"] = int(has_lof)
            row[f"{gene}_hotspot"] = int(has_hotspot)
            eid = entrez_fallback.get(gene)
            cn_col = f"{gene} ({eid})" if eid and f"{gene} ({eid})" in cna.columns else None
            if cn_col is None:
                matches = [c for c in cna.columns if c.startswith(f"{gene} (")]
                cn_col = matches[0] if matches else None
            row[f"{gene}_cn_log2"] = float(cna.loc[model_id, cn_col]) if cn_col else np.nan
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = OUT_DIR / "ovarian_mutations_and_cna.csv"
    out_df.to_csv(out_path, index=False)
    return {"n_mut_cna_models": len(out_df), "n_mut_cna_genes": len(MUT_CNA_GENES)}


def _compartment_mask(cell_info: pd.DataFrame, compartment: str) -> pd.Series:
    if compartment == "malignant_tumor":
        return cell_info["cell_type"] == "EOC"
    if compartment == "stromal_CAF":
        return (cell_info["cell_type"] == "Stromal") & cell_info["cell_subtype"].str.startswith("CAF", na=False)
    if compartment == "immune_T_myeloid":
        return (cell_info["cell_type"] == "Immune") & cell_info["cell_subtype"].isin(["T-cells", "Macrophages"])
    raise ValueError(compartment)


def export_gse165897(gene_symbols: List[str], patient_table: List[dict]) -> Dict[str, Any]:
    umi_path = GSE_DIR / "GSE165897_UMIcounts_HGSOC.tsv.gz"
    info_path = GSE_DIR / "GSE165897_cellInfo_HGSOC.tsv.gz"
    if not umi_path.is_file() or not info_path.is_file():
        raise FileNotFoundError(f"Missing GSE165897 files under {GSE_DIR}")

    cell_info = pd.read_csv(info_path, sep="\t", compression="gzip")
    counts = pd.read_csv(umi_path, sep="\t", compression="gzip", index_col=0)

    pt_lookup = {r["patient_id"]: r for r in patient_table}
    phases = {"treatment-naive": "Pre", "post-NACT": "Post"}
    compartments = ["malignant_tumor", "stromal_CAF", "immune_T_myeloid"]

    genes_avail = [g for g in gene_symbols if g in counts.index]
    missing_genes = [g for g in gene_symbols if g not in counts.index]

    rows: List[Dict[str, Any]] = []
    for patient_id in sorted(pt_lookup.keys()):
        meta = pt_lookup[patient_id]
        for phase_raw, phase_label in phases.items():
            for compartment in compartments:
                mask = (
                    (cell_info["patient_id"] == patient_id)
                    & (cell_info["treatment_phase"] == phase_raw)
                )
                comp_mask = _compartment_mask(cell_info, compartment)
                cell_list = cell_info.loc[mask & comp_mask, "cell"].astype(str).tolist()
                if not cell_list:
                    continue
                valid_cells = [c for c in cell_list if c in counts.columns]
                if not valid_cells:
                    continue
                cell_counts = counts.loc[:, valid_cells]
                summed = cell_counts.sum(axis=1)
                total = float(summed.sum())
                if total <= 0:
                    log2_cpm = pd.Series(0.0, index=genes_avail)
                else:
                    cpm = (summed.reindex(genes_avail).fillna(0) / total) * 1e6
                    log2_cpm = np.log2(cpm + 1.0)
                rec = {
                    "patient_id": patient_id,
                    "timepoint": phase_label,
                    "treatment_phase_raw": phase_raw,
                    "compartment": compartment,
                    "n_cells": int(len(valid_cells)),
                    "pfi_days": meta.get("pfi_days"),
                    "four_state_class": meta.get("four_state_class"),
                }
                for g in genes_avail:
                    rec[g] = float(log2_cpm.get(g, 0.0))
                rows.append(rec)

    out_df = pd.DataFrame(rows)
    out_path = OUT_DIR / "gse165897_human_paired_compartments.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  GSE165897 pseudobulks: {len(out_df)} rows ({len(pt_lookup)} patients × compartments)")
    if missing_genes:
        print(f"  WARN: {len(missing_genes)} key genes absent from GSE matrix")

    return {
        "n_gse_rows": len(out_df),
        "n_gse_patients": len(pt_lookup),
        "n_gse_genes_exported": len(genes_avail),
        "n_gse_genes_missing": len(missing_genes),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_stats(path: Path) -> Dict[str, Any]:
    st: Dict[str, Any] = {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        st["rows"] = len(df)
        st["cols"] = len(df.columns)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
        st["rows"] = len(df)
        st["cols"] = len(df.columns)
    return st


def write_manifest(
    files: List[Path],
    meta: Dict[str, Any],
    *,
    tarball_info: Optional[Dict[str, Any]] = None,
) -> Path:
    manifest = {
        "created_by": "scripts/export_phylo_v11_inputs.py",
        "alias_notes": {
            "ovarian_expression.parquet": "copy of ovarian_expression_all_genes.parquet",
            "ovarian_expression.csv": "copy of ovarian_expression_key_panel.csv",
            "ovarian_crispr.parquet": "copy of ovarian_crispr_all_genes.parquet",
            "ovarian_crispr.csv": "copy of ovarian_crispr_key_panel.csv",
        },
        "export_counts": meta,
        "files": [file_stats(p) for p in sorted(files, key=lambda x: x.name)],
    }
    if tarball_info is not None:
        manifest["phylo_upload_tarball"] = tarball_info
    manifest_path = OUT_DIR / "MANIFEST_SHA256.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def _tarball_size_mb(member_paths: List[Path], manifest_path: Path) -> float:
    """On-disk trial packed size (gzip) for candidate members + manifest."""
    members = list(member_paths)
    if manifest_path not in members and manifest_path.is_file():
        members = members + [manifest_path]

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tf:
        tpath = tf.name
    try:
        with tarfile.open(tpath, "w:gz") as tar:
            for p in members:
                if p.name in TAR_SKIP_ALIASES:
                    continue
                tar.add(p, arcname=p.name)
        return os.path.getsize(tpath) / (1024 * 1024)
    finally:
        Path(tpath).unlink(missing_ok=True)


def _select_tar_members() -> Tuple[List[Path], List[str]]:
    """Pick files for Phylo upload tarball under MAX_TAR_MB."""
    by_name = {p.name: p for p in OUT_DIR.iterdir() if p.is_file() and p.name != TARBALL_NAME}
    included: List[Path] = []
    omitted: List[str] = []

    manifest_path = OUT_DIR / "MANIFEST_SHA256.json"
    for name in TAR_MEMBER_PRIORITY:
        p = by_name.get(name)
        if p is None:
            if name != "MANIFEST_SHA256.json":
                omitted.append(f"{name} (missing)")
            continue
        trial = included + [p]
        if _tarball_size_mb(trial, manifest_path) <= MAX_TAR_MB:
            included.append(p)
        else:
            omitted.append(f"{name} (exceeds {MAX_TAR_MB} MB cap with current set)")

    for name in sorted(TAR_SKIP_ALIASES):
        if name in by_name and name not in [p.name for p in included]:
            omitted.append(f"{name} (v10 alias; folder-only)")

    return included, omitted


def build_tarball(manifest_path: Path) -> Tuple[Path, List[str], List[str]]:
    tar_path = OUT_DIR / TARBALL_NAME
    if tar_path.is_file():
        tar_path.unlink()

    members, omitted = _select_tar_members()
    if manifest_path not in members:
        members = [manifest_path] + [p for p in members if p != manifest_path]

    with tarfile.open(tar_path, "w:gz") as tar:
        for p in members:
            if p.name in TAR_SKIP_ALIASES:
                continue
            tar.add(p, arcname=p.name)

    mb = tar_path.stat().st_size / (1024 * 1024)
    member_names = [p.name for p in members]
    print(f"  tarball: {tar_path.name} = {mb:.2f} MB ({len(member_names)} members)")
    if omitted:
        print(f"  tarball omitted ({len(omitted)}): {omitted[:6]}{'...' if len(omitted) > 6 else ''}")
    if mb > MAX_TAR_MB:
        raise RuntimeError(f"tarball {mb:.2f} MB exceeds {MAX_TAR_MB} MB cap")
    if mb > TARGET_TAR_MB:
        print(f"  NOTE: tarball above {TARGET_TAR_MB} MB target but under hard cap")
    return tar_path, member_names, omitted


def rebuild_tarball_only() -> int:
    """Refresh manifest tarball metadata without re-reading DepMap / GSE."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "MANIFEST_SHA256.json"
    if not manifest_path.is_file():
        print(f"FATAL: run full export first (no {manifest_path})", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text())
    meta: Dict[str, Any] = manifest.get("export_counts") or {}
    export_files = [
        p for p in OUT_DIR.iterdir() if p.is_file() and p.name != TARBALL_NAME
    ]
    tar_path, tar_members, tar_omitted = build_tarball(manifest_path)
    tarball_info = {
        "filename": TARBALL_NAME,
        "bytes": tar_path.stat().st_size,
        "sha256": sha256_file(tar_path),
        "members": tar_members,
        "omitted_from_tarball": tar_omitted,
        "max_mb": MAX_TAR_MB,
        "skip_aliases": sorted(TAR_SKIP_ALIASES),
    }
    write_manifest(export_files, meta, tarball_info=tarball_info)
    tar_path, tar_members, tar_omitted = build_tarball(manifest_path)
    tarball_info.update(
        {
            "bytes": tar_path.stat().st_size,
            "sha256": sha256_file(tar_path),
            "members": tar_members,
            "omitted_from_tarball": tar_omitted,
        }
    )
    write_manifest(export_files, meta, tarball_info=tarball_info)
    print("\n=== TARBALL-ONLY SUMMARY ===")
    print(f"tarball: {tar_path}")
    print(f"sha256:  {sha256_file(tar_path)}")
    print(f"bytes:   {tar_path.stat().st_size}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-check",
        action="store_true",
        help="Verify required input paths exist, then exit (no export).",
    )
    parser.add_argument(
        "--tarball-only",
        action="store_true",
        help="Rebuild upload tarball + manifest from existing data/phylo_v11_inputs/.",
    )
    args = parser.parse_args()

    if args.dry_check:
        return dry_check_required_paths()

    if not args.tarball_only:
        rc = dry_check_required_paths(quiet_ok=True)
        if rc != 0:
            return rc

    if args.tarball_only:
        return rebuild_tarball_only()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    constants = _load_constants()
    gene_map = build_key_panel_gene_map(constants)
    print(f"Key panel: {len(gene_map)} genes (STRESS_34 + PROLIF_106 + DDR_4 + Alpha extras)")

    sample_info = pd.read_parquet(DEPMAP_CACHE / "sample_info.parquet")
    ov_all = ovarian_model_ids(sample_info)
    print(f"Ovarian lineage models in sample_info: {len(ov_all)}")

    meta: Dict[str, Any] = {}
    meta.update(export_expression(ov_all, gene_map, sample_info))
    meta.update(export_crispr(ov_all, gene_map))
    meta.update(export_mutations_cna(ov_all, gene_map))
    meta.update(
        export_gse165897(sorted(gene_map.keys()), constants.GSE165897_11_PATIENT_TABLE)
    )

    export_files = [
        p
        for p in OUT_DIR.iterdir()
        if p.is_file() and p.name != TARBALL_NAME
    ]
    manifest_path = write_manifest(export_files, meta)
    tar_path, tar_members, tar_omitted = build_tarball(manifest_path)
    tarball_info = {
        "filename": TARBALL_NAME,
        "bytes": tar_path.stat().st_size,
        "sha256": sha256_file(tar_path),
        "members": tar_members,
        "omitted_from_tarball": tar_omitted,
        "max_mb": MAX_TAR_MB,
        "skip_aliases": sorted(TAR_SKIP_ALIASES),
    }
    write_manifest(export_files, meta, tarball_info=tarball_info)
    tar_path, tar_members, tar_omitted = build_tarball(manifest_path)
    tarball_info.update(
        {
            "bytes": tar_path.stat().st_size,
            "sha256": sha256_file(tar_path),
            "members": tar_members,
            "omitted_from_tarball": tar_omitted,
        }
    )
    write_manifest(export_files, meta, tarball_info=tarball_info)

    print("\n=== EXPORT SUMMARY ===")
    print(json.dumps(meta, indent=2))
    print(f"tarball: {tar_path}")
    print(f"sha256:  {sha256_file(tar_path)}")
    print(f"bytes:   {tar_path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
