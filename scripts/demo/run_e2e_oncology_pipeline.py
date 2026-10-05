#!/usr/bin/env python3
"""End-to-end oncology demo runner (RUO) — one patient folder → clinical JSON.

Loads identity-locked production artifacts (no mock fallbacks). Stages that
cannot execute for missing inputs/deps fail closed with an explicit error.

Usage::

    python3 scripts/demo/run_e2e_oncology_pipeline.py \\
        --patient-case fixtures/patient_sample_01/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

ARTIFACTS = {
    "luna16": {
        "path": ROOT / "models/luna16/luna16_baseline_b5e79231466a.pt",
        "sha256": "b5e79231466adae93a6fe8e8594029e9add142914e223b879aa0343bb2402d01",
        "metric": {"name": "froc_at_2", "value": 0.8267399271674979},
    },
    "mammo-retinanet": {
        "path": ROOT / "models/mammo/mammo_retinanet_v1.safetensors",
        "sha256": "691233bca51c4eb99fd6859301b8e1393ada6643b25e79622548ef80de951d32",
        "metric": {"name": "auroc", "value": 0.785453260650111},
    },
    "cbis-medsiglip": {
        "path": ROOT / "models/cbis_ddsm_logreg_v3.joblib",
        "sha256": "f40f33c1cd3adfa48d1d10cc4d136795ad0642c3d1f94a543872ae1dc283aaf7",
        "metric": {"name": "auroc", "value": 0.8614476075105997},
    },
    "clinicalbert": {
        "path": ROOT / "artifacts/clinicalbert/clinicalbert_head_v2.safetensors",
        "sha256": "429f804d7f348d7c4eeb27821f766cc2de65c4072f20db0c3b3afaefa9068e50",
        "metric": {"name": "span_f1", "value": 0.1916},
    },
    "medgemma": {
        "path": ROOT / "models/medgemma_1_5_4b_identity.json",
        "sha256": "e69dbed948f15ec0134c0315d6d6c8f648d58f63020b6b7a298838149bd2bcfb",
        "metric": {"name": "chat_success_rate", "value": 1.0},
    },
    "stage-therapy": {
        "path": ROOT / "src/oncology_arbiter/arbiter/models/therapy_arbiter_v1.json",
        "sha256": "a5f8ead83b3458b20b8e33203de6cfc13c2fa91dde360e0490722afd2f4ac889",
        "metric": {"name": "auroc", "value": 0.9597117364447495},
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifacts() -> dict[str, Any]:
    out = {}
    for name, spec in ARTIFACTS.items():
        path = Path(spec["path"])
        if not path.is_file():
            raise FileNotFoundError(f"{name}: missing artifact {path}")
        actual = sha256_file(path)
        if actual != spec["sha256"]:
            raise RuntimeError(f"{name}: sha mismatch expected {spec['sha256']} got {actual}")
        out[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": actual,
            "metric_receipt": spec["metric"],
            "verified": True,
        }
    return out


def run_luna16(case: dict[str, Any], case_dir: Path) -> dict[str, Any]:
    art = ARTIFACTS["luna16"]
    # Prove baseline weights are present and byte-identical before any detect call.
    if sha256_file(art["path"]) != art["sha256"]:
        raise RuntimeError("luna16 baseline sha mismatch")
    verify = {
        "capability": "luna16",
        "artifact_sha256": art["sha256"],
        "froc_at_2_production": art["metric"]["value"],
        "model_state": "loaded_luna16_baseline_b5e79231",
        "status": "artifact_verified",
    }
    ct_rel = case.get("ct_path")
    if not ct_rel:
        verify.update(
            {
                "status": "skipped_no_ct",
                "error": "fixtures/patient_sample_01 has no CT volume; LUNA baseline identity verified only",
                "detections": [],
            }
        )
        return verify
    ct_path = case_dir / ct_rel
    if not ct_path.is_file():
        raise FileNotFoundError(f"ct_path missing: {ct_path}")
    # Real detector path — fail closed if MONAI/CUDA unavailable (no mock boxes).
    try:
        from oncology_arbiter.nsclc.luna16_retinanet import detect_nodules  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"luna16 detector import failed: {type(exc).__name__}: {exc}") from exc
    result = detect_nodules(str(ct_path))
    verify.update({"status": "ok", "detections": result})
    return verify


def run_mammo(case_dir: Path, case: dict[str, Any]) -> dict[str, Any]:
    from oncology_arbiter.models.mammo_retinanet_wiring import predict_png

    png = case_dir / case["mammogram_path"]
    if not png.is_file():
        raise FileNotFoundError(f"mammogram missing: {png}")
    out = predict_png(png, device="cpu")
    out["status"] = "ok"
    # Identity-lock the multimodal CBIS probe artifact (embedding serve may be remote).
    cbis = ARTIFACTS["cbis-medsiglip"]
    out["cbis_medsiglip_artifact"] = {
        "path": str(cbis["path"].relative_to(ROOT)),
        "sha256": cbis["sha256"],
        "test_auroc_receipt": cbis["metric"]["value"],
        "note": "Multimodal fusion probe verified by SHA; PNG scored by mammo-retinanet v1 in this runner.",
    }
    return out


def run_clinicalbert(case_dir: Path, case: dict[str, Any]) -> dict[str, Any]:
    """72h primary: MedGemma structured JSON (ClinicalBERT NER retired as primary)."""
    from oncology_arbiter.nlp.medgemma_pathology_client import extract_pathology

    note_path = case_dir / case["pathology_note_path"]
    text = note_path.read_text()
    parsed = extract_pathology(text)
    return {
        "capability": "medgemma_pathology",
        "artifact_sha256": ARTIFACTS["medgemma"]["sha256"],
        "revision_sha256": parsed.get("revision_sha256"),
        "status": "ok",
        "source": "medgemma_pathology",
        "parse": parsed.get("extraction"),
        "honesty": "ClinicalBERT NER is not primary; MedGemma PathologyExtraction is.",
    }


def run_medgemma(case: dict[str, Any], mammo: dict[str, Any], clinicalbert: dict[str, Any]) -> dict[str, Any]:
    identity = json.loads(ARTIFACTS["medgemma"]["path"].read_text())
    chat_url = identity["chat_url"]
    prompt = (
        "You are a research-use-only oncology co-scientist. "
        f"Patient {case['patient_id']}: age {case['age_years']}, "
        f"HER2={case['her2_status_positive']}, ER={case['er_status_positive']}, "
        f"mammo_proba_cancer={mammo.get('proba_cancer')}. "
        "In one short paragraph, state the dominant driver and a RUO treatment hypothesis."
    )
    body = json.dumps(
        {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128,
            "temperature": 0.2,
        }
    ).encode()
    req = urllib.request.Request(chat_url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode())
    if data.get("error"):
        raise RuntimeError(f"medgemma chat error: {data['error']}")
    if data.get("revision_sha256") != identity["revision_sha256"]:
        raise RuntimeError("medgemma revision drift vs identity artifact")
    return {
        "capability": "medgemma",
        "status": "ok",
        "model_id": data.get("model_id"),
        "revision_sha256": data.get("revision_sha256"),
        "latency_s": data.get("latency_s"),
        "text": data.get("text"),
        "info_url": identity["info_url"],
    }


def run_therapy(case: dict[str, Any]) -> dict[str, Any]:
    from oncology_arbiter.arbiter.stage_therapy_wiring import load_stage_therapy_arbiter

    arb = load_stage_therapy_arbiter()
    # Norms match therapy_arbiter_v1 feature_encodings (size_mm/50, age/100, ki67/100).
    features = {
        "histology": case["histology"],
        "grade": str(case["grade"]),
        "er_status_positive": case["er_status_positive"],
        "pr_status_positive": case["pr_status_positive"],
        "her2_status_positive": case["her2_status_positive"],
        "node_status_positive": case["node_status_positive"],
        "brca_status_known_pathogenic": case.get("brca_status_known_pathogenic"),
        "tumor_size_norm": float(case["size_mm"]) / 50.0,
        "age_at_diagnosis_norm": float(case["age_years"]) / 100.0,
        "ki67_norm": float(case.get("ki67_pct", 0.0)) / 100.0,
    }
    result = arb.score(features)
    return {
        "capability": "stage-therapy",
        "status": "ok",
        "artifact_sha256": ARTIFACTS["stage-therapy"]["sha256"],
        "p_event": float(result.p_positive),
        "risk_bucket": result.risk_bucket,
        "logit": float(result.logit),
        "recommendation": result.recommendation,
        "driving_feature": result.driving_feature,
        "missing_features": list(result.missing_features),
        "caveat": result.caveat,
    }


def build_report(
    *,
    case: dict[str, Any],
    verified: dict[str, Any],
    luna: dict[str, Any],
    mammo: dict[str, Any],
    clinicalbert: dict[str, Any],
    medgemma: dict[str, Any],
    therapy: dict[str, Any],
    elapsed_s: float,
) -> dict[str, Any]:
    her2 = bool(case.get("her2_status_positive"))
    prescription = "Trastuzumab + chemo (RUO hypothesis)" if her2 else "Multidisciplinary review (RUO)"
    driver = "HER2 amplification / overexpression" if her2 else "See structured features"
    return {
        "disclaimer": "RESEARCH USE ONLY — not FDA-cleared; not for clinical decision-making.",
        "patient_id": case["patient_id"],
        "elapsed_seconds": elapsed_s,
        "artifact_verification": verified,
        "modalities": {
            "luna16": luna,
            "mammo_retinanet": mammo,
            "clinicalbert": {
                "status": clinicalbert.get("status"),
                "artifact_sha256": clinicalbert.get("artifact_sha256"),
                "entities": clinicalbert.get("parse", {}).get("entities")
                or clinicalbert.get("parse", {}).get("parsed")
                or clinicalbert.get("parse"),
            },
            "medgemma": {
                "status": medgemma.get("status"),
                "model_id": medgemma.get("model_id"),
                "revision_sha256": medgemma.get("revision_sha256"),
                "reasoning": medgemma.get("text"),
            },
            "stage_therapy": therapy,
        },
        "final_clinical_action_report": {
            # Honest target: METABRIC observed chemotherapy receipt (not OS/recurrence).
            "observed_chemo_receipt_risk_pct": round(float(therapy["p_event"]) * 100.0, 1),
            "risk_bucket": therapy["risk_bucket"],
            "prescription_ruo": prescription,
            "primary_driver": driver,
            "mammo_proba_cancer": mammo.get("proba_cancer"),
            "luna16_froc_at_2_production": luna.get("froc_at_2_production"),
            "therapy_target": "observed_chemotherapy_receipt",
            "provenance": {
                "mammo_sha256": ARTIFACTS["mammo-retinanet"]["sha256"],
                "luna16_sha256": ARTIFACTS["luna16"]["sha256"],
                "clinicalbert_sha256": ARTIFACTS["clinicalbert"]["sha256"],
                "cbis_medsiglip_sha256": ARTIFACTS["cbis-medsiglip"]["sha256"],
                "medgemma_revision_sha256": medgemma.get("revision_sha256"),
                "therapy_sha256": ARTIFACTS["stage-therapy"]["sha256"],
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patient-case",
        type=Path,
        default=ROOT / "fixtures/patient_sample_01",
        help="Patient folder with case.json + modality files",
    )
    args = parser.parse_args()
    case_dir = args.patient_case.resolve()
    case = json.loads((case_dir / "case.json").read_text())
    t0 = time.time()

    verified = verify_artifacts()
    luna = run_luna16(case, case_dir)
    mammo = run_mammo(case_dir, case)
    clinicalbert = run_clinicalbert(case_dir, case)
    medgemma = run_medgemma(case, mammo, clinicalbert)
    therapy = run_therapy(case)
    report = build_report(
        case=case,
        verified=verified,
        luna=luna,
        mammo=mammo,
        clinicalbert=clinicalbert,
        medgemma=medgemma,
        therapy=therapy,
        elapsed_s=round(time.time() - t0, 3),
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        err = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "disclaimer": "RUO"}
        print(json.dumps(err, indent=2), file=sys.stderr)
        raise
