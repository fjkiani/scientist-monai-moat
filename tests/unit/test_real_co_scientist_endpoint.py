"""Clinical-contract tests for the model-backed MedGemma Co-Scientist route."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.agents.supervisor import (
    AgentPhase,
    Hypothesis,
    _extract_json,
    _final_model_text,
    _hypothesis_to_query,
    plan_stage,
)
from oncology_arbiter.api.app import create_app
from oncology_arbiter.api.services.tumor_board import TumorBoardAssembler
from oncology_arbiter.auth.bootstrap import ENV_KEY_HASH, ENV_TENANT_ID, ENV_TENANT_NAME
from oncology_arbiter.models.llm_client import GemmaClient, LlmResponse
from oncology_arbiter.models.medgemma_1_5_4b_wiring import verify_artifact_identity

CLINICAL_KEY = "oa_live_" + "c0" * 16

CLINICAL_CONTEXT = {
    "patient": "De-identified woman in her 50s",
    "pathology": (
        "Invasive ductal carcinoma, grade 3, 22 mm; ER 95% positive, "
        "PR negative, HER2 IHC 1+ (negative), Ki-67 35%."
    ),
    "nodes": "Sentinel node with 3 mm macrometastasis in 1 of 3 nodes.",
    "clinical_question": (
        "Rank evidence-grounded hypotheses that materially affect adjuvant "
        "systemic therapy planning; identify what remains unproven."
    ),
}

HYPOTHESES = [
    {
        "claim": "Endocrine therapy is indicated because the tumor is strongly ER-positive.",
        "rationale": (
            "The pathology reports 95% ER staining. The exact regimen still "
            "depends on menopausal status and patient-specific contraindications."
        ),
        "source_scope": "molecular",
        "initial_confidence": 0.93,
    },
    {
        "claim": "HER2-directed therapy is not indicated by the reported IHC 1+ result.",
        "rationale": (
            "HER2 IHC 1+ is reported as negative in this early-stage specimen. "
            "This statement does not address later HER2-low metastatic indications."
        ),
        "source_scope": "molecular",
        "initial_confidence": 0.89,
    },
    {
        "claim": "The node-positive high-grade features warrant formal chemotherapy benefit assessment.",
        "rationale": (
            "Grade 3 disease and a nodal macrometastasis increase recurrence risk. "
            "A treatment decision requires complete staging and applicable genomic evidence."
        ),
        "source_scope": "clinical_history",
        "initial_confidence": 0.84,
    },
]


class ScriptedMedGemma:
    """Identity-bound response script; the real supervisor owns all phase logic."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.calls: list[list[dict[str, str]]] = []

    def next(self, messages: list[dict[str, str]]) -> LlmResponse:
        index = len(self.calls)
        if index >= len(self._texts):
            raise AssertionError(f"unexpected MedGemma call {index + 1}")
        self.calls.append(messages)
        identity = verify_artifact_identity()
        return LlmResponse(
            text=self._texts[index],
            route="medgemma_modal",
            model=identity["model_id"],
            prompt_tokens=100 + index,
            completion_tokens=40 + index,
            thinking_tokens=0,
            est_cost_usd=0.0,
            latency_s=0.02,
            app_version=identity["app_version"],
            request_id=f"medgemma-clinical-{index + 1:02d}",
            honesty_warning=identity["honesty_warning"],
            revision_sha256=identity["revision_sha256"],
        )


def _success_script(
    *,
    malformed_pair: bool = False,
    malformed_reflection: bool = False,
    winner_only_pair: bool = False,
) -> ScriptedMedGemma:
    pair_two = (
        "not valid tournament JSON"
        if malformed_pair
        else json.dumps({"winner": "A"})
        if winner_only_pair
        else json.dumps({
            "winner": "B",
            "margin": "moderate",
            "reason": "The HER2 status directly excludes an otherwise distinct treatment class.",
        })
    )
    texts = [
        json.dumps({"hypotheses": HYPOTHESES}),
        (
            "unfinished reflection without final JSON"
            if malformed_reflection
            else json.dumps({
                "critique": (
                    "The fetched endocrine guideline supports treatment "
                    "for ER-positive disease."
                ),
                "evidence_alignment": "supports",
            })
        ),
        json.dumps({
            "critique": "The fetched HER2 guideline supports interpreting IHC 1+ as HER2-negative.",
            "evidence_alignment": "supports",
        }),
        json.dumps({
            "critique": (
                "Node-positive grade 3 disease has retrieved evidence that a "
                "chemotherapy benefit assessment is warranted."
            ),
            "evidence_alignment": "partial",
        }),
        json.dumps({
            "winner": "A",
            "margin": "clear",
            "reason": "The endocrine indication is directly supported by receptor status and fetched guidance.",
        }),
        pair_two,
        json.dumps({
            "winner": "A",
            "margin": "marginal",
            "reason": "The compared claim has more directly retrieved evidence in this run.",
        }),
    ]
    if not malformed_pair and not winner_only_pair:
        texts.append(
            "Fetched evidence supports endocrine therapy and confirms the reported HER2-negative "
            "classification. Node-positive grade 3 features justify a formal chemotherapy "
            "benefit assessment using the retrieved evidence. Complete staging and menopausal "
            "status remain required. This is a research synthesis rather than a treatment directive."
        )
    return ScriptedMedGemma(texts)


def _install_model_and_evidence(
    monkeypatch: pytest.MonkeyPatch,
    script: ScriptedMedGemma,
) -> None:
    def fake_chat(
        _client: GemmaClient,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> LlmResponse:
        del max_tokens, temperature
        return script.next(messages)

    evidence_calls = 0

    def fake_retrieve(
        _stage: str,
        hypothesis: Any,
        seen_urls: set[str],
        *,
        context: dict[str, Any] | None = None,
        errors: list[str] | None = None,
        max_hits: int = 3,
    ) -> list[dict[str, Any]]:
        del errors, max_hits
        nonlocal evidence_calls
        assert context and "Invasive ductal carcinoma" in context["pathology"]
        evidence_calls += 1
        if evidence_calls == 1:
            url = "https://pubmed.ncbi.nlm.nih.gov/29863451/"
            title = "Adjuvant endocrine therapy evidence in hormone-receptor-positive breast cancer"
            snippet = "Endocrine therapy reduces recurrence in hormone-receptor-positive early breast cancer."
        elif evidence_calls == 2:
            url = "https://pubmed.ncbi.nlm.nih.gov/37284804/"
            title = "HER2 testing recommendations for breast cancer"
            snippet = "IHC 0 or 1+ is classified as HER2-negative under the tested early-stage criteria."
        else:
            url = "https://pubmed.ncbi.nlm.nih.gov/24799467/"
            title = "Chemotherapy benefit in node-positive early breast cancer"
            snippet = "Node-positive high-grade disease warrants a formal chemotherapy benefit assessment."
        hypothesis.evidence_urls.append(url)
        seen_urls.add(url)
        records = [{
            "url": url,
            "title": title,
            "snippet": snippet,
            "source": "pubmed_search",
            "hypothesis_id": hypothesis.hypothesis_id,
        }]
        if evidence_calls == 1:
            records.append({
                "url": "https://unfetched.invalid/hallucinated",
                "title": "Unfetched record",
                "snippet": "Must be removed by the honesty gate.",
                "source": "untrusted",
                "hypothesis_id": hypothesis.hypothesis_id,
            })
        return records

    monkeypatch.setattr(GemmaClient, "chat", fake_chat)
    monkeypatch.setattr(
        "oncology_arbiter.agents.supervisor._retrieve_evidence",
        fake_retrieve,
    )


def _enable_verified_auth(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> dict[str, str]:
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_MODE", "on")
    monkeypatch.setenv("ONCOLOGY_ARBITER_SKIP_DEMO_PREWARM", "1")
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_DB_PATH", str(tmp_path / "tenants.sqlite"))
    monkeypatch.setenv(ENV_TENANT_ID, "clinical-contract")
    monkeypatch.setenv(ENV_TENANT_NAME, "clinical-contract")
    monkeypatch.setenv(ENV_KEY_HASH, hashlib.sha256(CLINICAL_KEY.encode()).hexdigest())
    return {"X-API-Key": CLINICAL_KEY}


def _post(client: TestClient, headers: dict[str, str] | None = None) -> Any:
    return client.post(
        "/v1/co-scientist/run",
        headers=headers or {},
        json={
            "stage": "biopsy",
            "context": CLINICAL_CONTEXT,
            "n_hypotheses": 3,
            "n_evidence_top_k": 2,
        },
    )


def test_medgemma_reasoning_wrapper_yields_final_pairwise_json() -> None:
    raw = """<unused94>thought
    I should return {winner: 'A'} after comparing the evidence.
    <unused95>final
    {"winner":"A","margin":"clear","reason":"Fetched evidence supports A."}
    """
    assert _extract_json(raw) == {
        "winner": "A",
        "margin": "clear",
        "reason": "Fetched evidence supports A.",
    }


def test_medgemma_thought_json_is_salvaged_when_final_marker_missing() -> None:
    raw = """<unused94>thought
    {"winner":"B","margin":"moderate","reason":"Fetched evidence supports B."}
    """
    assert _extract_json(raw) == {
        "winner": "B",
        "margin": "moderate",
        "reason": "Fetched evidence supports B.",
    }


def test_medgemma_turn_terminator_spam_is_removed_from_meta_review() -> None:
    assert _final_model_text(
        "Evidence supports endocrine therapy.<end_of_turn><end_of_turn>"
    ) == "Evidence supports endocrine therapy."
    assert _final_model_text("<unused94>thought\nunfinished reasoning") == ""


def test_evidence_query_includes_cancer_context_for_generic_biomarker() -> None:
    hypothesis = Hypothesis(
        hypothesis_id="clinical-query",
        claim="High Ki-67 suggests a higher recurrence risk.",
        rationale="Proliferation marker.",
        source_scope="molecular",
        initial_confidence=0.8,
    )
    query = _hypothesis_to_query(hypothesis, context=CLINICAL_CONTEXT)
    assert "Ki-67" in query
    assert "Invasive" in query
    assert "ductal" in query
    assert "carcinoma" in query


def test_real_endpoint_runs_all_model_phases_and_emits_elo_receipts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    headers = _enable_verified_auth(monkeypatch, tmp_path)
    monkeypatch.setenv(
        "MEDGEMMA_MODAL_URL",
        "https://crispro-test--medgemma-1-5-4b-chat.modal.run",
    )
    script = _success_script()
    _install_model_and_evidence(monkeypatch, script)

    response = _post(TestClient(create_app()), headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["provenance"]["model_state"] == "loaded_medgemma_1p5_4b"
    assert body["provider"] == "medgemma_modal"
    identity = verify_artifact_identity()
    assert body["model_id"] == identity["model_id"]
    assert body["app_name"] == identity["app_version"]
    assert body["model_revision"] == identity["revision_sha256"]
    assert body["tournament_pairs_expected"] == 3
    assert body["tournament_pairs_completed"] == 3
    assert body["reflection_calls_expected"] == 3
    assert body["reflection_calls_completed"] == 3
    assert len(body["reflection_receipts"]) == 3
    assert body["tokens_in"] > 0
    assert body["tokens_out"] > 0
    assert all(item["model_request_id"] for item in body["reflection_receipts"])
    assert len(body["tournament_matches"]) == 3
    assert all(match["completed"] for match in body["tournament_matches"])
    assert all(match["model_request_id"] for match in body["tournament_matches"])
    assert all(
        match["elo_a_before"] != match["elo_a_after"]
        or match["elo_b_before"] != match["elo_b_after"]
        for match in body["tournament_matches"]
    )
    assert max(item["elo"] for item in body["hypotheses"]) > 1500.0
    assert min(item["elo"] for item in body["hypotheses"]) < 1500.0
    assert body["meta_review"].startswith("Fetched evidence supports")
    assert body["meta_review_request_id"]
    assert body["meta_review_response_sha256"]

    returned_urls = {item["url"] for item in body["evidence"]}
    assert returned_urls == {
        "https://pubmed.ncbi.nlm.nih.gov/29863451/",
        "https://pubmed.ncbi.nlm.nih.gov/37284804/",
        "https://pubmed.ncbi.nlm.nih.gov/24799467/",
    }
    assert body["honesty_gate"]["seen_urls_count"] == 3
    assert body["honesty_gate"]["evidence_kept"] == 3
    assert body["honesty_gate"]["evidence_dropped"] == 1

    systems = [
        messages[0].get("content", "")
        for messages in script.calls
        if messages and messages[0].get("role") == "system"
    ]
    user_blobs = [
        "\n".join(
            message.get("content", "")
            for message in messages
            if message.get("role") == "user"
        )
        for messages in script.calls
    ]
    assert len(script.calls) == 8
    assert sum("REFLECT phase" in prompt for prompt in systems) == 3
    assert sum("TOURNAMENT phase" in prompt for prompt in systems) == 3
    assert sum("META_REVIEW phase" in prompt for prompt in systems) == 1
    assert sum("GENERATE phase" in prompt for prompt in systems) == 1
    assert "grade 3" in user_blobs[0]


def test_real_endpoint_refuses_incomplete_pairwise_judgments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    headers = _enable_verified_auth(monkeypatch, tmp_path)
    monkeypatch.setenv(
        "MEDGEMMA_MODAL_URL",
        "https://crispro-test--medgemma-1-5-4b-chat.modal.run",
    )
    script = _success_script(malformed_pair=True)
    _install_model_and_evidence(monkeypatch, script)

    response = _post(TestClient(create_app()), headers=headers)
    assert response.status_code == 503
    assert "llm_tournament_incomplete" in response.json()["detail"]
    assert len(script.calls) == 7, "META_REVIEW must not run after an incomplete tournament"


def test_real_endpoint_refuses_incomplete_reflection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    headers = _enable_verified_auth(monkeypatch, tmp_path)
    monkeypatch.setenv(
        "MEDGEMMA_MODAL_URL",
        "https://crispro-test--medgemma-1-5-4b-chat.modal.run",
    )
    script = _success_script(malformed_reflection=True)
    _install_model_and_evidence(monkeypatch, script)

    response = _post(TestClient(create_app()), headers=headers)
    assert response.status_code == 503
    assert "llm_reflection_incomplete" in response.json()["detail"]
    assert len(script.calls) == 4, "TOURNAMENT must not run after incomplete REFLECT"


def test_real_endpoint_refuses_empty_evidence_without_running_elo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    headers = _enable_verified_auth(monkeypatch, tmp_path)
    monkeypatch.setenv(
        "MEDGEMMA_MODAL_URL",
        "https://crispro-test--medgemma-1-5-4b-chat.modal.run",
    )
    script = _success_script()

    def fake_chat(
        _client: GemmaClient,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> LlmResponse:
        del max_tokens, temperature
        return script.next(messages)

    def fake_retrieve(
        _stage: str,
        hypothesis: Any,
        seen_urls: set[str],
        *,
        context: dict[str, Any] | None = None,
        errors: list[str] | None = None,
        max_hits: int = 3,
    ) -> list[dict[str, Any]]:
        del hypothesis, seen_urls, context, errors, max_hits
        return []

    monkeypatch.setattr(GemmaClient, "chat", fake_chat)
    monkeypatch.setattr(
        "oncology_arbiter.agents.supervisor._retrieve_evidence",
        fake_retrieve,
    )
    response = _post(TestClient(create_app()), headers=headers)
    assert response.status_code == 503, response.text
    assert "llm_evidence_empty" in response.json()["detail"]
    assert len(script.calls) == 1, "REFLECT/TOURNAMENT must not run without fetched URLs"


def test_biopsy_evidence_plan_searches_literature_not_web_fetch() -> None:
    tools = plan_stage("biopsy").phase_tools[AgentPhase.EVIDENCE]
    assert "oncology_arbiter.tools.pubmed_search" in tools
    assert "oncology_arbiter.tools.europe_pmc_search" in tools
    assert "oncology_arbiter.tools.web_fetch" not in tools


def test_case_full_elo_ranked_maps_medgemma_tournament_not_offline() -> None:
    assert TumorBoardAssembler.elo_ranked_from_medgemma(None) == []
    ranked = TumorBoardAssembler.elo_ranked_from_medgemma({
        "llm_provider": "medgemma_modal",
        "stage": "biopsy",
        "hypotheses": [{
            "hypothesis_id": "clinical-h1",
            "claim": "ER-positive disease warrants endocrine therapy.",
            "initial_confidence": 0.93,
            "elo": 1532.0,
            "source_scope": "molecular",
            "evidence_urls": ["https://pubmed.ncbi.nlm.nih.gov/29863451/"],
            "tournament_wins": 2,
            "tournament_losses": 0,
            "evidence_alignment": "supports",
        }],
    })
    assert ranked[0]["hyp_id"] == "clinical-h1"
    assert ranked[0]["statement"].startswith("ER-positive")
    assert ranked[0]["rating"] == 1532.0
    assert ranked[0]["ranking_kind"] == "medgemma_pairwise_elo"
    assert ranked[0]["provider"] == "medgemma_modal"
    assert ranked[0]["evidence"][0]["url"].endswith("29863451/")


def test_real_endpoint_refuses_missing_medgemma_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    headers = _enable_verified_auth(monkeypatch, tmp_path)
    monkeypatch.delenv("MEDGEMMA_MODAL_URL", raising=False)
    response = _post(TestClient(create_app()), headers=headers)
    assert response.status_code == 503
    assert "MEDGEMMA_MODAL_URL is required" in response.json()["detail"]


def test_real_endpoint_requires_api_key_when_auth_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _enable_verified_auth(monkeypatch, tmp_path)
    response = _post(TestClient(create_app()))
    assert response.status_code == 401


def test_real_endpoint_rejects_anonymous_when_auth_mode_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_MODE", "off")
    monkeypatch.setenv(
        "MEDGEMMA_MODAL_URL",
        "https://crispro-test--medgemma-1-5-4b-chat.modal.run",
    )
    response = _post(TestClient(create_app()))
    assert response.status_code == 401


def test_real_endpoint_rejects_unpinned_medgemma_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    headers = _enable_verified_auth(monkeypatch, tmp_path)
    monkeypatch.setenv("MEDGEMMA_MODAL_URL", "https://evil.example/medgemma")
    script = _success_script()
    _install_model_and_evidence(monkeypatch, script)
    response = _post(TestClient(create_app()), headers=headers)
    assert response.status_code == 503
    assert "identity-pinned" in response.json()["detail"]
    assert script.calls == []


def test_real_endpoint_refuses_winner_only_pairwise_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    headers = _enable_verified_auth(monkeypatch, tmp_path)
    monkeypatch.setenv(
        "MEDGEMMA_MODAL_URL",
        "https://crispro-test--medgemma-1-5-4b-chat.modal.run",
    )
    script = _success_script(winner_only_pair=True)
    _install_model_and_evidence(monkeypatch, script)
    response = _post(TestClient(create_app()), headers=headers)
    assert response.status_code == 503
    assert "llm_tournament_incomplete" in response.json()["detail"]
    assert len(script.calls) == 7


def test_real_endpoint_refuses_unreflected_hypothesis_in_tournament(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    headers = _enable_verified_auth(monkeypatch, tmp_path)
    monkeypatch.setenv(
        "MEDGEMMA_MODAL_URL",
        "https://crispro-test--medgemma-1-5-4b-chat.modal.run",
    )
    script = _success_script()
    evidence_calls = 0

    def fake_chat(
        _client: GemmaClient,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> LlmResponse:
        del max_tokens, temperature
        return script.next(messages)

    def fake_retrieve(
        _stage: str,
        hypothesis: Any,
        seen_urls: set[str],
        *,
        context: dict[str, Any] | None = None,
        errors: list[str] | None = None,
        max_hits: int = 3,
    ) -> list[dict[str, Any]]:
        del context, errors, max_hits
        nonlocal evidence_calls
        evidence_calls += 1
        if evidence_calls > 2:
            return []
        url = f"https://pubmed.ncbi.nlm.nih.gov/{29863451 + evidence_calls}/"
        hypothesis.evidence_urls.append(url)
        seen_urls.add(url)
        return [{
            "url": url,
            "title": "Retrieved evidence",
            "snippet": "Fetched literature.",
            "source": "pubmed_search",
            "hypothesis_id": hypothesis.hypothesis_id,
        }]

    monkeypatch.setattr(GemmaClient, "chat", fake_chat)
    monkeypatch.setattr(
        "oncology_arbiter.agents.supervisor._retrieve_evidence",
        fake_retrieve,
    )
    response = _post(TestClient(create_app()), headers=headers)
    assert response.status_code == 503
    assert "llm_reflection_incomplete" in response.json()["detail"]
    assert len(script.calls) == 3, "TOURNAMENT must not include an unreflected hypothesis"
