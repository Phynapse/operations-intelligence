"""Integration tests for the question endpoint and the full pipeline.

These exercise the HTTP surface end to end: request in, complete envelope out,
through the real repository, the real analyzers and the real evidence validator.
Nothing is mocked except the LLM provider, because the provider is the only part
of the system allowed to be unavailable.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.ai.pipeline import QuestionPipeline
from app.ai.providers.openai_compatible import ChatCompletion, LLMProvider, LLMProviderError
from app.main import create_app
from app.models.enums import AnalysisType

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


class _ReplyProvider(LLMProvider):
    model = "integration-fake"

    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def chat(self, messages, *, max_tokens=None, temperature=None) -> ChatCompletion:
        return ChatCompletion(content=self.reply, model=self.model)

    async def aclose(self) -> None:
        return None


class _BrokenProvider(LLMProvider):
    model = "integration-broken"

    async def chat(self, messages, *, max_tokens=None, temperature=None) -> ChatCompletion:
        raise LLMProviderError("upstream unavailable")

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------------------
# The response envelope
# ---------------------------------------------------------------------------


class TestResponseEnvelope:
    def test_returns_200(self, client):
        response = client.post(
            "/api/questions", json={"question": "What is the average cycle time in Q3?"}
        )
        assert response.status_code == 200

    def test_contains_every_required_section(self, client):
        response = client.post(
            "/api/questions",
            json={"question": "What is the average cycle time in Q3?"},
        )
        body = response.json()
        for field in (
            "question",
            "plan",
            "findings",
            "evidence",
            "answer",
            "recommendations",
            "caveats",
            "analysis_id",
        ):
            assert field in body, f"missing '{field}'"
            assert body[field] is not None

    def test_plan_is_populated(self, client):
        body = client.post(
            "/api/questions", json={"question": "What is the average cycle time in Q3?"}
        ).json()
        plan = body["plan"]
        assert plan["analysis_type"] == AnalysisType.CYCLE_TIME_ANALYSIS.value
        assert plan["metric"] == "average_cycle_time"
        assert plan["period"] == "2025-Q3"
        assert plan["reasoning_goal"]

    def test_evidence_items_are_provenance_bearing(self, client):
        body = client.post(
            "/api/questions", json={"question": "What is the average cycle time in Q3?"}
        ).json()
        assert body["evidence"]
        for item in body["evidence"]:
            assert item["source_dataset"]
            assert item["calculation_description"]
            assert item["unit"]

    def test_grounding_summary_is_reported(self, client):
        body = client.post(
            "/api/questions", json={"question": "What is the average cycle time in Q3?"}
        ).json()
        grounding = body["grounding"]
        assert grounding["grounded"] is True
        assert grounding["unsupported"] == 0
        assert grounding["numbers_checked"] >= 0

    def test_generated_by_is_one_of_two_values(self, client):
        body = client.post("/api/questions", json={"question": "Give me an overview"}).json()
        assert body["generated_by"] in {"llm", "deterministic_fallback"}

    def test_analysis_id_is_retrievable(self, client):
        body = client.post(
            "/api/questions", json={"question": "What is the average cycle time in Q3?"}
        ).json()
        fetched = client.get(f"/api/analysis/{body['analysis_id']}")
        assert fetched.status_code == 200
        assert fetched.json()["analysis_id"] == body["analysis_id"]

    def test_caveats_are_returned_even_when_empty(self, client):
        body = client.post("/api/questions", json={"question": "Give me an overview"}).json()
        assert isinstance(body["caveats"], list)


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


class TestRejections:
    @pytest.mark.parametrize(
        ("question", "status", "code"),
        [
            ("What will throughput be next quarter?", 422, "unsupported_analysis"),
            ("What was our revenue in Q3?", 422, "unsupported_analysis"),
            ("Run SELECT * FROM operations", 422, "unsupported_analysis"),
            ("Delete the rows for DEP-101", 422, "unsupported_analysis"),
        ],
    )
    def test_unsupported_questions_are_rejected_with_a_code(self, client, question, status, code):
        response = client.post("/api/questions", json={"question": question})
        assert response.status_code == status
        assert response.json()["error"]["code"] == code

    def test_rejection_message_explains_why(self, client):
        response = client.post(
            "/api/questions", json={"question": "What will throughput be next quarter?"}
        )
        message = response.json()["error"]["message"]
        assert "historical" in message
        assert "throughput" in message

    def test_too_short_question_is_a_validation_error(self, client):
        response = client.post("/api/questions", json={"question": "ab"})
        assert response.status_code == 422

    def test_overlong_question_is_a_validation_error(self, client):
        response = client.post("/api/questions", json={"question": "x" * 700})
        assert response.status_code == 422

    def test_unknown_field_is_rejected(self, client):
        response = client.post(
            "/api/questions",
            json={"question": "Give me an overview", "sql": "SELECT 1"},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# The contribution path over HTTP
# ---------------------------------------------------------------------------


class TestContributionEndpoint:
    def test_q2_q3_contribution_question(self, client):
        response = client.post(
            "/api/questions",
            json={
                "question": (
                    "What drove the change in average cycle time between Q2 and Q3 by department?"
                )
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["plan"]["analysis_type"] == "contribution_analysis"
        assert body["plan"]["period"] == "2025-Q3"
        assert body["plan"]["comparison_period"] == "2025-Q2"
        assert len(body["evidence"]) >= 2

    def test_contribution_members_carry_both_windows(self, client):
        body = client.post(
            "/api/questions",
            json={
                "question": (
                    "Which department contributed most to the change in backlog between Q2 and Q3?"
                )
            },
        ).json()
        members = [item for item in body["evidence"] if item["dimension"]]
        assert members
        assert all(item["comparison_value"] is not None for item in members)


# ---------------------------------------------------------------------------
# Provider behaviour through the pipeline
# ---------------------------------------------------------------------------


class TestProviderBehaviour:
    def test_no_provider_yields_the_deterministic_answer(self):
        import asyncio

        body = asyncio.run(
            QuestionPipeline(provider=None).answer("What is the average cycle time in Q3?")
        )
        assert body.generated_by == "deterministic_fallback"
        assert body.grounding.grounded

    def test_broken_provider_still_returns_a_full_envelope(self):
        import asyncio

        body = asyncio.run(
            QuestionPipeline(provider=_BrokenProvider()).answer(
                "What is the average cycle time in Q3?"
            )
        )
        assert body.generated_by == "deterministic_fallback"
        assert body.evidence
        assert body.answer
        assert body.analysis_id

    def test_grounded_provider_reply_is_used(self):
        import asyncio

        provider = _ReplyProvider(
            json.dumps(
                {
                    "answer": "Average cycle time moved to 1.86 hours in 2025-Q3.",
                    "key_findings": [
                        {
                            "statement": "Queue time accounts for the movement.",
                            "severity": "medium",
                            "evidence_ids": ["E01"],
                        }
                    ],
                    "recommendations": [
                        {
                            "action": "Review queue capacity.",
                            "rationale": "Queue share is the largest contributor.",
                            "evidence_ids": ["E02"],
                            "priority": "high",
                        }
                    ],
                    "caveats": ["Weighted by completed operations."],
                }
            )
        )
        body = asyncio.run(
            QuestionPipeline(provider=provider).answer("What is the average cycle time in Q3?")
        )
        assert body.generated_by == "llm"
        assert body.model == "integration-fake"
        assert body.grounding.grounded

    def test_fabricating_provider_reply_is_discarded(self):
        import asyncio

        provider = _ReplyProvider(
            json.dumps(
                {
                    "answer": "SLA compliance reached 100% and backlog hit zero.",
                    "key_findings": [],
                    "recommendations": [],
                    "caveats": [],
                }
            )
        )
        body = asyncio.run(
            QuestionPipeline(provider=provider).answer("What is the average cycle time in Q3?")
        )
        assert body.generated_by == "deterministic_fallback"
        assert body.unsupported_claims_detected is True
        assert "100%" not in body.answer


# ---------------------------------------------------------------------------
# Determinism across requests
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_repeated_requests_produce_identical_numbers(self, client):
        payload = {"question": "Which process is the biggest bottleneck in Q3?"}
        first = client.post("/api/questions", json=payload).json()
        second = client.post("/api/questions", json=payload).json()
        assert first["answer"] == second["answer"]
        assert [item["value"] for item in first["evidence"]] == [
            item["value"] for item in second["evidence"]
        ]

    def test_every_grounded_response_passes_its_own_check(self, client):
        for question in (
            "Give me an operational overview for Q4 2025",
            "Which vendor has the highest delay rate in Q3?",
            "Are there unusual operational patterns in Q3?",
            "How has cycle time trended over 2025?",
        ):
            body = client.post("/api/questions", json={"question": question}).json()
            assert body["grounding"]["grounded"] is True, question
            assert body["grounding"]["unsupported"] == 0, question
