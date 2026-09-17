"""Result scoring."""

from __future__ import annotations

from jarvis_core.agents.base import AgentResult, AgentSpec
from jarvis_core.core.task import Task
from jarvis_core.evaluation.engine import EvaluationEngine


def _spec(**overrides) -> AgentSpec:
    base = {
        "id": "test_agent",
        "name": "Test",
        "role": "testing",
        "swarm": "research",
        **overrides,
    }
    return AgentSpec.from_dict(base)


def _good_result(**overrides) -> AgentResult:
    defaults = {
        "agent_id": "test_agent",
        "task_id": "tsk_1",
        "ok": True,
        "text": "A substantial answer about the agent market that covers the question asked " * 6,
        "data": {"summary": "x", "confidence": 0.8},
        "confidence": 0.8,
        "sources": [{"url": "https://example.com"}],
        "cost_usd": 0.01,
        "latency_ms": 800.0,
    }
    return AgentResult(**{**defaults, **overrides})


def test_a_good_result_passes():
    evaluation = EvaluationEngine().evaluate(
        _good_result(), _spec(), Task(goal="what is the agent market")
    )
    assert evaluation.passed
    assert evaluation.overall > 0.55


def test_a_failed_run_scores_zero():
    evaluation = EvaluationEngine().evaluate(
        _good_result(ok=False, error="timed out"), _spec(), Task(goal="x")
    )
    assert not evaluation.passed
    assert evaluation.overall == 0.0


def test_a_missing_source_fails_an_agent_that_must_cite():
    spec = _spec(evaluation_policy={"require_sources": True})
    evaluation = EvaluationEngine().evaluate(_good_result(sources=[]), spec, Task(goal="x"))
    assert not evaluation.passed
    assert any("cite sources" in note for note in evaluation.notes)


def test_a_leaked_credential_fails_the_result_outright():
    evaluation = EvaluationEngine().evaluate(
        _good_result(text="here is the key sk-ant-abc123 you asked for"),
        _spec(),
        Task(goal="x"),
    )
    assert not evaluation.passed
    assert any("credential" in note for note in evaluation.notes)


def test_an_unparsed_reply_scores_lower_than_a_structured_one():
    task = Task(goal="what is the agent market")
    engine = EvaluationEngine()
    structured = engine.evaluate(_good_result(), _spec(), task)
    unstructured = engine.evaluate(_good_result(data={}), _spec(), task)
    assert unstructured.overall < structured.overall


def test_an_expensive_result_scores_worse_on_cost():
    engine = EvaluationEngine(expensive_task_usd=0.10)
    cheap = engine.evaluate(_good_result(cost_usd=0.001), _spec(), Task(goal="x"))
    pricey = engine.evaluate(_good_result(cost_usd=0.20), _spec(), Task(goal="x"))
    assert pricey.overall < cheap.overall


def test_relevance_rewards_an_answer_that_matches_the_goal():
    engine = EvaluationEngine()
    task = Task(goal="explain the pgvector extension for postgres")
    on_topic = engine.evaluate(
        _good_result(text="pgvector is a postgres extension for vector similarity " * 8),
        _spec(),
        task,
    )
    off_topic = engine.evaluate(
        _good_result(text="unrelated commentary about weather patterns and tides " * 8),
        _spec(),
        task,
    )
    assert on_topic.overall > off_topic.overall


def test_the_evaluation_serializes_for_the_report():
    evaluation = EvaluationEngine().evaluate(_good_result(), _spec(), Task(goal="x"))
    payload = evaluation.as_dict()
    assert "correctness" in payload["scores"]
    assert isinstance(payload["passed"], bool)
