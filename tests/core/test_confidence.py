# 来源：公众号@小林coding
"""Reranker Top1 三段式证据门控纯函数测试。"""
import pytest

from app.core.confidence import evaluate_evidence_gate, snapshot_from_hits


def _hit(score):
    return {"question": "退货运费谁出", "answer": "退货运费买家承担",
            "rerank_score": score, "section_path": "售后 / 退货",
            "id": 1, "content_type": "faq"}


def test_empty_hits_are_rejected():
    gate = evaluate_evidence_gate([], 0.3, 0.7)
    assert gate.top1_score == 0.0
    assert gate.decision == "reject"


@pytest.mark.parametrize(("score", "decision"), [
    (0.29, "reject"),
    (0.30, "review"),
    (0.69, "review"),
    (0.70, "accept"),
])
def test_three_way_boundaries(score, decision):
    assert evaluate_evidence_gate([_hit(score)], 0.3, 0.7).decision == decision


def test_trace_contains_only_explainable_gate_values():
    gate = evaluate_evidence_gate([_hit(0.52)], 0.3, 0.7)
    assert gate.trace == {"top1_score": 0.52, "decision": "review"}


@pytest.mark.parametrize(("low", "high"), [(-0.1, 0.7), (0.8, 0.7), (0.3, 1.1)])
def test_invalid_thresholds_fail_fast(low, high):
    with pytest.raises(ValueError):
        evaluate_evidence_gate([_hit(0.5)], low, high)


def test_snapshot_from_hits_top3_shape():
    hits = [_hit(0.9), _hit(0.8), _hit(0.7), _hit(0.6)]
    snap = snapshot_from_hits(hits)
    assert len(snap) == 3
    assert snap[0]["rerank_score"] == 0.9


def test_snapshot_empty_hits():
    assert snapshot_from_hits([]) == []
