# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""三段式闸:低分拒答、高分直通、临界分自检；
快照只要走了检索就写 state;fallback 落池带对 source 与快照。"""
import pytest

from app.graph import nodes


def _mk_hits(*scores):
    return [{"id": i, "question": f"q{i}", "answer": f"a{i}", "rerank_score": s,
             "section_path": "s", "content_type": "faq"} for i, s in enumerate(scores)]


@pytest.fixture
def stub_retrieval(monkeypatch):
    async def fake_understand(q):
        return {"standard": q, "expanded": []}
    monkeypatch.setattr(nodes.query_understanding, "understand", fake_understand)

    def set_hits(hits):
        async def fake_search(*a, **k):
            return hits
        monkeypatch.setattr(nodes.retrieval, "search_knowledge", fake_search)
    return set_hits


@pytest.fixture
def stub_selfcheck(monkeypatch):
    def set_result(useful, reason=""):
        async def fake_check(q, ev):
            return {"useful": useful, "reason": reason}
        monkeypatch.setattr(nodes.selfcheck, "check_sufficient", fake_check)
    return set_result


def _state(text="猫窝能水洗吗"):
    from langchain_core.messages import HumanMessage
    return {"messages": [HumanMessage(text)], "conversation_id": 1}


async def test_low_confidence_blocks_with_source_and_snapshot(stub_retrieval, stub_selfcheck, monkeypatch):
    monkeypatch.setattr("app.config.settings.rerank_reject_threshold", 0.3)
    monkeypatch.setattr("app.config.settings.rerank_accept_threshold", 0.7)
    stub_retrieval(_mk_hits(0.15, 0.14))          # 弱证据 → 置信度闸拦下
    stub_selfcheck(True)
    out = await nodes.retrieve_knowledge(_state())
    assert out["evidence_strong"] is False
    assert out["fallback_source"] == "retrieval_low_conf"
    assert len(out["retrieved_snapshot"]) == 2    # 走了检索就有快照
    assert out["evidence_confidence"] == 0.15
    assert out["trace"]["evidence_gate"]["decision"] == "reject"


async def test_selfcheck_fail_labels_self_check(stub_retrieval, stub_selfcheck, monkeypatch):
    monkeypatch.setattr("app.config.settings.rerank_reject_threshold", 0.3)
    monkeypatch.setattr("app.config.settings.rerank_accept_threshold", 0.7)
    stub_retrieval(_mk_hits(0.5, 0.3, 0.3))
    stub_selfcheck(False, "缺关键信息")
    out = await nodes.retrieve_knowledge(_state())
    assert out["evidence_strong"] is False
    assert out["fallback_source"] == "self_check"
    assert out["retrieved_snapshot"]              # 自评不过同样有快照


async def test_strong_evidence_keeps_snapshot_for_feedback(stub_retrieval, stub_selfcheck, monkeypatch):
    monkeypatch.setattr("app.config.settings.rerank_reject_threshold", 0.3)
    monkeypatch.setattr("app.config.settings.rerank_accept_threshold", 0.7)
    stub_retrieval(_mk_hits(0.9, 0.4, 0.3))
    async def must_not_run(*a, **k):
        raise AssertionError("高分证据不应调用 LLM 自检")
    monkeypatch.setattr(nodes.selfcheck, "check_sufficient", must_not_run)
    out = await nodes.retrieve_knowledge(_state())
    assert out["evidence_strong"] is True
    assert len(out["retrieved_snapshot"]) == 3    # 闸都过也留快照,供事后 👎 回捞
    assert out["citations"]
    assert out["trace"]["evidence_gate"]["decision"] == "accept"


async def test_fallback_reply_persists_source_and_snapshot(monkeypatch):
    recorded = {}

    async def fake_insert(cid, q, source, reason, retrieved_chunks=None):
        recorded.update(cid=cid, q=q, source=source, reason=reason, chunks=retrieved_chunks)
        return 1
    monkeypatch.setattr(nodes.repository, "insert_low_confidence", fake_insert)
    from langchain_core.messages import HumanMessage
    state = {"messages": [HumanMessage("猫窝能水洗吗")], "conversation_id": 7,
             "fallback_source": "self_check", "evidence_confidence": 0.41,
             "retrieved_snapshot": [{"question": "x", "answer": "y", "rerank_score": 0.4, "section_path": "s"}],
             "trace": {}}
    out = await nodes.fallback_reply(state)
    assert recorded["source"] == "self_check"
    assert recorded["chunks"] and recorded["chunks"][0]["question"] == "x"
    assert "0.41" in recorded["reason"]
    assert out["suggested_actions"] == [{"type": "transfer_human"}]


async def test_fallback_reply_zero_hit_stores_empty_list(monkeypatch):
    """走了检索但零命中:存 [](审核页显示「已检索,零命中」,真缺知识的强信号),不折叠成 NULL。"""
    recorded = {}

    async def fake_insert(cid, q, source, reason, retrieved_chunks=None):
        recorded["chunks"] = retrieved_chunks
        return 1
    monkeypatch.setattr(nodes.repository, "insert_low_confidence", fake_insert)
    from langchain_core.messages import HumanMessage
    await nodes.fallback_reply({"messages": [HumanMessage("冷门问题")], "conversation_id": 1,
                                "fallback_source": "retrieval_low_conf",
                                "retrieved_snapshot": [], "trace": {}})
    assert recorded["chunks"] == []               # 三态语义:[] ≠ NULL


async def test_fallback_reply_defaults_source_when_missing(monkeypatch):
    """非知识路兜底(如无 fallback_source 的旧路径)不许炸,回落 retrieval_low_conf。"""
    recorded = {}

    async def fake_insert(cid, q, source, reason, retrieved_chunks=None):
        recorded["source"] = source
        recorded["chunks"] = retrieved_chunks
        return 1
    monkeypatch.setattr(nodes.repository, "insert_low_confidence", fake_insert)
    from langchain_core.messages import HumanMessage
    out = await nodes.fallback_reply({"messages": [HumanMessage("嗯")], "conversation_id": 1, "trace": {}})
    assert recorded["source"] == "retrieval_low_conf"
    assert recorded["chunks"] is None             # 没快照就存 NULL,不存 []
    assert out["trace"] == {"route": "fallback"}
