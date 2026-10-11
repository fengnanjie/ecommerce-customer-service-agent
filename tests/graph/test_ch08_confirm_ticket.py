# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""投诉/人工走确定性升级出口，不进入 Agent，也不让模型直接执行写操作。"""
import pytest
from langchain_core.messages import HumanMessage

from app.graph import nodes


@pytest.mark.asyncio
@pytest.mark.parametrize("intent", ["投诉", "人工"])
async def test_escalation_route_returns_human_actions_without_model(monkeypatch, intent):
    def model_must_not_run(**kwargs):
        raise AssertionError("投诉/人工固定出口不应调用主 Agent")

    monkeypatch.setattr(nodes, "get_chat_model", model_must_not_run)
    out = await nodes.complaint_reply({
        "intent": intent,
        "messages": [HumanMessage("我要投诉并找人工")],
    })
    assert out["trace"]["route"] == "escalate"
    assert [a["type"] for a in out["suggested_actions"]] == [
        "transfer_human", "create_ticket"]
    assert out["suggested_actions"][1]["draft"]["ticket_type"] \
        == ("投诉" if intent == "投诉" else "咨询")


def test_refund_form_submission_is_pending_manual_review(client, monkeypatch):
    from app.api import actions

    seen = {}

    async def fake_create(conversation_id, description, ticket_type):
        seen.update(conversation_id=conversation_id, description=description,
                    ticket_type=ticket_type)
        return "R20260930001"

    monkeypatch.setattr(actions.repository, "create_ticket", fake_create)
    response = client.post("/api/actions/create-refund", json={
        "conversation_id": 7,
        "order_id": "1001",
        "reason": "质量问题",
    })
    assert response.status_code == 200
    assert response.json() == {
        "ticket_no": "R20260930001",
        "status": "退款申请已提交，等待人工审核",
    }
    assert seen["ticket_type"] == "退款"
    assert "订单号=1001" in seen["description"]
