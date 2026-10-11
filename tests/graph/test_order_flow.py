# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import pytest
from langchain_core.messages import ToolMessage

from app.graph import nodes
from app.tools.engine import ToolRun


@pytest.mark.asyncio
async def test_order_intent_skips_logistics_mcp(monkeypatch):
    async def must_not_load_specs():
        raise AssertionError("订单意图不应调用物流 MCP")

    monkeypatch.setattr(nodes.registry, "get_all_specs", must_not_load_specs)
    out = await nodes.fetch_logistics({"intent": "订单", "order_data": {"order_id": "1001"}})
    assert out["logistics_data"] == {}
    assert out["trace"]["fetch_logistics"]["skipped"] is True


@pytest.mark.asyncio
async def test_logistics_intent_calls_mcp_with_tracking_no(monkeypatch):
    class Spec:
        name = "query_logistics"

    async def specs():
        return [Spec()]

    seen = {}

    async def execute(tc, cid, available, **kwargs):
        seen.update(tc=tc, cid=cid)
        return ToolRun(
            tool_call_id=tc["id"], name=tc["name"], ok=True, status="成功",
            tool_message=ToolMessage(
                content='{"tracking_no":"SF1","status":"运输中","current_city":"杭州",'
                        '"trace":["杭州分拨中心 已发出"]}',
                tool_call_id=tc["id"], name=tc["name"], status="success"),
        )

    monkeypatch.setattr(nodes.registry, "get_all_specs", specs)
    monkeypatch.setattr(nodes.engine, "execute_tool_call", execute)
    out = await nodes.fetch_logistics({
        "intent": "物流", "conversation_id": 9, "user_id": "u1",
        "order_data": {"order_id": "1001", "tracking_no": "SF1"},
    })
    assert seen["tc"]["args"] == {"tracking_no": "SF1"}
    assert out["logistics_data"]["status"] == "运输中"
    assert out["trace"]["fetch_logistics"]["status"] == "成功"


@pytest.mark.asyncio
async def test_order_reply_merges_order_and_logistics_without_llm():
    out = await nodes.order_reply({
        "intent": "物流",
        "order_data": {
            "order_id": "1001", "product": "智能猫砂盆", "status": "已发货",
            "amount": 599, "created_at": "2026-09-30 10:00", "tracking_no": "SF1",
        },
        "logistics_data": {
            "tracking_no": "SF1", "status": "运输中", "current_city": "杭州",
            "trace": ["杭州分拨中心 已发出"],
        },
    })
    assert "订单 1001" in out["answer"]
    assert "已发货" in out["answer"] and "运输中" in out["answer"]
    assert "杭州分拨中心" in out["answer"]
    assert out["trace"]["route"] == "order_flow"
