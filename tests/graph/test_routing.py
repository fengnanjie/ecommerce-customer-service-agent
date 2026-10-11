# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import pytest
from langchain_core.messages import AIMessage

from app.config import settings
from app.graph.routing import (
    confidence_gate, route_after_order, route_by_intent, should_continue, INTENT_TO_ROUTE,
)


@pytest.mark.parametrize("intent,expect", [
    ("投诉", "escalate"),
    ("人工", "escalate"),
    ("其他", "fallback_script"),
    ("咨询", "knowledge"),
    ("退款退货", "aftersales"),
    ("售后", "aftersales"),
    ("物流", "order_flow"),
    ("订单", "order_flow"),
])
def test_route_by_intent_five_outlets(intent, expect):
    assert route_by_intent({"intent": intent}) == expect


def test_route_by_intent_unknown_defaults_fixed_fallback():
    assert route_by_intent({"intent": "火星语"}) == "fallback_script"
    assert route_by_intent({}) == "fallback_script"


def test_intent_to_route_covers_eight_classes():
    assert set(INTENT_TO_ROUTE) == {
        "投诉", "其他", "咨询", "退款退货", "售后", "人工", "物流", "订单"}


def test_route_after_order_uses_saved_route():
    assert route_after_order({"route": "order_flow"}) == "order"
    assert route_after_order({"route": "aftersales"}) == "aftersales"


def test_confidence_gate():
    assert confidence_gate({"evidence_strong": True}) == "strong"
    assert confidence_gate({"evidence_strong": False}) == "weak"
    assert confidence_gate({}) == "weak"


def test_should_continue_stops_when_no_tool_calls():
    state = {"messages": [AIMessage("答完了")], "steps": 1, "tokens_used": 0}
    assert should_continue(state) == "stop"


def test_should_continue_continues_on_tool_calls():
    ai = AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": "1"}])
    state = {"messages": [ai], "steps": 1, "tokens_used": 0}
    assert should_continue(state) == "continue"


def test_should_continue_stops_on_step_cap():
    ai = AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": "1"}])
    state = {"messages": [ai], "steps": settings.max_agent_steps, "tokens_used": 0}
    assert should_continue(state) == "stop"


def test_token_花销不再当停止条件():
    # 环里只按步数封顶。token 记账继续走(进 trace、给 ch09 统计),但不参与判断:
    # 混在这里卡,阈值估偏一次就把整条工具链掐断,而且掐在模型已经决定调工具之后
    ai = AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": "1"}])
    state = {"messages": [ai], "steps": 1, "tokens_used": 10_000_000}
    assert should_continue(state) == "continue"
