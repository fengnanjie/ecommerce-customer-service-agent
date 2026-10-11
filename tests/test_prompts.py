# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
from langchain_core.messages import HumanMessage

from app.core.prompts import CUSTOMER_SERVICE_PROMPT, EXTRACT_PROMPT, AGENT_SYSTEM, AGENT_PROMPT


def test_customer_service_prompt_renders_with_history():
    msgs = CUSTOMER_SERVICE_PROMPT.format_messages(
        history=[HumanMessage("你们卖猫粮吗?")]
    )
    assert msgs[0].type == "system"
    assert msgs[-1].content == "你们卖猫粮吗?"
    # 行为约束必须写进 system prompt
    for keyword in ("客服", "不", "订单"):
        assert keyword in msgs[0].content


def test_extract_prompt_renders_text():
    msgs = EXTRACT_PROMPT.format_messages(text="订单 MH1 坏了要退款")
    assert msgs[0].type == "system"
    assert "订单 MH1 坏了要退款" in msgs[-1].content


def test_agent_system_covers_tool_principles():
    for kw in ["小喵", "本轮实际提供", "Workflow", "query_product", "不要编造"]:
        assert kw in AGENT_SYSTEM


def test_agent_system_denies_business_writes():
    """业务 Agent 只能查询或生成草稿，建单/转人工由确定性流程处理。"""
    for phrase in [
        "退款只能生成申请草稿",
        "建工单、转人工和真正写入由确定性流程与用户确认处理",
        "不得自行承诺已完成这些动作",
    ]:
        assert phrase in AGENT_SYSTEM


def test_agent_prompt_has_history_placeholder():
    assert any(getattr(m, "variable_name", None) == "history"
               for m in AGENT_PROMPT.messages)
