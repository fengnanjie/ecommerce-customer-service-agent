# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph import nodes


def test_agent_messages_injects_evidence_on_knowledge():
    msgs = nodes._agent_messages(
        {"route": "knowledge", "evidence": "[1] 退货: 7天",
         "messages": [HumanMessage("能退吗")]})
    # 证据进的是用户侧消息,不进 system——system 要保持逐字不变才能命中前缀缓存
    assert isinstance(msgs[0], SystemMessage)
    assert "[1] 退货: 7天" not in msgs[0].content
    ctx = [m for m in msgs if isinstance(m, HumanMessage) and "[1] 退货: 7天" in m.content]
    assert len(ctx) == 1
    assert "query_faq" in ctx[0].content            # 指示别再检索,跟着证据一起搬过来了
    assert msgs[-2].content == "能退吗" and msgs[-1] is ctx[0]   # 紧跟用户那句之后


def test_agent_messages_no_evidence_without_rag():
    msgs = nodes._agent_messages({"route": "order_flow", "messages": [HumanMessage("订单1001")]})
    assert isinstance(msgs[0], SystemMessage)
    assert all("已检索到的知识证据" not in (m.content or "") for m in msgs)  # 整个列表都不该有


@pytest.fixture()
def builtin_only_specs(monkeypatch):
    """ch08:节点每轮现拉内置+MCP;单测锚定为仅内置,免网络依赖。"""
    from app.tools import registry

    async def only_builtin():
        return registry.builtin_specs()

    monkeypatch.setattr(nodes.registry, "get_all_specs", only_builtin)


@pytest.mark.asyncio
async def test_main_agent_accumulates_steps_and_tokens(monkeypatch, builtin_only_specs):
    ai = AIMessage("好的", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

    class FakeModel:
        def bind_tools(self, tools):
            return self
        def bind(self, **kwargs):
            return self
        async def ainvoke(self, msgs, config=None):
            return ai

    monkeypatch.setattr(nodes, "get_chat_model", lambda **k: FakeModel())
    out = await nodes.main_agent({"messages": [HumanMessage("hi")], "steps": 1, "tokens_used": 100})
    assert out["steps"] == 2
    assert out["tokens_used"] == 115
    assert out["messages"][0] is ai


@pytest.mark.asyncio
async def test_agent_tools_executes_normal_tool(monkeypatch, builtin_only_specs):
    from langchain_core.messages import ToolMessage

    from app.tools.engine import ToolRun

    async def fake_exec(tc, cid, specs, **kw):
        return ToolRun(tool_call_id=tc["id"], name=tc["name"], ok=True, status="成功",
                       tool_message=ToolMessage(content='{"status":"已发货"}',
                                                tool_call_id=tc["id"], name=tc["name"]))

    monkeypatch.setattr(nodes.engine, "execute_tool_call", fake_exec)
    ai = AIMessage("", tool_calls=[{"name": "query_product", "args": {"product_name": "猫粮"}, "id": "t1"}])
    out = await nodes.agent_tools({"route": "knowledge", "messages": [ai], "conversation_id": 5})
    assert out["messages"][0].name == "query_product"
    assert not out.get("suggested_actions")


@pytest.mark.asyncio
async def test_route_tool_allowlist(monkeypatch, builtin_only_specs):
    consultation = {s.name for s in await nodes._route_specs({"route": "knowledge"})}
    aftersales = {s.name for s in await nodes._route_specs({"route": "aftersales"})}
    assert consultation == {"query_product"}
    assert "prepare_refund_application" in aftersales
    assert "create_ticket" not in consultation | aftersales


@pytest.mark.asyncio
async def test_agent_tools_intercepts_refund_draft(monkeypatch, builtin_only_specs):
    async def fake_exec(tc, cid, specs, **kw):
        raise AssertionError("prepare_refund_application 不应进引擎(拦成前端退款表单)")

    monkeypatch.setattr(nodes.engine, "execute_tool_call", fake_exec)
    uid = "u-agent"
    mine = nodes.business.list_user_orders(uid)[0]["order_id"]   # 得是他自己的单才拦成表单
    ai = AIMessage("", tool_calls=[{"name": "prepare_refund_application",
                    "args": {"order_id": mine}, "id": "t8"}])
    out = await nodes.agent_tools({"route": "aftersales", "messages": [ai],
                                   "conversation_id": 5, "user_id": uid})
    assert out["suggested_actions"][0]["type"] == "refund_form"
    assert out["messages"][0].tool_call_id == "t8"


async def test_退款不给别人的单开表单入口(monkeypatch, builtin_only_specs):
    # 草稿工具在节点里就被拦成前端表单、不进执行引擎,所以工具内那道校验够不着它。
    # 这里守住:不是他的单,连「提交退款工单」这个入口都不该出现
    async def fake_exec(tc, cid, specs, **kw):
        raise AssertionError("prepare_refund_application 不应进引擎")

    monkeypatch.setattr(nodes.engine, "execute_tool_call", fake_exec)
    uid, other = "u-agent", "u-someone-else"
    # 得挑对方的**私有**单:演示单(1001/2002)每个账号都有,拿它当「别人的单」不成立
    his = [o["order_id"] for o in nodes.business.list_user_orders(other)
           if o["order_id"] not in nodes.business.DEMO_ORDER_IDS][0]
    ai = AIMessage("", tool_calls=[{"name": "prepare_refund_application",
                    "args": {"order_id": his}, "id": "t9"}])
    out = await nodes.agent_tools({"route": "aftersales", "messages": [ai],
                                   "conversation_id": 5, "user_id": uid})
    types = [a["type"] for a in out["suggested_actions"]]
    assert "refund_form" not in types          # 没有退款入口
    assert "select_order" in types             # 但给了出路:列出他自己的单
    assert out["messages"][0].status == "error"
