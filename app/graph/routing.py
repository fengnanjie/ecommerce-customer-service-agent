# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
from langchain_core.messages import AIMessage

from app.config import settings

# 八类意图 → 五条业务路径。意图标签用于识别，route 才是实际执行策略；
# 返回值 = build.py 条件边映射键,单一来源不漂移)
INTENT_TO_ROUTE: dict[str, str] = {
    "投诉": "escalate",
    "人工": "escalate",
    "其他": "fallback_script",
    "咨询": "knowledge",
    "退款退货": "aftersales",
    "售后": "aftersales",
    "物流": "order_flow",
    "订单": "order_flow",
}


def route_by_intent(state) -> str:
    """按意图分流；未知输入不交给 Agent 猜，保守进入固定兜底。"""
    return INTENT_TO_ROUTE.get(state.get("intent", ""), "fallback_script")


def route_after_order(state) -> str:
    """订单事实取回后的确定性分支：查询类直接组装，售后类继续取政策证据。"""
    return "order" if state.get("route") == "order_flow" else "aftersales"


def confidence_gate(state) -> str:
    """知识路生成前证据闸:强放行、弱兜底。"""
    return "strong" if state.get("evidence_strong") else "weak"


def should_continue(state) -> str:
    """ReAct 停止条件:无 tool_calls 就收敛,步数封顶则强停,否则继续。

    只用步数封顶,不在这里卡 token。防打转靠的是步数:一步一次模型调用,数得清、好解释,
    各家 Agent SDK 给的也是 max_turns / max_iterations 这类步数上限。
    token 花销是另一件事——它管的是烧钱和撑爆上下文,不是死循环,阈值要按真实用量分布标定,
    归第 9 章的成本控制一起讲。混在这里卡,一次估偏就把工具链整条掐断,而且掐在模型
    已经决定调工具之后,用户只看到半句「我这就去查」。"""
    last = state["messages"][-1]
    has_tool_calls = isinstance(last, AIMessage) and bool(last.tool_calls)
    if not has_tool_calls:
        return "stop"
    if state.get("steps", 0) >= settings.max_agent_steps:
        return "stop"
    return "continue"
