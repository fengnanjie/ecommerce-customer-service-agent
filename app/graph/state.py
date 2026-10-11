# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def merge_dict(a: dict | None, b: dict | None) -> dict:
    """trace 累加 reducer:后写覆盖同键,其余合并;b 为 None 是重置哨兵。
    入口(_graph_input)每轮传 trace=None 清零本轮 trace——merge 通道塞 {} 清不掉
    (merge(旧,{})=旧),故用 None 触发重置,杜绝上一轮条件键(如 retrieve_policy)跨轮残留。
    图内各节点写 trace 恒为真 dict(从不传 None),故 None 专属入口重置,不误伤累加。"""
    if b is None:
        return {}
    return {**(a or {}), **b}


class ConversationState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # 跨轮历史,checkpointer 续接
    summary: str            # ch07 早期轮次滚动摘要(入口每轮从 conversations 加载)
    summary_upto_msg_id: int  # 摘要覆盖到的 MySQL 消息 id,滑窗从其后接原文
    long_term_memories: list  # 用户级跨会话召回结果(最多 Top3,本轮入口重置)
    user_id: str
    conversation_id: int
    intent: str            # 八类之一:咨询/订单/物流/售后/退款退货/投诉/人工/其他
    resolved_query: str    # 指代消解+改写后的完整问句(下游检索/判意图都用它)
    intent_confidence: float  # 意图 JSON 的 confidence(0-1)
    order_id: str          # order_flow/aftersales:抽到或点选回填的订单号
    order_data: dict       # order_flow/aftersales:query_order 查到的订单数据
    logistics_data: dict   # order_flow:物流意图的 MCP 实时轨迹
    route: str             # knowledge | order_flow | aftersales | escalate | fallback_script
    evidence: str          # 知识路编号证据文本
    citations: list        # 引用 chunk(前端可点)
    evidence_strong: bool  # 生成前证据闸信号
    evidence_confidence: float  # 证据门控使用的 Reranker Top1 相关分(保留字段名兼容既有状态/页面)
    fallback_source: str        # ch09 兜底落池 source:retrieval_low_conf | self_check
    retrieved_snapshot: list    # ch09 召回快照(Top3 原文+得分):走了检索就有,落池/👎回捞共用
    answer: str            # 确定性节点产出的答复(agent 答复走流式,不落此字段)
    steps: int             # ReAct 步数(停止条件)
    tokens_used: int       # token 预算累加
    suggested_actions: list  # [{"type": "transfer_human"} | {"type": "create_ticket", "draft": {...}}]
    trace: Annotated[dict, merge_dict]  # 留痕
