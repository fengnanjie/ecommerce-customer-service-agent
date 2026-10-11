# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages


class SessionStore:
    """内存会话存储,session_id -> 消息列表。本章不做持久化。"""

    def __init__(self) -> None:
        self._sessions: dict[str, list[BaseMessage]] = {}

    def get(self, session_id: str) -> list[BaseMessage]:
        return self._sessions.get(session_id, [])

    def append(self, session_id: str, *messages: BaseMessage) -> None:
        self._sessions.setdefault(session_id, []).extend(messages)


def trim_history(messages: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    return trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
        start_on="human",
        allow_partial=False,
    )


def recent_complete_turns(messages: list[BaseMessage], turns: int,
                          exclude_last_human: bool = True) -> list[BaseMessage]:
    """取最近 N 个完整 user/assistant 轮次，不把“6 条消息”误称成“6 轮”。

    工具消息不会进入 coref/意图的历史视图；当前尚未得到 assistant 回复的最后一条
    human 默认排除，因为调用方会把本轮 query 单独传给模型。
    """
    dialog = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]
    if exclude_last_human and dialog and isinstance(dialog[-1], HumanMessage):
        dialog = dialog[:-1]
    grouped: list[list[BaseMessage]] = []
    current: list[BaseMessage] = []
    for msg in dialog:
        if isinstance(msg, HumanMessage):
            if current:
                grouped.append(current)
            current = [msg]
        elif current:
            current.append(msg)
    if current:
        grouped.append(current)
    complete = [g for g in grouped if any(isinstance(m, AIMessage) for m in g)]
    return [m for group in complete[-max(turns, 0):] for m in group]


def text_within_tokens(text: str, max_tokens: int) -> str:
    """按近似 token 预算截断文本，保留靠后的新内容。仅用于辅助上下文视图。"""
    if not text or max_tokens <= 0:
        return ""
    if count_tokens_approximately([HumanMessage(text)]) <= max_tokens:
        return text
    # 中文不能按空格安全切词，二分字符长度并保留尾部；最终仍用同一 token 估算器校验。
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = text[-mid:]
        if count_tokens_approximately([HumanMessage(candidate)]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[-lo:] if lo else ""


def text_prefix_within_tokens(text: str, max_tokens: int) -> str:
    """按近似 token 预算截断并保留前缀，适合必须保留安全说明/证据标题的文本。"""
    if not text or max_tokens <= 0:
        return ""
    if count_tokens_approximately([HumanMessage(text)]) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens_approximately([HumanMessage(text[:mid])]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] if lo else ""


# ---- ch07/ch11 锚点切窗 + 摘要注入(三层上下文的短期层)----

_DB_ID_PREFIX = "db-"


def _db_msg_id(m: BaseMessage) -> int | None:
    """从消息 id 解出 MySQL 消息 id(入口约定 HumanMessage.id=f"db-{msg_id}");非锚点返回 None。"""
    mid = getattr(m, "id", None)
    if isinstance(mid, str) and mid.startswith(_DB_ID_PREFIX):
        try:
            return int(mid[len(_DB_ID_PREFIX):])
        except ValueError:
            return None
    return None


def build_window(messages: list[BaseMessage], summary_upto_msg_id: int,
                 max_tokens: int) -> list[BaseMessage]:
    """滑动窗口:摘要边界锚点切窗(第一条 db-id > 边界的用户消息起)+ trim_messages token 兜底。
    锚点缺失(旧会话)/边界为 0 → 整段进 trim,退化纯 token 裁剪;裁到空宁可超预算回退原窗。
    只在调模型前现拼,不改 State。"""
    start = 0
    if summary_upto_msg_id:
        for i, m in enumerate(messages):
            if isinstance(m, HumanMessage):
                did = _db_msg_id(m)
                if did is not None and did > summary_upto_msg_id:
                    start = i
                    break
    window = messages[start:]
    trimmed = trim_history(window, max_tokens=max_tokens)
    return trimmed or window


def summary_line(summary: str | None) -> str:
    """coref/意图的历史文本前缀:单行摘要;无摘要空串。"""
    return f"(早前对话摘要:{summary})" if summary else ""


def summary_system(summary: str | None) -> SystemMessage | None:
    """main_agent 拼装用:摘要 system 消息(紧跟人设 system);无摘要 None。"""
    if not summary:
        return None
    return SystemMessage(f"## 早前对话摘要(更早轮次已压缩,其中事实可信)\n{summary}")
