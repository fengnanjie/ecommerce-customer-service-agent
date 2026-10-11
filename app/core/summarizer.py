# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""中期增量摘要:软阈值后台压缩,硬阈值在下一次模型调用前同步兜底。

触发条件统一按 token 而不是消息条数；压缩只切完整轮次，摘要与边界原子更新。
"""
import asyncio
import logging
import time

from pydantic import BaseModel, Field

from app.config import settings
from app.core import llm
from app.core.prompts import SUMMARY_PROMPT
from app.db import repository
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately

logger = logging.getLogger(__name__)

_running: dict[int, asyncio.Task] = {}   # cid -> 在跑任务(防抖:在跑不重复起)


class _Summary(BaseModel):
    summary: str = Field(description="早期对话滚动摘要,只含事实与诉求")


def _row_messages(messages) -> list:
    return [HumanMessage(m.content or "") if m.role == "user" else AIMessage(m.content or "")
            for m in messages if m.role in ("user", "assistant")]


def dialog_tokens(messages) -> int:
    return count_tokens_approximately(_row_messages(messages)) if messages else 0


def compute_boundary(messages, target_tokens: int, max_turns: int,
                     min_turns: int = 2) -> int | None:
    """求压缩边界，使边界后保留完整轮次且接近 target_tokens。

    最多保留 max_turns，同时至少保住最近 min_turns。若没有可滑出的旧段则返回 None。
    """
    starts = [i for i, m in enumerate(messages) if m.role == "user"]
    if len(starts) <= min_turns:
        return None
    keep_from = starts[-min_turns]
    kept_turns = min_turns
    # 从近到远尝试多收完整轮次，直到预算或轮数上限。
    for idx in reversed(starts[:-min_turns]):
        if kept_turns >= max_turns:
            break
        if dialog_tokens(messages[idx:]) > target_tokens:
            break
        keep_from = idx
        kept_turns += 1
    if keep_from <= 0:
        return None
    return messages[keep_from - 1].id


async def unsummarized_tokens(conversation_id: int) -> int:
    conv = await repository.get_conversation(conversation_id)
    if conv is None:
        return 0
    rows = await repository.list_dialog_messages(conversation_id)
    upto = conv.summary_upto_msg_id or 0
    return dialog_tokens([m for m in rows if m.id > upto])


async def summarize_dialog(old_summary: str, dialog: str) -> str:
    """LLM 滚动重写:旧摘要 + 新滑出段 → 新摘要。严格 JSON(structured output)。"""
    model = llm.structured(_Summary, model=settings.summary_model or None)
    r: _Summary = await (SUMMARY_PROMPT | model).ainvoke(
        {"old_summary": old_summary or "(无)", "dialog": dialog})
    return (r.summary or "").strip()


async def run_summary(conversation_id: int) -> None:
    """任务体:读消息 → 算边界 → 压缩 → 原子更新两字段。异常往外抛,由 done 回调统一 log。"""
    t0 = time.monotonic()
    conv = await repository.get_conversation(conversation_id)
    if conv is None:
        return
    msgs = await repository.list_dialog_messages(conversation_id)
    boundary = compute_boundary(
        msgs,
        target_tokens=settings.context_window_target_tokens,
        max_turns=settings.context_window_turns,
        min_turns=settings.context_window_min_turns,
    )
    old_upto = conv.summary_upto_msg_id or 0
    if boundary is None or boundary <= old_upto:
        logger.info("summary skip conv=%s boundary=%s upto=%s",
                    conversation_id, boundary, old_upto)
        return
    seg = [m for m in msgs if old_upto < m.id <= boundary]
    dialog = "\n".join(
        f"{'用户' if m.role == 'user' else '客服'}:{m.content}" for m in seg if m.content)
    logger.info("summary start conv=%s msgs=%s upto %s->%s",
                conversation_id, len(seg), old_upto, boundary)
    summary = await summarize_dialog(conv.summary or "", dialog)
    if not summary:
        raise ValueError("摘要为空,放弃更新")
    await repository.update_conversation_summary(conversation_id, summary, boundary)
    logger.info("summary done conv=%s upto=%s len=%s cost=%.0fms",
                conversation_id, boundary, len(summary), (time.monotonic() - t0) * 1000)


async def maybe_schedule_summary(conversation_id: int) -> None:
    """轮结束后调用:未摘要原文达软 token 阈值 → 后台压缩,不阻塞回复。
    本函数在用户回复路径上,任何异常只 log 不外抛(触发条件仍在,下一轮重来)。"""
    try:
        if conversation_id in _running:
            return
        tokens = await unsummarized_tokens(conversation_id)
        if tokens < settings.context_window_soft_tokens:
            return
        # 上面两个 await 期间可能有并发触发已入表(TOCTOU),入表前再查一次防双起
        if conversation_id in _running:
            return
        logger.info("summary soft-trigger conv=%s raw_tokens≈%s", conversation_id, tokens)
        task = asyncio.create_task(run_summary(conversation_id))
        _running[conversation_id] = task

        def _done(t: asyncio.Task) -> None:
            if _running.get(conversation_id) is t:   # 只清自己,防误清后来者
                _running.pop(conversation_id, None)
            if not t.cancelled() and t.exception() is not None:
                logger.error("summary failed conv=%s", conversation_id, exc_info=t.exception())

        task.add_done_callback(_done)
    except Exception:
        logger.exception("summary trigger check failed conv=%s", conversation_id)


async def ensure_hard_summary(conversation_id: int) -> None:
    """模型调用前硬兜底：原文达到 3000 token 时等待/执行压缩。

    正常轮次只会走 2400 token 的后台路径；仅当后台尚未完成且继续增长到硬阈值时，
    才用同步等待换取上下文不超预算。失败继续由调用方日志记录，不吞掉用户请求。
    """
    tokens = await unsummarized_tokens(conversation_id)
    if tokens < settings.context_window_max_tokens:
        return
    try:
        task = _running.get(conversation_id)
        if task is not None:
            await task
            return
        logger.info("summary hard-trigger conv=%s raw_tokens≈%s", conversation_id, tokens)
        await run_summary(conversation_id)
    except Exception:
        # 摘要上游故障不应让客服主链 500；main_agent 仍会用 3000-token 滑窗硬裁剪兜底。
        logger.exception("summary hard-trigger failed, fallback to token window conv=%s",
                         conversation_id)
