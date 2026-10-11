# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import asyncio
from typing import Annotated, Literal

from langchain_core.tools import InjectedToolArg, tool

from app.config import settings
from app.db import repository
from app.tools import registry


@tool
async def create_ticket(
    description: str,
    ticket_type: Literal["售后", "投诉", "咨询"],
    conversation_id: Annotated[int, InjectedToolArg],
) -> dict:
    """创建人工工单。仅当用户明确要求建工单/要求人工跟进时才调用。
    业务必填项只有 description(用户已经说明的实际问题)和 ticket_type(售后/投诉/咨询);
    订单号、购买时间、商品型号、联系方式和期望方案均非必填,不得为此追问。用户尚未说明
    实际问题时才追问 description;一旦问题已说清就立即调用。工单关联的会话号由系统注入,
    你不要传。严禁编造或用占位文本。"""
    if settings.demo_ticket_delay_seconds > 0:      # 验收 6:环境变量注入写操作超时演示
        await asyncio.sleep(settings.demo_ticket_delay_seconds)
    ticket_no = await repository.create_ticket(conversation_id, description, ticket_type)
    return {"ticket_no": ticket_no, "status": "已转人工"}


registry.register(registry.spec_from_langchain_tool(
    create_ticket, source="builtin", inject_conversation=True))
