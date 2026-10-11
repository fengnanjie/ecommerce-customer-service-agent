# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from pydantic import Field

from app.tools import registry
from app.tools.builtin.orders import NOT_OWNED
from app.tools.business import owns_order


@tool
async def prepare_refund_application(
    order_id: Annotated[str, Field(description="要退款的订单号")],
    user_id: Annotated[str, InjectedToolArg],
    reason: Annotated[str | None, Field(description="退款原因(可选,最终以前端固定类目下拉为准)")] = None,
) -> dict:
    """判定这一单符合条件后生成退款申请草稿。该工具不写库、不执行退款；
    用户确认表单后只创建待人工审核工单。
    发起人身份由系统注入,你不要传 user_id。"""
    # 即使只生成草稿也必须校验订单归属，避免暴露他人订单的售后入口。
    if not owns_order(user_id, order_id):
        return dict(NOT_OWNED)
    return {"status": "申请草稿待用户确认", "order_id": order_id}


registry.register(registry.spec_from_langchain_tool(
    prepare_refund_application, source="builtin", inject_user_id=True))
