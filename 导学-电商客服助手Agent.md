# 电商客服助手Agent 项目导学（当前实现）

> 源码基线：2026-09-30 本地工作区。本文不包含主题分类器微调内容。

## 1. 前置知识（面试高频标注）

| 知识点 | 为何需要 | 在本项目中的位置 | 高频度 |
| --- | --- | --- | --- |
| LangGraph 状态图 | 理解确定性步骤、条件边和 Agent 循环如何共存 | `app/graph/build.py`、`routing.py` | 高 |
| Tool Calling 与副作用隔离 | 区分查询、生成动作草稿和真正写入 | `app/tools/engine.py`、`app/api/actions.py` | 高 |
| 混合检索与精排 | 解释 Dense、BM25、RRF、Reranker 的职责 | `app/core/retrieval.py` | 高 |
| 置信度门控 | 解释何时回答、复核和拒答 | `app/core/confidence.py`、`nodes.py` | 高 |
| 三层上下文 | 控制长对话输入并保留跨会话事实 | `app/core/memory.py`、`summarizer.py`、`long_term_memory.py` | 高 |
| 可观测与审计 | 区分在线 Trace、业务日志和持久审计 | `app/core/observability.py`、`app/main.py`、`app/db/models.py` | 中高 |

## 2. 重点亮点与学习顺序（先看这个）

| 亮点标题 | 为什么重要 | 通用技术关键词 | 先看哪些文件 | 建议顺序 |
| --- | --- | --- | --- | --- |
| 可控业务编排 | 八类意图只对应五种执行策略，固定流程不交给模型自由决定 | 状态机、条件路由、HITL | `app/graph/routing.py`、`build.py`、`nodes.py` | 1 |
| 检索质量治理 | 检索质量和回答权限在生成前完成判断 | Hybrid Search、RRF、Rerank、LLM Judge | `app/core/retrieval.py`、`confidence.py`、`selfcheck.py` | 2 |
| 副作用隔离 | Agent 只生成退款申请草稿，实际写入由用户确认后的 API 完成并进入人工审核 | 最小权限、确认门、审计 | `app/tools/builtin/refunds.py`、`app/api/actions.py` | 3 |
| 分层上下文 | 不同任务读取不同上下文，避免把所有历史无差别塞给模型 | 滑窗、增量摘要、向量记忆 | `app/config.py`、`app/core/memory.py` | 4 |
| 工具扩展与治理 | 内置工具和 MCP 服务统一发现，但业务路径再做最小白名单 | MCP、JSON Schema、超时重试、审计 | `app/tools/registry.py`、`engine.py`、`mcp_client.py` | 5 |
| 观测与留档 | Langfuse、滚动文件日志和 MySQL 审计分别承担不同职责 | Trace、结构化日志、持久化 | `app/core/observability.py`、`app/main.py` | 6 |

## 3. 必备知识点

- [ ] 能画出八类意图到五条路径的映射。
- [ ] 能解释订单/物流为什么不用 Agent。
- [ ] 能区分“能否退款判断、退款申请草稿、待审核工单、实际退款”。
- [ ] 能解释 Query 改写、Dense、BM25、RRF、Rerank 的先后关系。
- [ ] 能解释低阈值、高阈值和中间 LLM Judge 的职责。
- [ ] 能说明 Langfuse Trace、`log/app.log`、MySQL 工具审计各自保存什么。

## 4. 推荐阅读（结合仓库）

| 主题 | 通用技术点 | 建议阅读位置 | 预计时间 | 读完能回答什么 |
| --- | --- | --- | --- | --- |
| 主链路 | 状态图与条件边 | `app/graph/build.py` → `routing.py` | 25 分钟 | 为什么是五路而不是八套流程 |
| 确定性订单链 | 身份校验、MCP 串联、模板回答 | `app/graph/nodes.py` 中 `fetch_order`、`fetch_logistics`、`order_reply` | 30 分钟 | 为什么物流必须先查订单 |
| 售后链 | 取证、判断、草稿、确认 | `retrieve_policy`、`main_agent`、`agent_tools`、`app/api/actions.py` | 40 分钟 | Agent 到底有没有写权限 |
| RAG | 混合召回和精排 | `app/core/retrieval.py`、`app/kb/milvus_client.py` | 45 分钟 | RRF 和 Reranker 是否重复 |
| Context | 三层记忆与任务视图 | `app/config.py`、`app/core/memory.py`、`app/graph/nodes.py` | 45 分钟 | 8000 Token 如何分配 |
| Observability | Trace、日志和审计 | `app/core/observability.py`、`app/main.py`、`app/tools/engine.py` | 30 分钟 | Langfuse 停了系统是否还能跑 |

## 5. 自学提醒

若某文件或原理看不懂，请继续追问 AI；本技能负责给学习路径与题目，不提供逐行讲解。

## 6. 项目技术定位

后端与 AI Agent 交叉项目：FastAPI/MySQL 提供业务与持久化能力，LangGraph 负责业务状态编排，RAG、上下文工程与受限 Agent 负责非确定性理解和回答。

## 7. 核心原理解析

### 7.1 八类意图、五条执行路径

问题：按意图各写一套流程会产生重复，而全部交给 Agent 又可能漏查必需数据。机制：意图只描述用户诉求，执行路径描述系统控制策略。当前映射为：咨询→知识路径；订单/物流→订单路径；售后/退款退货→售后路径；投诉/人工→升级路径；其他→固定兜底。落点：`INTENT_TO_ROUTE` 与 LangGraph 条件边。

### 7.2 Workflow 与 Agent 的责任边界

问题：模型可能跳过订单归属校验、政策检索或错误执行副作用。机制：Workflow 固化身份校验、订单查询、物流串联、政策取证、置信度门控和人工升级；Agent 只在咨询及售后判断中使用，并按路径获得最小工具集。落点：`_ROUTE_TOOL_ALLOWLIST`。

### 7.3 售后写操作

问题：`submit_refund` 容易被理解为模型直接退款。机制：工具改名为 `prepare_refund_application`，只生成前端草稿；用户提交后创建状态为待处理的退款工单，等待人工审核，不调用支付退款。落点：`app/tools/builtin/refunds.py` 与 `/api/actions/create-refund`。

### 7.4 RAG 与拒答

问题：单路检索会在口语、型号和跨文档问题上偏科。机制：标准 Query 用 BGE-M3 做 Dense，扩展词仅进入 BM25，Milvus 用 RRF 融合候选，BGE-Reranker-v2-M3 精排；Top1 低于 0.22 直接拒答，达到 0.65 直接回答，中间区由独立 LLM 判断证据是否充分。

### 7.5 三层上下文

问题：全量历史会增加噪声和 Token 成本。机制：短期原文在 2400 Token 后异步触发摘要、3000 Token 前同步兜底并压缩至约 1800 Token；中期摘要与长期记忆共享 1200 Token 池；长期记忆按用户在 MySQL 保存事实、Milvus 语义召回 Top3。主 Agent 应用侧输入预算为 8000 Token。

### 7.6 可观测与存档

问题：只看 Langfuse 页面无法替代业务审计，只写本地日志又无法还原模型调用树。机制：Langfuse 按会话聚合在线 Trace，并通过 Docker named volumes 持久化到 PostgreSQL、ClickHouse、MinIO；`log/app.log` 按 10 MiB 滚动保留 10 份；消息、低置信问题和工具审计保存到 MySQL。三者互补，Langfuse 不可用时主流程继续运行。

## 8. 关键设计决策

| 决策 | 备选 | 当前取舍 | 风险 | 验证 |
| --- | --- | --- | --- | --- |
| 订单物流固定 Workflow | 交给通用 Agent 自选工具 | 固化先订单后物流 | MCP 不可用时只能返回订单快照 | 故障注入与节点测试 |
| 闲聊并入其他 | 保留九分类 | 使用统一自然引导 | 失去独立闲聊统计 | 通过其他子标签离线分析即可 |
| 售后保留 Agent | 全部规则引擎 | Workflow 取证、Agent 判断 | 仍需防止错误判定 | 结构化输出与人工审核 |
| 动态发现后再白名单 | 将全部工具暴露给模型 | 最小权限工具集 | 新工具需登记业务路径 | 工具集合单测 |
| 自建 Langfuse 持久化 | 仅控制台临时观察 | named volumes 保留 Trace | 本地磁盘损坏仍会丢失 | 定期备份卷/导出，生产配置保留策略 |

## 9. 量化与验证

- 已完成代码级验证：`pytest -q`，460 项通过。
- 历史 RAG 基线：300 题五桶评测中，混合精排 Recall@5=98.3%、MRR=89.4%、答案覆盖度=97.4%；本轮编排改造后尚未重跑全量在线链路。
- 历史阈值校准：低线 0.22 保留 95.4% 可答题；高线 0.65 对库外题直接放行率为 0%。这不能直接等价为完整“误拦率/误放率”，需要把中间区 LLM Judge 一起重测。
- 下一轮建议：固定评测集与模型版本，分别统计 Recall@5、MRR、端到端可答覆盖、误放率、误拦率、P50/P95 延迟和单问 Token。
