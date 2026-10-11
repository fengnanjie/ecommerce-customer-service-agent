# 电商客服助手Agent

基于 FastAPI、LangGraph、RAG 与 MCP 的电商智能客服系统。支持查询订单物流、回答政策 FAQ、
处理退款子流程、从对话中挖掘知识，并训练主题分类器。

代码是随课程一章章长出来的，不分支：ch01 一个纯对话接口起步，到 ch10 收尾时是下面这套东西。

## 技术栈

FastAPI + LangGraph / LangChain + SQLAlchemy / MySQL + Milvus。

聊天、嵌入、重排三组上游各自直连，没有网关那一层。模型名和地址都在 `.env` 里配
(`CHAT_*` / `EMBED_*` / `RERANK_*` 三组)，换供应商、换模型不用改代码。

## 快速开始

需要 Python 3.12、[uv](https://docs.astral.sh/uv/) 和 Docker。

```bash
cp .env.example .env      # 配置 CHAT_* / EMBED_* / RERANK_* 三组参数
uv sync                   # 安装依赖
docker compose up -d      # 启动 MySQL 与 Milvus
make seed                 # 写入业务测试数据
make dev                  # 启动 MCP 服务与主应用
```

浏览器打开 <http://localhost:8000> 就是聊天页。

## 代码怎么组织

| 位置 | 装的是什么 |
| - | - |
| `app/api/` | HTTP 入口。聊天、Agent、知识库录入、复核、验收页、成本看板 |
| `app/graph/` | LangGraph 那张图。`state` 状态、`nodes` 节点、`routing` 分流规则、`build` 组装 |
| `app/core/` | 单点能力。上游客户端、检索、重排、意图、指代、摘要、置信度、飞轮、可观测 |
| `app/kb/` | 知识怎么进库。切块、嵌入、双写 MySQL 与 Milvus、去重、从对话里挖问答对 |
| `app/tools/` | 工具系统。内置 `@tool`、MCP 客户端、注册表、统一执行引擎 |
| `app/db/` | 表模型与仓储 |
| `app/static/` | 前端页面。聊天、知识库录入、飞轮待审、观测与成本、主题分布、分类器验收 |
| `mcp_servers/` | 两台业务 MCP Server，物流和售后各一台，独立进程 |
| `sql/` | 各章的建表与迁移，容器首启按文件名顺序自动执行 |
| `scripts/` | 建库、评估、微调这些离线活 |
| `docs/superpowers/` | 各章的 spec 和 plan。课程实战篇教的就是这套流程，留着当范本 |

## 当前业务编排（8 类意图 → 5 条路径）

| 路径 | 意图 | 执行方式 |
| - | - | - |
| 咨询 | 咨询 | Workflow 强制完成 Query 归一化、混合检索和证据门控，Agent 仅基于证据回答，并可按需查询动态商品数据 |
| 订单物流 | 订单、物流 | 固定 Workflow：校验订单归属并查订单；物流问题再用订单中的物流单号调用物流 MCP，最后模板化合并输出 |
| 售后退款 | 售后、退款退货 | 固定取订单与政策证据，Agent 只做证据上的判断、售后状态查询或生成退款申请草稿；提交后创建待人工审核工单，不直接退款 |
| 人工升级 | 投诉、人工 | 固定安抚/升级话术，提供转人工与建工单入口，不进入 Agent |
| 兜底 | 其他 | 寒暄、越界和表达不清统一使用固定引导，不调用模型 |

工具层统一登记内置工具与物流/售后 MCP 工具，并由 JSON Schema、超时、只读重试、
写权限和审计日志治理。Agent 还会按业务路径应用最小工具白名单；新增同类只读工具可通过
注册中心扩展，但新增强制步骤、写操作或确认节点仍须修改 LangGraph。

## 各章长出了什么，怎么验

`make test` 跑全部单测，不打真实模型。下面这些要真服务在跑。

| 章 | 这一章长出来的东西 | 验收 |
| - | - | - |
| ch01 | 流式对话、结构化提取 | `make eval` |
| ch02 | 五个 `@tool` 业务工具，单轮 Function Calling | `make eval-agent` |
| ch03 | 切块、嵌入、MySQL 与 Milvus 双写、对话挖知识 | `make kb-build` `make kb-vectorize` `make eval-retrieval` |
| ch04 | 混合检索、RRF、重排、Query 改写、四策略评估 | `make smoke-rag` `make eval-rag` |
| ch05 | LangGraph Workflow 骨架 + 受限 Agent 的 ReAct 环 | `make eval-ch05` |
| ch06 | 八类意图五路分流、指代消解、订单选择与售后申请流程 | `make smoke-interrupt` `make eval-ch06` |
| ch07 | 上下文管理。滑窗、摘要、前缀缓存 | `make eval-ch07` |
| ch08 | 工具系统。MCP 动态发现、统一执行引擎、审计日志 | `make eval-ch08` |
| ch09 | Langfuse 自部署、数据飞轮、成本账 | `make langfuse-up` `make flywheel` `make eval-flywheel` `make cost-report` |
| ch10 | 主题分类器。语料、微调、阈值扫描、ONNX 推理服务 | `make ch10-corpus` `make ch10-train` `make ch10-eval` |

运行 `make help` 可以查看带说明的完整目标清单。

## 端口

| 端口 | 是什么 |
| - | - |
| 8000 | 应用 |
| 8101 / 8102 | 业务 MCP Server，物流 / 售后 |
| 8110 | ch10 主题分类器推理服务（`make classifier-up` 之后） |
| 3000 | Langfuse（`make langfuse-up` 之后） |
| 19530 | Milvus |

应用那几个页面：`/` 聊天、`/kb` 知识库录入、`/review` 飞轮待审、`/observability` 观测与成本、
`/topics` 主题分布、`/acceptance` 分类器验收。
