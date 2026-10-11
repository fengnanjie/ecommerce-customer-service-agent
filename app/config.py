# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 必填,故意不给默认值。模型名是跟着账号走的,给一个默认值等于替学员猜:猜错不会在
    # 启动时报错,而是在第一次调模型时抛一个看不懂的上游 401。没有默认值,缺了就在启动
    # 时报 chat_model Field required,直接指向 .env。调优常量(top_k、步数、阈值这些)
    # 是另一类,它们该留在代码里带默认值。
    chat_model: str
    # 三组上游各自直连,不经网关。base_url 和 key 跟着账号走,所以和 chat_model 一样必填;
    # 嵌入和重排都在硅基流动、地址固定,给了默认值,只有 key 要填。
    chat_base_url: str
    chat_api_key: str
    embed_base_url: str = "https://api.siliconflow.cn/v1"
    embed_api_key: str
    rerank_base_url: str = "https://api.siliconflow.cn/v1"
    rerank_api_key: str
    # 思考链控制。三项都留空就用上游默认,填了才传。
    # 关不关是上游的能力、不是参数写法:同一个 thinking.type,硅基流动的 DeepSeek 和
    # MiniMax-M3 上能关(实测 22.4s/318 token → 1.4s/16),MiniMax-M2.x 上会被静默忽略。
    chat_thinking: str = ""           # disabled 关 / adaptive 开
    chat_reasoning_effort: str = ""   # 开着时的强度:low / high / max
    chat_reasoning_split: str = ""    # MiniMax 专属:true 让思考进 reasoning_content 独立字段,
                                      # 而不是混在 content 里的 <think>。关不掉思考链时靠它兜底
    token_budget: int = 2000
    database_url: str = "mysql+asyncmy://root:root@localhost:3306/ecommerce_agent"
    test_database_url: str = "mysql+asyncmy://root:root@localhost:3306/ecommerce_agent_ch02_test"
    # ch03 知识库检索
    embed_model: str = "BAAI/bge-m3"          # 上游真实名(无网关别名可用)
    milvus_uri: str = "http://localhost:19530"   # ch04: Lite → Standalone
    # ch04 混合检索 + 重排
    rerank_model: str = "BAAI/bge-reranker-v2-m3"   # 上游真实名
    recall_top_k: int = 50        # dense/BM25 各召回 Top-50
    rerank_top_k: int = 10        # 精排出 Top-10
    subquery_split: bool = True   # 一问多意图时按子句拆开各检索一遍再轮转合并(见 retrieval)
    # 三段式证据门控：低于 reject 直接拒答；[reject, accept) 交 LLM 自检；
    # 高于 accept 直接放行。默认值会由 scripts/calibrate_confidence.py 重新校准。
    rerank_reject_threshold: float = 0.22  # 完整线上链路校准:240 可答题保留率 95.4%
    rerank_accept_threshold: float = 0.65  # 库外最高0.629之上留0.02安全余量
    # ch05 图编排
    max_agent_steps: int = 6              # ReAct 环最大步数(封顶,超则兜底)
    # token 花销不在环里卡:那是成本控制,归 ch09 跟 Langfuse 的账一起看。
    # tokens_used 仍逐步累加,进 trace 供排障和统计,只是不再当停止条件。
    checkpointer_db_path: str = "data/ch05_checkpoints.sqlite"  # LangGraph checkpointer(data/*.db* 已 gitignore)
    # ch06 意图识别模型配置(先求准:默认大模型;降级路仅种子,本章不实现运行时)
    intent_model: str = ""            # 意图识别模型名;空=回落 chat_model
    intent_small_model: str = ""      # 降级路小模型(种子,未接运行时)
    intent_mode: str = "accuracy"     # accuracy=只用大模型;cost=小模型判→低置信升级大模型(未实现)
    intent_conf_threshold: float = 0.6  # cost 模式升级阈值(种子)
    # ch07/11 Context Engineering:应用侧输入上限 + 三层记忆
    # 8000 是本应用主动控制的输入预算,不是模型厂商的上下文长度；输出另由 TOKEN_BUDGET 控制。
    context_total_max_tokens: int = 8000
    context_tool_schema_reserve_tokens: int = 1000
    context_current_user_max_tokens: int = 600  # 超限显式拒绝,不静默截断用户问题
    context_window_soft_tokens: int = 2400    # 轮后异步摘要软阈值
    context_window_max_tokens: int = 3000     # 下轮调用前同步压缩硬阈值
    context_window_target_tokens: int = 1800  # 压缩后短期原文目标
    context_window_turns: int = 8             # 最多保留 8 个完整 user/assistant 轮次
    context_window_min_turns: int = 2         # 再长也至少保住最近 2 个完整轮次
    context_memory_pool_tokens: int = 1200    # 中期摘要与长期记忆共享池
    context_summary_max_tokens: int = 800
    context_long_term_max_tokens: int = 600
    context_route_evidence_max_tokens: int = 1200
    context_coref_max_tokens: int = 1200
    context_intent_max_tokens: int = 800
    context_coref_recent_turns: int = 3
    context_intent_recent_turns: int = 2
    summary_model: str = ""                  # 摘要模型,空=回落 chat_model
    # 长期记忆:MySQL 是事实源,Milvus 仅作可重建的用户级向量索引。
    long_term_memory_enabled: bool = True
    long_term_memory_collection: str = "customer_long_term_memories"
    long_term_memory_top_k: int = 3
    long_term_memory_candidate_k: int = 8
    # 初始经验阈值,上线前需用真实记忆检索集校准,不能当作效果结论。
    long_term_memory_threshold: float = 0.62
    long_term_memory_extract_every_turns: int = 6
    long_term_memory_max_candidates: int = 4
    # ch08 工具系统(注册中心 + 执行引擎 + MCP 接入)
    mcp_logistics_url: str = "http://127.0.0.1:8101/mcp"    # 物流 MCP Server
    mcp_aftersales_url: str = "http://127.0.0.1:8102/mcp"   # 售后 MCP Server
    tool_default_timeout: float = 5.0    # 内置工具默认超时(秒)
    mcp_tool_timeout: float = 10.0       # MCP 工具默认超时(走 HTTP,放宽)
    tool_max_retries: int = 2            # 只读工具暂时性故障最大重试次数
    demo_ticket_delay_seconds: float = 0.0  # 验收 6:>0 时 create_ticket 人为变慢(写超时演示)
    # ch09 可观测(Langfuse 自部署;三者齐全才挂回调,缺省时系统照常跑、测试环境不依赖)
    # 环境变量名与课程 README/SDK 一致:LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = ""     # 如 http://localhost:3000(自部署地址,数据不出门)
    # ch09 置信度闸(正式版):阈值由 make calibrate-confidence 在 ch04 评估集上校准回填,不拍脑袋
    # 2026-09-27 实测:可答桶 p50=0.715 vs 应拒桶 p50=0.102/max=0.287,YoudenJ 最优点 0.17


settings = Settings()
