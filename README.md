# RAG 研究型问答 Agent

面向 LangChain / LangGraph 官方文档的对话式 RAG Agent：先判断用户问题该不该查知识库，再决定是否改写问题、走向量库检索、评判检索结果是否够用，最后带着上下文作答；同时具备跨会话的长期记忆。

> 本项目改造自 [langchain-ai/rag-research-agent-template](https://github.com/langchain-ai/rag-research-agent-template)（MIT License），原始的 `index_graph`、`researcher_graph` 等模板代码已在重构中移除。

## 功能特性

- **三层级联路由**：规则关键词 → 语义路由（BGE-M3 + semantic-router）→ JEV 决策模型，逐层兜底，把问题分为 `langchain` / `general` / `more-info` 三类。
- **混合检索**：稠密向量（硅基流动 BGE-M3，1024 维）+ 稀疏向量（BM25，jieba 中英分词）双路召回，RRF 融合后用智谱 cross-encoder 重排。
- **意图改写**：问题表述模糊时先分析意图并改写成适合检索的规范问题；表述清晰则原样检索（可用 `enable_intent_rewrite` 关闭做对照实验）。
- **结果自检**：`grade_documents` 给检索结果打 0~1 分，不合格就回到改写/检索环节重试，最多 3 次，超限后用历轮最高分文档强制作答。
- **澄清中断**：信息不足时用 `interrupt` 向用户追问并等待补充，同一轮最多 2 次，无法中断时降级为普通作答。
- **短期记忆**：Postgres checkpointer 按 `thread_id` 保存会话消息历史，支持多轮对话与中断恢复。
- **长期记忆**：Postgres store 按用户保存 `fact`（客观事实）、`preference`（回答偏好）、`status`（阶段性状态）三类记忆，按 `相似度 × 权重` 检索，`status` 类随时间衰减、随访问次数加成。
- **可降级**：Postgres 未配置或连接失败时，记忆功能整体退化为不可用，主流程照常运行。

## 架构

### 图结构

```
START
  └─ analyze_and_route_query        # 三层路由
       ├─ general    ─────────────► respond_to_general_query ─► END
       ├─ more-info  ─────────────► ask_for_clarification ───┐
       └─ langchain  ─────────────► analyze_and_rewrite_question
                                          └─► retrieve_context
                                                └─► grade_documents
                                                      ├─ good ─► respond ─► END
                                                      └─ bad  ─► 回到 analyze_and_rewrite_question（≤3 次）
                                                        （ask_for_clarification 补充后回到路由节点）
```

| 节点 | 职责 |
|---|---|
| `analyze_and_route_query` | 调用 `resolve_route` 做三层路由，结果写入 `router` 与 `route_layer` |
| `ask_for_clarification` | 大模型生成追问话术，`interrupt` 等待用户补充 |
| `analyze_and_rewrite_question` | 判断问题是否需要改写，产出检索用的有效问题 |
| `retrieve_context` | 抽取英文检索关键词 + 单 query 混合检索 + 重排，产出最终 `documents` |
| `grade_documents` | 给检索结果打分，记录 `grade_score` / `best_documents` |
| `respond` | 结合文档、会话历史与长期记忆生成回答 |
| `respond_to_general_query` | 不查知识库，直接回答普通问题 |

### 检索链路

1. **稠密路**：用中文有效问题走 BGE-M3（多语言、可跨语种命中英文语料）。
2. **稀疏路**：用 LLM 抽取的英文检索关键词走 BM25（纯词匹配，中文分词打不中英文词表，所以要做关键词抽取）。
3. **融合**：两路各取 20 条，按 RRF（`k=60`）合并。
4. **重排**：取前 20 条候选送智谱 cross-encoder，保留前 4 条作为最终上下文（重排失败自动回退为原序）。
5. **自检**：`grade_documents` 判定不足则重试，最多 3 轮。

### 记忆

- **短期记忆**：`langgraph` 的 `PostgresSaver`，按 `thread_id` 隔离会话，是 `interrupt` 澄清能正常工作的前提。
- **长期记忆**：`PostgresStore` 以 BGE-M3 向量索引记忆正文，命名空间为 `("memories", user_id)`。两个工具挂在回复节点上，由模型自行决定是否调用：
  - `load_user_memories`：按问题做向量检索，取相关性最高的 5 条；
  - `update_user_memory`：读取最近 2 轮对话，交给 langmem extractor 抽取并写入（模型只决定“要不要存”，不负责写什么）。
- 记忆权重：`fact` / `preference` 恒为 1；`status` 按 `exp(-λ·Δt) + 0.1·ln(1+access_count)` 计算，半衰期 30 天。

## 目录结构

```
.
├── langgraph.json              # LangGraph 平台入口：图定义与 env 文件
├── docker-compose.yml          # 本地 Postgres（pgvector 镜像）
├── pyproject.toml              # 依赖与工具配置
├── .env.example                # 环境变量模板
├── knowledge/                  # 知识库原始 Markdown（索引脚本的输入）
├── scripts/
│   └── index_knowledge.py      # 分块 + 向量化 + 写入 Milvus
├── src/
│   ├── retrieval_graph/
│   │   ├── graph.py            # 主图：路由、检索、评判、回复
│   │   ├── routing.py          # 三层路由实现
│   │   ├── memory.py           # 短期/长期记忆
│   │   ├── configuration.py    # 可配置项
│   │   ├── prompts.py          # 全部提示词
│   │   └── state.py            # 图状态定义
│   └── shared/
│       ├── hybrid_retrieval.py # 稠密+稀疏检索、RRF 融合
│       ├── retrieval.py        # 嵌入模型工厂
│       ├── rerank.py           # 智谱 cross-encoder 重排
│       ├── configuration.py    # 公共配置
│       └── utils.py            # 模型加载、文档格式化
└── tests/unit_tests/           # 单元测试
```

## 环境要求

| 依赖 | 说明 |
|---|---|
| Python 3.10 ~ 3.14 | 项目以 3.12 验证，`.python-version` 已指定 3.12 |
| [uv](https://docs.astral.sh/uv/) | 依赖与虚拟环境管理 |
| Docker Desktop | 本地 Postgres（含 pgvector） |
| Zilliz Cloud 实例 | 向量库，Serverless 即可；记录 URI 与 API Key |
| 智谱开放平台 Key | 对话模型（GLM）与 cross-encoder 重排共用 |
| 硅基流动 Key | BGE-M3 嵌入模型 |
| JEV Key | 三层路由第三层决策模型（前两层命中时不会被调用） |

## 快速开始

### 1. 安装依赖

```powershell
uv sync --extra dev
```

### 2. 配置环境变量

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`，按下表填写。**`.env` 已在 `.gitignore` 中，不要提交。**

| 变量 | 必填 | 用途 |
|---|---|---|
| `GLM_API_KEY` | 是 | 智谱 GLM 对话模型 + 重排接口 |
| `SiliconEmbedding_API_KEY` | 是 | 硅基流动 BGE-M3 嵌入（知识库与记忆共用） |
| `Mivilus_URL` / `Mivilus_API_KEY` | 是 | Zilliz Cloud / Milvus 连接 |
| `POSTGRES_URI` | 建议 | 记忆后端连接串，格式 `postgresql://<user>:<password>@localhost:5433/rag_memory` |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | 建议 | 供 `docker compose` 读取，**必须与 `POSTGRES_URI` 中的账号/密码/库名一致** |
| `JEV_API_KEY` | 是 | 路由第三层 |
| `JEV_BASE_URL` / `JEV_MODEL` | 否 | 默认 `https://api.typesafe.ai/v1`、`jev-latest` |
| `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` / `LANGSMITH_TRACING` | 否 | 调用链追踪 |

可选项：`SILICONFLOW_BASE_URL`（嵌入端点）、`MIVILUS_COLLECTION`（collection 名，默认 `enterprise_knowledge`）、`JEV_DECIDE_PATH`、`ZHIPU_RERANK_URL`、`ZHIPU_RERANK_MODEL`、`BM25_MODEL_PATH`。

### 3. 启动 Postgres

```powershell
docker compose up -d
docker compose ps
```

镜像使用 `pgvector/pgvector:pg16`，因为长期记忆需要 `pgvector` 扩展（官方 `postgres:16` 镜像不含该扩展，长期记忆会直接禁用）。

> 如果之前用旧镜像起过同名容器/数据卷，先执行 `docker compose down -v` 清空旧数据再启动。

### 4. 建立知识库索引

```powershell
uv run python scripts/index_knowledge.py
```

脚本会扫描 `knowledge/**/*.md`，按 `##` / `###` 标题结构分块（块大小 800、重叠 80），抽取 frontmatter 元数据，生成稠密 + BM25 稀疏向量，并**删除并重建** collection 后批量写入。

> 该脚本会 `drop_collection`，不要在生产 collection 上直接运行。首次运行还会在 `knowledge/.bm25_model.pkl` 生成 BM25 模型（检索端加载同一份词表，删掉后需重新索引）。

### 5. 启动服务

```powershell
uv run langgraph dev
```

- API 默认地址：`http://127.0.0.1:2024`
- 打开终端里输出的 Studio 链接即可在网页里对话、查看每个节点的执行轨迹
- 图名在 `langgraph.json` 中注册为 `retrieval_graph`

调用时通过 `configurable` 传入：

- `thread_id`：短期记忆的会话标识，多轮对话必须传同一个值；
- `user_id`（或平台注入的 `langgraph_user_id`）：长期记忆的归属用户，缺省为 `default-user`。

## 测试与代码检查

```powershell
uv run pytest tests/unit_tests
uv run ruff check .
uv run ruff format .
uv run mypy --strict src/
```

## 常用可配置项

通过 Studio 或调用时的 `configurable` 覆盖，默认值见 `src/retrieval_graph/configuration.py`：

| 配置 | 默认值 | 说明 |
|---|---|---|
| `query_model` | `zhipu/glm-5.3-flash` | 路由/改写/评判用模型 |
| `response_model` | `zhipu/glm-5.3-flash` | 生成回答用模型（需支持工具调用） |
| `memory_model` | `zhipu/glm-5.3-flash` | 长期记忆抽取用模型 |
| `embedding_model` | `siliconflow/BAAI/bge-m3` | 嵌入模型 |
| `enable_intent_rewrite` | `true` | 关闭后直接用原问题检索 |
| `context_top_k` | `8` | 关闭重排时进入回答的文档数 |
| `rerank_enabled` | `true` | 是否启用智谱重排 |
| `rerank_candidates` / `rerank_top_n` | `20` / `4` | 送排候选数与最终保留数 |
| `routing_rule_keywords` | 见配置 | 第一层规则路由关键词 |
| `routing_semantic_examples` | 见配置 | 第二层语义路由示例问题 |
| `routing_semantic_auto_fit` | `true` | 是否自动拟合语义路由阈值 |

## 常见问题

**长期记忆不可用（日志提示 PostgresStore 初始化失败）**
Postgres 缺少 `pgvector` 扩展或连接串错误。确认使用的是 `pgvector/pgvector:pg16` 镜像，且 `POSTGRES_URI` 可连通、账号密码与 `POSTGRES_USER` / `POSTGRES_PASSWORD` 一致。

**澄清追问不生效，直接进入普通回答**
没有可用的 checkpointer（`POSTGRES_URI` 未配置或连接失败）时无法中断，代码会降级为普通作答。修复 Postgres 后重启即可。

**检索结果为空 / 回答与知识库无关**
先确认第 4 步索引成功，且 `MIVILUS_COLLECTION` 与索引时的 collection 一致；再确认 `knowledge/.bm25_model.pkl` 存在（缺失会导致稀疏路报错，重新跑索引脚本即可）。

**接口返回 401 / 403**
对应服务的 Key 未填、填错或额度不足：智谱（对话+重排）、硅基流动（嵌入）、Zilliz（向量库）、JEV（路由第三层）分别检查。

**端口被占用**
`5433`（Postgres）或 `2024`（langgraph dev）被占用时，改 `docker-compose.yml` 的端口映射或 `uv run langgraph dev --port <端口>`，并同步修改 `POSTGRES_URI`。

**Windows 上命令报错**
本项目示例命令均为 PowerShell。`make` 目标文件已移除，直接使用上文 `uv run ...` 命令即可。

## 已知限制

- 检索是**单 query** 直检索（不再生成多 query、不做 step 级配额合并），复杂多跳问题依赖 `grade_documents` 的重试闭环。
- 知识库语料为英文、提问多为中文，稀疏路必须依赖 LLM 抽取的英文关键词，抽取质量直接影响 BM25 命中。
- `langgraph dev` 平台模式（`LANGSMITH_LANGGRAPH_API_VARIANT=local_dev`）下由平台依据 `POSTGRES_URI` 托管持久化，图内不再挂自定义 checkpointer/store；脱离平台直接运行时才显式接入（见 `graph.py` 末尾分支）。
- 长期记忆的写入由模型自主触发，并非每轮必写。

## 许可证

本项目沿用上游的 MIT License（Copyright (c) 2024 LangChain），详见 [LICENSE](./LICENSE)。分发或再发布时请保留原始版权声明与许可文本。
