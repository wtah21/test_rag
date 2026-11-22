"""Milvus（Mivilus 云数据库）混合检索模块。

采用“稠密 + 稀疏”双路检索：
- 稠密向量：硅基流动 BGE-M3（仅稠密），字段 dense
- 稀疏向量：BM25（jieba 中英分词），字段 sparse
两路结果按 RRF（Reciprocal Rank Fusion）算法融合，支持元数据过滤表达式。

BM25 模型在入库时对知识库语料进行拟合（计算 IDF 与词表），持久化为 pickle；
检索端加载同一模型将查询文本编码为 BM25 稀疏向量，保证与库内权重一致。
"""

from __future__ import annotations

import os
import pickle
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from langchain_core.documents import Document

from shared.rerank import rerank_documents

# 项目根目录（src/shared/hybrid_retrieval.py -> 上两级）
ROOT = Path(__file__).resolve().parent.parent.parent

# 连接参数（读取 .env）
MIVILUS_URI = os.environ.get("Mivilus_URL", "")
MIVILUS_TOKEN = os.environ.get("Mivilus_API_KEY", "")

# collection 与字段约定（与 scripts/index_knowledge.py 保持一致）
COLLECTION_NAME = os.environ.get("MIVILUS_COLLECTION", "enterprise_knowledge")
DENSE_FIELD = "dense"  # FLOAT_VECTOR，BGE-M3 1024 维
SPARSE_FIELD = "sparse"  # SPARSE_FLOAT_VECTOR，BM25
TEXT_FIELD = "text"  # 分块后的文档正文
# 可被 filter 引用的元数据字段（白名单）
FILTERABLE_FIELDS = {
    "pk",
    "doc_title",
    "category",
    "source",
    "url",
    "H2",
    "H3",
}

DENSE_EMBEDDING_MODEL = "siliconflow/BAAI/bge-m3"
DENSE_DIM = 1024  # BGE-M3 稠密向量维度

# 双路各取 top-k，RRF 融合后取 top-n
ROUTE_TOP_K = 20
FINAL_TOP_K = 8
# RRF 融合常数
RRF_K = 60

# BM25 模型持久化路径（入库脚本保存，检索端加载）
BM25_MODEL_PATH = Path(
    os.environ.get("BM25_MODEL_PATH", str(ROOT / "knowledge" / ".bm25_model.pkl"))
)

# 检索表达式允许使用的逻辑操作符
_OP_TOKENS = {"AND", "OR", "NOT", "IN"}

_EN_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "of",
    "for",
    "in",
    "on",
    "to",
    "at",
    "by",
    "with",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "we",
    "you",
    "your",
    "they",
    "them",
    "their",
    "he",
    "she",
    "his",
    "her",
    "not",
    "do",
    "does",
    "did",
    "can",
    "could",
    "will",
    "would",
    "should",
    "may",
    "might",
    "have",
    "has",
    "had",
    "about",
    "into",
    "over",
    "after",
    "before",
    "between",
    "under",
    "again",
    "further",
    "once",
    "here",
    "there",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "just",
    "don",
    "now",
    "i",
    "me",
    "my",
    "ourselves",
    "ours",
    "what",
    "which",
    "who",
    "whom",
    "why",
    "how",
    "when",
    "where",
    "up",
    "down",
    "out",
    "off",
    "above",
    "below",
}

_ZH_STOPWORDS = {
    "的",
    "了",
    "和",
    "与",
    "或",
    "在",
    "是",
    "有",
    "被",
    "为",
    "也",
    "都",
    "这",
    "那",
    "之",
    "而",
    "及",
    "等",
    "中",
    "对",
    "并",
    "但",
    "可以",
    "一个",
    "我们",
    "你们",
    "他们",
    "自己",
    "因为",
    "所以",
    "如果",
    "就是",
    "进行",
    "这个",
    "那个",
    "我们",
    "使用",
    "用于",
    "以及",
}


_SYMBOL_ONLY = re.compile(r"^[\W_]+$", re.UNICODE)


def jieba_analyzer(text: str) -> list[str]:
    """模块级中英混合分词器（jieba 分词 + 去停用词/纯符号 token）。

    必须为模块级函数，以便随 BM25 模型一起被 pickle 持久化。
    """
    import jieba

    jieba.initialize()
    stopwords = _EN_STOPWORDS | _ZH_STOPWORDS
    tokens: list[str] = []
    for token in jieba.lcut(text):
        token = token.strip().lower()
        if not token or token in stopwords or _SYMBOL_ONLY.match(token):
            continue
        tokens.append(token)
    return tokens


def build_jieba_analyzer() -> Callable[[str], list[str]]:
    """构建中英混合分词器（jieba 分词 + 去停用词/纯符号 token）。"""
    return jieba_analyzer


def build_bm25(
    analyzer: Callable[[str], list[str]] | None = None,
) -> Any:
    """构建 BM25 稀疏向量函数（需在入库时对语料调用 fit）。"""
    from milvus_model.sparse.bm25 import BM25EmbeddingFunction

    return BM25EmbeddingFunction(analyzer=analyzer or build_jieba_analyzer())


def save_bm25(bm25: Any, path: Path = BM25_MODEL_PATH) -> Path:
    """持久化已拟合的 BM25 模型（检索端加载同一词表与 IDF）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bm25, f)
    return path


def load_bm25(path: Path = BM25_MODEL_PATH) -> Any:
    """加载入库时拟合的 BM25 模型。"""
    with open(path, "rb") as f:
        bm25 = pickle.load(f)
    # milvus_model 0.2.x 中 BM25EmbeddingFunction 为工厂函数而非类，无法 isinstance，
    # 改用鸭子类型校验：必须同时具备文档/查询编码能力。
    if not hasattr(bm25, "encode_queries") or not hasattr(bm25, "encode_documents"):
        raise ValueError(f"BM25 模型文件格式不正确: {path}")
    return bm25


def _csr_to_dict(sparse_vec: Any) -> dict[int, float]:
    """将 scipy 稀疏向量转为 Milvus 稀疏查询所需的 {token_id: weight}。"""
    coo = sparse_vec.tocoo()
    return {int(i): float(v) for i, v in zip(coo.col, coo.data)}


def make_client() -> Any:
    """创建 Mivilus（Zilliz Cloud Milvus）客户端。"""
    from pymilvus import MilvusClient

    uri = os.environ.get("Mivilus_URL", MIVILUS_URI)
    token = os.environ.get("Mivilus_API_KEY", MIVILUS_TOKEN)
    if not uri or not token:
        raise ValueError("缺少 Mivilus 连接配置（Mivilus_URL / Mivilus_API_KEY）")
    return MilvusClient(uri=uri, token=token)


def validate_filter(expr: str) -> str:
    """校验元数据过滤表达式，防止非法语法/注入并确保字段在白名单内。

    支持形如：category == "langgraph" 或 category in ["langchain", "langgraph"]。
    返回空串表示不过滤。
    """
    expr = (expr or "").strip()
    if not expr:
        return ""
    # 剥离引号内的字符串字面量，引号内允许任意字符（如 URL、文件路径等值），
    # 仅对表达式骨架做非法字符检查，避免误伤字面量内容（如 / : . 等）。
    stripped = re.sub(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'", '""', expr)
    # 剥离字面量后骨架为空（如 LLM 误输出 "\"\"" 这类纯引号串），视为不过滤
    if not stripped.strip().strip('"'):
        return ""
    if re.search(r"[^\w\"'\s=!<>()\[\],.&|*]", stripped):
        raise ValueError(f"过滤表达式含非法字符: {expr}")
    # 校验剩余 token 均为逻辑操作符或白名单字段
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", stripped):
        if token.upper() in _OP_TOKENS or token in FILTERABLE_FIELDS:
            continue
        raise ValueError(
            f"过滤表达式引用了未知字段 '{token}'，可用字段: "
            f"{sorted(FILTERABLE_FIELDS)}。表达式: {expr}"
        )
    return expr


def hybrid_search(
    query: str,
    filter_expr: str = "",
    route_top_k: int = ROUTE_TOP_K,
    final_top_k: int = FINAL_TOP_K,
    sparse_query: Optional[str] = None,
    enable_sparse: bool = True,
    rerank: bool = False,
    rerank_top_n: int = 4,
    rerank_candidates: int = 30,
) -> list[Document]:
    """对单个查询执行“稠密 + 稀疏”双路检索并用 RRF 融合结果。

    知识库语料为英文而提问常为中文：稠密路（BGE-M3，多语言/跨语种）直接使用
    query 即可命中；稀疏路 BM25 是纯词匹配，中文分词打不中英文语料词表，因此
    支持传入 sparse_query（英文检索关键词）作为 BM25 的编码文本，缺省时退化为
    query。对不含英文专名的宽泛概念题，BM25 通用词命中多为泛化文档、RRF 反而
    挤掉稠密路的好文档，可通过 enable_sparse=False 关闭稀疏路（结果等同于
    纯稠密检索，跳过 BM25 编码与稀疏检索）。

    Args:
        query: 用户查询文本（稠密向量检索用，通常为中文原问题或改写后问题）。
        filter_expr: Milvus 元数据过滤表达式（如 category == "langgraph"），空串不过滤。
        route_top_k: 每路检索返回的候选数。
        final_top_k: RRF 融合后返回的文档数。
        sparse_query: BM25（稀疏）检索用查询文本；为 None 时与 query 相同。
        enable_sparse: 是否启用稀疏路；False 时仅稠密检索（跳过 RRF 融合，
            结果即稠密路 top final_top_k）。
        rerank: 是否在 RRF 融合后用 cross-encoder 重排；False 时行为不变。
        rerank_top_n: 重排后返回的文档数。
        rerank_candidates: 送入重排的 RRF 候选数（>= rerank_top_n）。

    Returns:
        按相关性降序排列的 Document 列表（正文 + 元数据）；开启重排时返回
        rerank_top_n 条，否则返回 final_top_k 条。
    """
    from shared.retrieval import make_text_encoder

    filter_expr = validate_filter(filter_expr)
    sparse_text = (
        sparse_query if (sparse_query and str(sparse_query).strip()) else query
    )

    # 稠密查询向量（BGE-M3 via 硅基流动）
    dense_encoder = make_text_encoder(DENSE_EMBEDDING_MODEL)
    dense_vec = dense_encoder.embed_query(query)

    client = make_client()
    output_fields = [TEXT_FIELD, *sorted(FILTERABLE_FIELDS - {"pk"})]

    # 稠密路检索
    dense_hits = client.search(
        collection_name=COLLECTION_NAME,
        data=[dense_vec],
        anns_field=DENSE_FIELD,
        limit=route_top_k,
        filter=filter_expr or None,
        output_fields=output_fields,
        search_params={"metric_type": "COSINE"},
    )[0]

    sparse_hits: list[Any] = []
    if enable_sparse:
        # 稀疏查询向量（BM25，与入库同模型同词表；文本为检索关键词以匹配英文语料）
        bm25 = load_bm25()
        sparse_vec = _csr_to_dict(bm25.encode_queries([sparse_text])[0])
        sparse_hits = client.search(
            collection_name=COLLECTION_NAME,
            data=[sparse_vec],
            anns_field=SPARSE_FIELD,
            limit=route_top_k,
            filter=filter_expr or None,
            output_fields=output_fields,
            search_params={"metric_type": "IP"},
        )[0]

    # RRF 融合：score(doc) = Σ 1/(RRF_K + rank)，按文档主键聚合。
    # 注意：主键字段名为 pk（见 schema），pymilvus 3.x 中 Hit 顶层 dict 以
    # {pk 名: 值, distance, entity} 组织，需通过 hit.id / hit["pk"] 读取主键。
    rrf_scores: dict[int, float] = {}
    for hits in (dense_hits, sparse_hits):
        for rank, hit in enumerate(hits, start=1):
            pk = int(hit.id)
            rrf_scores[pk] = rrf_scores.get(pk, 0.0) + 1.0 / (RRF_K + rank)

    merged = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    limit = rerank_candidates if rerank else final_top_k
    candidates = _rrf_candidates(merged, dense_hits, sparse_hits, limit)
    if rerank and candidates:
        return rerank_documents(query, candidates, top_n=rerank_top_n)
    return candidates


def _rrf_candidates(
    merged: list[tuple[int, float]],
    dense_hits: list[Any],
    sparse_hits: list[Any],
    limit: int,
) -> list[Document]:
    """按 RRF 融合次序回查实体，构建前 limit 条候选文档。"""
    by_id = {int(hit.id): hit for hit in dense_hits}
    for hit in sparse_hits:
        by_id.setdefault(int(hit.id), hit)

    documents: list[Document] = []
    for pk, score in merged[:limit]:
        entity = by_id.get(pk)
        if entity is None:
            continue
        metadata = {field: entity.get(field) for field in FILTERABLE_FIELDS - {"pk"}}
        metadata = {k: v for k, v in metadata.items() if v is not None}
        metadata["rrf_score"] = round(score, 4)
        documents.append(
            Document(page_content=entity.get(TEXT_FIELD, ""), metadata=metadata)
        )
    return documents
