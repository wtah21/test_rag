"""将 knowledge/ 目录下的官方文档分块、向量化并入库到 Mivilus（Zilliz Cloud Milvus）。

流程：
1. 扫描并加载 knowledge/**/*.md
2. 解析每个文档的 YAML frontmatter（title / source / category / url）
3. 基于文档结构（Markdown 标题层级）分块：按 H1~H4 标题切分，保留标题层级链
4. 超长块按 Markdown 分隔符二次切分，控制块大小
5. 生成双路向量：
   - 稠密向量：硅基流动 BGE-M3（OpenAI 兼容接口，1024 维）
   - 稀疏向量：BM25（jieba 中英分词，对知识库语料拟合 IDF 与词表）
6. 重建 collection（含 dense + sparse 两个向量字段）并批量写入

用法：
    uv run python scripts/index_knowledge.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# 先加载 .env，保证下方 shared 模块 import 时能读取连接配置
load_dotenv(ROOT / ".env")

from shared.hybrid_retrieval import (  # noqa: E402
    COLLECTION_NAME,
    DENSE_DIM,
    DENSE_FIELD,
    FILTERABLE_FIELDS,
    MIVILUS_TOKEN,
    MIVILUS_URI,
    SPARSE_FIELD,
    TEXT_FIELD,
    _csr_to_dict,
    build_bm25,
    make_client,
    save_bm25,
)
from shared.retrieval import make_text_encoder  # noqa: E402

KNOWLEDGE_DIR = ROOT / "knowledge"
DENSE_EMBEDDING_MODEL = "siliconflow/BAAI/bge-m3"

# 基于文档结构分块：以 H1~H4 标题为切分点，标题链写入块元数据
HEADERS_TO_SPLIT = [("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4")]
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80

MARKDOWN_SEPARATORS = [
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n```",
    "\n\n",
    "\n",
    " ",
    "",
]

# 每批写入的文档条数
INSERT_BATCH_SIZE = 50

FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
META_FIELD_PATTERN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$")


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """解析 Markdown 文件开头的 YAML frontmatter，返回 (元数据, 正文)。"""
    match = FRONTMATTER_PATTERN.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        field = META_FIELD_PATTERN.match(line.strip())
        if field:
            meta[field.group(1)] = field.group(2).strip().strip("'\"")
    return meta, text[match.end() :]


def load_documents() -> list[Document]:
    """扫描 knowledge/ 目录，加载并基于文档结构分块所有 Markdown 文档。"""
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    refine_splitter = RecursiveCharacterTextSplitter(
        separators=MARKDOWN_SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    documents: list[Document] = []
    files = sorted(KNOWLEDGE_DIR.rglob("*.md"))
    if not files:
        raise FileNotFoundError(f"在 {KNOWLEDGE_DIR} 下未找到任何 Markdown 文件")

    for path in files:
        text = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        if not body.strip():
            print(f"  跳过空文档: {path.relative_to(KNOWLEDGE_DIR)}")
            continue

        # 基于标题层级切分：每个块继承所在标题层级链（H1~H4）
        chunks = splitter.split_text(body)
        refined: list[Document] = []
        for chunk in chunks:
            # 超长块按 Markdown 分隔符二次切分，子块保留父块的元数据
            if len(chunk.page_content) > CHUNK_SIZE:
                refined.extend(refine_splitter.split_documents([chunk]))
            else:
                refined.append(chunk)

        for chunk in refined:
            chunk.metadata.update(
                {
                    "doc_title": meta.get("title", path.stem),
                    "category": meta.get("category", ""),
                    "url": meta.get("url", ""),
                    "source": str(path.relative_to(KNOWLEDGE_DIR)),
                }
            )
            documents.append(chunk)

        print(
            f"  {path.relative_to(KNOWLEDGE_DIR)}: {len(chunks)} 个结构块"
            f" -> {len(refined)} 个最终块"
        )

    return documents


def create_dual_vector_collection(client) -> None:
    """重建 collection：dense(FLOAT_VECTOR) + sparse(SPARSE_FLOAT_VECTOR) 双向量字段。"""
    from pymilvus import DataType
    from pymilvus.milvus_client.index import IndexParams

    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        print(f"已删除旧 collection: {COLLECTION_NAME}")

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name=TEXT_FIELD, datatype=DataType.VARCHAR, max_length=65535)
    for field in sorted(FILTERABLE_FIELDS - {"pk"}):
        schema.add_field(field_name=field, datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(
        field_name=DENSE_FIELD, datatype=DataType.FLOAT_VECTOR, dim=DENSE_DIM
    )
    schema.add_field(field_name=SPARSE_FIELD, datatype=DataType.SPARSE_FLOAT_VECTOR)

    index_params = IndexParams()
    # Zilliz Cloud serverless 使用 AUTOINDEX；metric 与字段类型匹配
    index_params.add_index(
        field_name=DENSE_FIELD, index_type="AUTOINDEX", metric_type="COSINE"
    )
    index_params.add_index(
        field_name=SPARSE_FIELD, index_type="AUTOINDEX", metric_type="IP"
    )
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(
        f"已创建 collection '{COLLECTION_NAME}': "
        f"{TEXT_FIELD}(varchar) + dense({DENSE_DIM}维) + sparse(BM25)"
    )


def main() -> None:
    """加载知识库、拟合 BM25、生成双路向量并重建 collection。"""
    if not MIVILUS_URI or not MIVILUS_TOKEN:
        raise ValueError("缺少 Mivilus 连接配置（Mivilus_URL / Mivilus_API_KEY）")

    print("== 1/4 加载并分块知识库文档（基于文档结构） ==")
    documents = load_documents()
    print(f"共生成 {len(documents)} 个文档块\n")

    print("== 2/4 拟合 BM25 稀疏模型（jieba 中英分词） ==")
    corpus = [doc.page_content for doc in documents]
    bm25 = build_bm25()
    bm25.fit(corpus)
    model_path = save_bm25(bm25)
    print(f"BM25 词表维度: {bm25.dim}，模型已保存: {model_path}\n")

    print("== 3/4 生成稠密向量（BGE-M3 via 硅基流动）与稀疏向量 ==")
    dense_encoder = make_text_encoder(DENSE_EMBEDDING_MODEL)
    # 校验稠密维度与 collection 声明一致
    probe = dense_encoder.embed_query("dimension probe")
    if len(probe) != DENSE_DIM:
        raise ValueError(f"稠密向量维度 {len(probe)} 与声明维度 {DENSE_DIM} 不一致")
    print(f"稠密向量维度: {len(probe)}")

    print("\n== 4/4 重建 collection 并批量写入 ==")
    client = make_client()
    create_dual_vector_collection(client)

    texts = [doc.page_content for doc in documents]
    # BM25 稀疏向量一次编码全部语料，逐行转为 Milvus 插入所需的 {token_id: weight}
    sparse_matrix = bm25.encode_documents(texts)
    sparse_vectors = [
        _csr_to_dict(sparse_matrix[i]) for i in range(sparse_matrix.shape[0])
    ]
    dense_vectors: list[list[float]] = []
    for i in range(0, len(texts), INSERT_BATCH_SIZE):
        batch = texts[i : i + INSERT_BATCH_SIZE]
        dense_vectors.extend(dense_encoder.embed_documents(batch))
        print(f"  稠密向量化 {min(i + INSERT_BATCH_SIZE, len(texts))}/{len(texts)}")

    for start in range(0, len(documents), INSERT_BATCH_SIZE):
        end = min(start + INSERT_BATCH_SIZE, len(documents))
        rows: list[dict] = []
        for i in range(start, end):
            doc = documents[i]
            row: dict = {TEXT_FIELD: doc.page_content}
            for field in sorted(FILTERABLE_FIELDS - {"pk"}):
                row[field] = str(doc.metadata.get(field, ""))
            row[DENSE_FIELD] = dense_vectors[i]
            row[SPARSE_FIELD] = sparse_vectors[i]
            rows.append(row)
        client.insert(collection_name=COLLECTION_NAME, data=rows)
        print(f"  已写入 {end}/{len(documents)} 条")

    print(
        f"\n完成：{len(documents)} 个文档块（稠密 BGE-M3 + 稀疏 BM25）"
        f"已写入 collection '{COLLECTION_NAME}'"
    )


if __name__ == "__main__":
    main()
