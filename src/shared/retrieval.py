"""文本编码器（embedding）构造函数。

本模块负责按 provider 创建稠密向量编码器；检索链路（稠密 + BM25 混合检索、
RRF 融合）由 shared.hybrid_retrieval 提供。
"""

import os
from typing import Any, Optional

from langchain_core.embeddings import Embeddings

## 编码器构造函数


class SiliconFlowEmbeddings(Embeddings):
    """基于硅基流动 OpenAI 兼容 /v1/embeddings 接口的 BGE-M3 稠密向量封装。

    硅基流动（SiliconFlow）只返回稠密向量，稀疏向量由 BM25 另行生成。
    """

    # 单次请求的最大文本条数（超出分批）
    _BATCH_SIZE = 32

    def __init__(self, model: str, dimension: Optional[int] = None) -> None:
        """初始化编码器。

        Args:
            model: 硅基流动上的模型名，例如 ``BAAI/bge-m3``。
            dimension: 可选的输出维度，传给接口的 ``dimensions`` 参数。
        """
        self.model = model
        self.dimension = dimension
        base_url = os.environ.get(
            "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
        ).rstrip("/")
        self._url = f"{base_url}/embeddings"
        self._api_key = os.environ["SiliconEmbedding_API_KEY"]

    def _call(self, texts: list[str]) -> list[list[float]]:
        import httpx

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        embeddings: list[list[float]] = []
        with httpx.Client(timeout=120) as client:
            for i in range(0, len(texts), self._BATCH_SIZE):
                batch = texts[i : i + self._BATCH_SIZE]
                payload: dict[str, Any] = {"model": self.model, "input": batch}
                if self.dimension:
                    payload["dimensions"] = self.dimension
                resp = client.post(self._url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()["data"]
                # data 与输入顺序一致（OpenAI 兼容格式按 index 排序）
                data.sort(key=lambda d: d["index"])
                embeddings.extend(item["embedding"] for item in data)
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量编码文档文本，返回与输入等长的向量列表。"""
        return self._call(texts)

    def embed_query(self, text: str) -> list[float]:
        """编码单条查询文本，返回一个向量。"""
        return self._call([text])[0]


def make_text_encoder(model: str) -> Embeddings:
    """连接到配置好的文本编码器。"""
    provider, model = model.split("/", maxsplit=1)
    match provider:
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model)
        case "cohere":
            from langchain_cohere import CohereEmbeddings

            return CohereEmbeddings(model=model)  # type: ignore
        case "siliconflow":
            # BGE-M3 等模型走硅基流动 OpenAI 兼容接口（仅稠密向量）
            return SiliconFlowEmbeddings(model=model)
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")
