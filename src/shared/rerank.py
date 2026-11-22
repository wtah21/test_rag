"""本地 cross-encoder 重排（bge-reranker-v2-m3）。

优先用 CUDA 上的 PyTorch fp16 推理；无可用 GPU 时回退到 ONNX int8（CPU）。
模型首次调用时懒加载并复用；推理失败时回退 RRF 原序，保证检索链路不中断。
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from langchain_core.documents import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

# PyTorch 目录（safetensors 原模型，GPU 路径使用）
RERANK_MODEL_PATH = Path(
    os.environ.get("RERANK_MODEL_PATH", r"E:\trae\model\bge-reranker-v2-m3")
)
# ONNX int8 目录（CPU 回退路径使用）
RERANK_ONNX_PATH = Path(
    os.environ.get("RERANK_ONNX_PATH", r"E:\trae\model\bge-reranker-v2-m3-onnx-int8")
)
# auto: 有 CUDA 用 GPU，否则用 ONNX CPU；可强制 cuda / cpu
RERANK_DEVICE = os.environ.get("RERANK_DEVICE", "auto").lower()
RERANK_MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "512"))
RERANK_BATCH_SIZE = int(os.environ.get("RERANK_BATCH_SIZE", "16"))

_load_lock = threading.Lock()
_tokenizer: Any = None
_model: Any = None
_session: Any = None
_device = "cpu"


def _use_cuda() -> bool:
    if RERANK_DEVICE == "cuda":
        return True
    if RERANK_DEVICE == "cpu":
        return False
    return torch.cuda.is_available()


def _get_torch_model() -> tuple[Any, Any, str]:
    """懒加载 PyTorch 模型：GPU 用 fp16，CPU 用 fp32。"""
    global _tokenizer, _model, _device
    with _load_lock:
        if _model is None:
            if not (RERANK_MODEL_PATH / "config.json").exists():
                raise FileNotFoundError(f"未找到重排模型: {RERANK_MODEL_PATH}")
            _tokenizer = AutoTokenizer.from_pretrained(str(RERANK_MODEL_PATH))
            model = AutoModelForSequenceClassification.from_pretrained(
                str(RERANK_MODEL_PATH)
            )
            _device = "cuda" if _use_cuda() else "cpu"
            model = (model.half() if _device == "cuda" else model.float()).to(_device)
            model.eval()
            _model = model
            logger.info("已加载本地重排模型（PyTorch %s）", _device)
    return _tokenizer, _model, _device


def _get_onnx_model() -> tuple[Any, Any]:
    """懒加载 ONNX int8 会话（CPU 回退）。"""
    global _tokenizer, _session
    with _load_lock:
        if _session is None:
            model_file = RERANK_ONNX_PATH / "model.onnx"
            if not model_file.exists():
                raise FileNotFoundError(f"未找到 ONNX 重排模型: {model_file}")
            import onnxruntime as ort

            _tokenizer = AutoTokenizer.from_pretrained(str(RERANK_ONNX_PATH))
            _session = ort.InferenceSession(
                str(model_file), providers=["CPUExecutionProvider"]
            )
            logger.info("已加载本地重排模型（ONNX int8 CPU）")
    return _tokenizer, _session


def _score_torch(query: str, texts: list[str]) -> list[float]:
    tokenizer, model, device = _get_torch_model()
    scores: list[float] = []
    for start in range(0, len(texts), RERANK_BATCH_SIZE):
        batch = texts[start : start + RERANK_BATCH_SIZE]
        encoded = tokenizer(
            [[query, text] for text in batch],
            padding=True,
            truncation=True,
            max_length=RERANK_MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits.view(-1).float()
        scores.extend(torch.sigmoid(logits).cpu().tolist())
    return scores


def _score_onnx(query: str, texts: list[str]) -> list[float]:
    tokenizer, session = _get_onnx_model()
    input_names = {item.name for item in session.get_inputs()}
    scores: list[float] = []
    for start in range(0, len(texts), RERANK_BATCH_SIZE):
        batch = texts[start : start + RERANK_BATCH_SIZE]
        encoded = tokenizer(
            [[query, text] for text in batch],
            padding=True,
            truncation=True,
            max_length=RERANK_MAX_LENGTH,
            return_tensors="np",
        )
        feed: dict[str, Any] = {
            "input_ids": np.asarray(encoded["input_ids"], dtype=np.int64),
            "attention_mask": np.asarray(encoded["attention_mask"], dtype=np.int64),
        }
        if "token_type_ids" in input_names and "token_type_ids" in encoded:
            feed["token_type_ids"] = np.asarray(
                encoded["token_type_ids"], dtype=np.int64
            )
        logits = np.asarray(session.run(None, feed)[0]).reshape(-1)
        scores.extend((1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))).tolist())
    return scores


def _score_documents(query: str, texts: list[str]) -> list[float]:
    """批量计算 (query, 文档) 相关性分数，sigmoid 归一化到 0~1。"""
    if _use_cuda():
        return _score_torch(query, texts)
    return _score_onnx(query, texts)


def rerank_documents(
    query: str, documents: list[Document], top_n: int = 4
) -> list[Document]:
    """用本地 cross-encoder 重排候选文档，返回前 top_n 条。

    失败（模型缺失、推理异常、返回长度不符）时回退为原顺序取前 top_n 条，
    保证检索链路不因重排失败而中断。

    Args:
        query: 用于计算相关性的查询文本。
        documents: RRF 融合后的候选文档（按 rrf_score 降序）。
        top_n: 重排后保留的文档数。

    Returns:
        按相关性降序排列的前 top_n 条文档，元数据中写入 rerank_score；
        失败时返回原序前 top_n 条。
    """
    if not documents:
        return []
    if not query:
        return documents[:top_n]

    try:
        scores = _score_documents(query, [doc.page_content for doc in documents])
        if len(scores) != len(documents):
            raise ValueError("重排分数数量与候选文档数不一致")
        ranked = sorted(zip(scores, documents), key=lambda pair: pair[0], reverse=True)
        results: list[Document] = []
        for score, doc in ranked[:top_n]:
            metadata = dict(doc.metadata or {})
            metadata["rerank_score"] = round(float(score), 4)
            results.append(Document(page_content=doc.page_content, metadata=metadata))
        return results
    except Exception:  # noqa: BLE001
        logger.warning("本地重排失败，回退 RRF 原序取前 %d 条", top_n, exc_info=True)
        return documents[:top_n]
