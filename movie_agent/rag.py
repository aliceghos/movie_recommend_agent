"""
RAG 检索层。

三段式检索链路，把「召回」和「精排」拆开：

    query ──┬─→ FAISS 向量召回 (top 20) ──┐
            └─→ BM25 关键词召回 (top 20) ─┴─→ 去重合并 ─→ LLM 精排 ─→ top k 带引用返回

为什么要混合召回：向量擅长语义近似（「视觉风格」命中「霓虹美学」），
BM25 擅长专名精确匹配（「布莱德·朗尼」这类音译片名、人名）。单靠任一路都会漏。

索引带 manifest：记录每个源文件的哈希与切分参数，语料或参数变了自动重建，
避免「索引文件存在就永不重建」导致的陈旧检索结果。
"""

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

# data/ 目录相对于本文件的路径
_DATA_DIR = Path(__file__).parent.parent / "data"
_KNOWLEDGE_DIR = _DATA_DIR / "knowledge"
_REVIEWS_DIR = _DATA_DIR / "reviews"
_BOOKS_DIR = _DATA_DIR / "books"
_INDEX_DIR = _DATA_DIR / "index"
_MANIFEST_PATH = _INDEX_DIR / "manifest.json"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# 召回阶段每路取多少候选，精排后收敛到 k
RECALL_K = 20

_LABELS = {
    "genre_knowledge": "类型知识",
    "movie_review": "影评",
    "film_books": "电影书",
}

# 全局单例（每个进程只加载一次）
_vectorstore: FAISS | None = None
_bm25: BM25Retriever | None = None


# ---------------------------------------------------------------------------
# 文档加载与切分
# ---------------------------------------------------------------------------

def _source_files() -> list[Path]:
    """参与索引的全部源文件，顺序稳定以便算 manifest。"""
    files = sorted(_KNOWLEDGE_DIR.glob("*.txt"))
    files += sorted(_REVIEWS_DIR.glob("*.txt"))
    if _BOOKS_DIR.exists():
        files += sorted(_BOOKS_DIR.glob("*.pdf"))
    return files


def _load_documents() -> list[Document]:
    """加载 knowledge/ reviews/ 的 .txt 与 books/ 的 .pdf，附加来源元数据。"""
    docs: list[Document] = []

    for txt_path in sorted(_KNOWLEDGE_DIR.glob("*.txt")):
        docs.append(Document(
            page_content=txt_path.read_text(encoding="utf-8"),
            metadata={"source": str(txt_path), "category": "genre_knowledge", "filename": txt_path.name},
        ))

    for txt_path in sorted(_REVIEWS_DIR.glob("*.txt")):
        docs.append(Document(
            page_content=txt_path.read_text(encoding="utf-8"),
            metadata={"source": str(txt_path), "category": "movie_review", "filename": txt_path.name},
        ))

    if _BOOKS_DIR.exists():
        for pdf_path in sorted(_BOOKS_DIR.glob("*.pdf")):
            extracted = 0
            for page_doc in PyPDFLoader(str(pdf_path)).load():
                cleaned = "\n".join(
                    line.strip() for line in page_doc.page_content.splitlines() if line.strip()
                )
                if not cleaned:
                    continue
                page_doc.page_content = cleaned
                page_doc.metadata.update({
                    "source": str(pdf_path),
                    "category": "film_books",
                    "filename": pdf_path.name,
                })
                docs.append(page_doc)
                extracted += 1
            if extracted == 0:
                # 扫描版 PDF 没有文字层，PyPDF 抽不出任何内容。静默跳过会让
                # 工具描述里承诺的书籍内容其实检索不到，所以显式报出来。
                print(f"[rag] {pdf_path.name}: no extractable text (scanned PDF?) — skipped")

    return docs


def _split(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    return splitter.split_documents(docs)


def _weights_are_cached() -> bool:
    """嵌入模型权重是否已在本地 HF 缓存里。"""
    from huggingface_hub import try_to_load_from_cache

    return isinstance(
        try_to_load_from_cache(EMBEDDING_MODEL, "config.json"), str
    )


def _get_embeddings() -> HuggingFaceEmbeddings:
    """本地嵌入模型。

    权重已缓存时强制离线加载：sentence-transformers 启动时会去拉
    ``adapter_config.json`` 等可选文件，这些请求不走 ``HF_ENDPOINT`` 镜像，
    在国内会卡在重试上（实测连续 5 次重试后仍失败）。首次运行无缓存时
    仍走在线下载，交由 ``HF_ENDPOINT`` 选镜像。
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"local_files_only": _weights_are_cached()},
    )


# ---------------------------------------------------------------------------
# Manifest —— 让索引在语料/参数变化后自动失效
# ---------------------------------------------------------------------------

def _file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    stat = path.stat()
    return {
        "path": str(path.relative_to(_DATA_DIR)),
        "size": stat.st_size,
        "sha256": digest,
    }


def compute_manifest() -> dict[str, Any]:
    """当前磁盘状态对应的 manifest。"""
    return {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "sources": [_file_fingerprint(p) for p in _source_files()],
    }


def _read_manifest() -> dict[str, Any] | None:
    if not _MANIFEST_PATH.exists():
        return None
    try:
        return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def index_is_stale() -> bool:
    """索引缺失、语料变动、或切分/嵌入参数变动时返回 True。"""
    if not (_INDEX_DIR / "index.faiss").exists():
        return True
    return _read_manifest() != compute_manifest()


# ---------------------------------------------------------------------------
# 索引构建与加载
# ---------------------------------------------------------------------------

def build_index(verbose: bool = False) -> FAISS:
    """切分 → 嵌入 → 落盘 FAISS + manifest。"""
    started = time.perf_counter()
    docs = _load_documents()
    chunks = _split(docs)

    if verbose:
        per_file: dict[str, int] = {}
        for c in chunks:
            per_file[c.metadata.get("filename", "?")] = per_file.get(c.metadata.get("filename", "?"), 0) + 1
        for name, count in per_file.items():
            print(f"  {name}: {count} chunks")

    vs = FAISS.from_documents(chunks, _get_embeddings())

    _INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(_INDEX_DIR))
    _MANIFEST_PATH.write_text(
        json.dumps(compute_manifest(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if verbose:
        print(f"  built {len(chunks)} chunks from {len(docs)} documents "
              f"in {time.perf_counter() - started:.1f}s")
    return vs


def get_vectorstore() -> FAISS:
    """返回向量库单例；manifest 不匹配时自动重建。"""
    global _vectorstore, _bm25
    if _vectorstore is None:
        if index_is_stale():
            reason = "missing" if not (_INDEX_DIR / "index.faiss").exists() else "stale (corpus or params changed)"
            print(f"[rag] index {reason} — rebuilding")
            _vectorstore = build_index()
            _bm25 = None
        else:
            _vectorstore = FAISS.load_local(
                str(_INDEX_DIR), _get_embeddings(),
                allow_dangerous_deserialization=True,
            )
    return _vectorstore


# ---------------------------------------------------------------------------
# BM25 关键词召回
# ---------------------------------------------------------------------------

_CJK = r"\u4e00-\u9fff"


def _tokenize(text: str) -> list[str]:
    """中英混排分词。

    ``BM25Retriever`` 默认按空格切词，对中文等于不分词 —— 整段变成一个 token，
    BM25 那一路就废了。这里对 CJK 取字 + 相邻二字组（近似词），
    对拉丁文取小写单词，不引入额外的分词器依赖。
    """
    tokens = [w.lower() for w in re.findall(r"[A-Za-z0-9']+", text)]
    cjk = re.findall(f"[{_CJK}]", text)
    tokens.extend(cjk)
    tokens.extend(a + b for a, b in zip(cjk, cjk[1:]))
    return tokens


def _all_chunks(vs: FAISS) -> list[Document]:
    """从 FAISS 的 docstore 取回全部 chunk，避免为了 BM25 再解析一遍源文件。"""
    store = getattr(vs.docstore, "_dict", None)
    if store:
        return list(store.values())
    return _split(_load_documents())  # 兜底：docstore 结构变化时回退到重新加载


def get_bm25() -> BM25Retriever:
    """返回 BM25 检索器单例，语料与向量库共用同一份 chunk。"""
    global _bm25
    if _bm25 is None:
        vs = get_vectorstore()
        _bm25 = BM25Retriever.from_documents(
            _all_chunks(vs), preprocess_func=_tokenize
        )
        _bm25.k = RECALL_K
    return _bm25


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------

def _doc_key(doc: Document) -> tuple[str, str]:
    return (doc.metadata.get("source", ""), doc.page_content[:200])


def _recall(query: str) -> list[Document]:
    """向量 + BM25 双路召回，按内容去重。向量路的顺序优先。"""
    candidates: list[Document] = []
    seen: set[tuple[str, str]] = set()

    for source in (_vector_recall, _bm25_recall):
        for doc in source(query):
            key = _doc_key(doc)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(doc)

    return candidates


def _vector_recall(query: str) -> list[Document]:
    return get_vectorstore().similarity_search(query, k=RECALL_K)


def _bm25_recall(query: str) -> list[Document]:
    try:
        return get_bm25().invoke(query)
    except Exception as exc:  # noqa: BLE001 - BM25 失败不该拖垮向量路
        print(f"[rag] bm25 recall skipped: {type(exc).__name__}: {exc}")
        return []


_RERANK_PROMPT = """You are ranking retrieved passages by relevance to a query.

Query: {query}

Candidates:
{candidates}

Return the numbers of the {k} most relevant candidates, most relevant first,
as a comma-separated list. Output nothing else. Example: 3,1,7,2"""


def _rerank(query: str, candidates: list[Document], k: int) -> list[Document]:
    """LLM 精排。任何失败都回落到召回原序，绝不让检索整体失败。"""
    if len(candidates) <= k:
        return candidates

    from movie_agent.llm import get_utility_llm

    listing = "\n".join(
        f"[{i}] ({_LABELS.get(d.metadata.get('category', ''), 'other')} · "
        f"{d.metadata.get('filename', '?')}) {d.page_content[:300]}"
        for i, d in enumerate(candidates, 1)
    )

    try:
        reply = get_utility_llm().invoke([
            HumanMessage(content=_RERANK_PROMPT.format(query=query, candidates=listing, k=k))
        ])
        picked = [int(n) for n in re.findall(r"\d+", str(reply.content))]
        ordered = [candidates[i - 1] for i in picked if 1 <= i <= len(candidates)]
        # 去重后仍不足 k 条时用召回原序补齐
        result: list[Document] = []
        seen: set[tuple[str, str]] = set()
        for doc in ordered + candidates:
            key = _doc_key(doc)
            if key not in seen:
                seen.add(key)
                result.append(doc)
            if len(result) == k:
                break
        return result
    except Exception as exc:  # noqa: BLE001 - 精排是增强项，失败即降级
        print(f"[rag] rerank skipped: {type(exc).__name__}: {exc}")
        return candidates[:k]


def _citation(doc: Document) -> str:
    """来源标注，供模型在回答里引用。PDF 会带上页码。"""
    label = _LABELS.get(doc.metadata.get("category", ""), "资料")
    parts = [label, doc.metadata.get("filename", "unknown")]
    page = doc.metadata.get("page")
    if page is not None:
        parts.append(f"第 {int(page) + 1} 页")
    return "【" + " · ".join(parts) + "】"


def search_knowledge(query: str, k: int = 4, *, rerank: bool = True) -> str:
    """在本地知识库中检索与 query 最相关的段落。

    Args:
        query: 检索问题或关键词
        k: 最终返回的段落数量
        rerank: 是否启用 LLM 精排。关掉可省一次 LLM 调用，用于快速路径与测试对照。

    Returns:
        格式化的检索结果，每段带来源标注（类型 · 文件名 · 页码）。
    """
    candidates = _recall(query)
    if not candidates:
        return "No relevant information found in the local knowledge base."

    results = _rerank(query, candidates, k) if rerank else candidates[:k]

    lines = [f"Retrieved {len(results)} passage(s) from the local knowledge base:\n"]
    for i, doc in enumerate(results, 1):
        lines.append(f"[{i}] {_citation(doc)}")
        lines.append(doc.page_content.strip())
        lines.append("")
    lines.append("When you use these passages, cite the source label in your answer.")
    return "\n".join(lines)
