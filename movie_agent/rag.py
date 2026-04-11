"""
RAG 模块：加载 data/ 目录下的影评与电影类型知识文档，
使用 LangChain + FAISS 构建本地向量检索，供 Agent 工具调用。
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# data/ 目录相对于本文件的路径
_DATA_DIR = Path(__file__).parent.parent / "data"
_KNOWLEDGE_DIR = _DATA_DIR / "knowledge"
_REVIEWS_DIR = _DATA_DIR / "reviews"
_BOOKS_DIR = _DATA_DIR / "books"
_INDEX_DIR = _DATA_DIR / "index"

# 全局向量库单例（每次 session 只加载一次）
_vectorstore: FAISS | None = None


def _load_documents() -> list[Document]:
    """加载 knowledge/ 和 reviews/ 下的所有 .txt 文件，附加来源元数据。"""
    docs: list[Document] = []

    for txt_path in sorted(_KNOWLEDGE_DIR.glob("*.txt")):
        text = txt_path.read_text(encoding="utf-8")
        docs.append(Document(
            page_content=text,
            metadata={"source": str(txt_path), "category": "genre_knowledge", "filename": txt_path.name},
        ))

    for txt_path in sorted(_REVIEWS_DIR.glob("*.txt")):
        text = txt_path.read_text(encoding="utf-8")
        docs.append(Document(
            page_content=text,
            metadata={"source": str(txt_path), "category": "movie_review", "filename": txt_path.name},
        ))

    if _BOOKS_DIR.exists():
        for pdf_path in sorted(_BOOKS_DIR.glob("*.pdf")):
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            for page_doc in pages:
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

    return docs


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )


def _build_vectorstore() -> FAISS:
    """将文档切分后嵌入，构建 FAISS 向量库并持久化到磁盘。"""
    docs = _load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = _get_embeddings()
    vs = FAISS.from_documents(chunks, embeddings)

    _INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(_INDEX_DIR))

    return vs


def get_vectorstore() -> FAISS:
    """返回全局向量库单例。优先从磁盘加载，不存在时构建并保存。"""
    global _vectorstore
    if _vectorstore is None:
        if (_INDEX_DIR / "index.faiss").exists():
            _vectorstore = FAISS.load_local(
                str(_INDEX_DIR), _get_embeddings(),
                allow_dangerous_deserialization=True,
            )
        else:
            _vectorstore = _build_vectorstore()
    return _vectorstore


def search_knowledge(query: str, k: int = 4) -> str:
    """
    在本地知识库（影评 + 类型知识）中检索与 query 最相关的段落。

    Args:
        query: 检索问题或关键词
        k: 返回的最相关段落数量

    Returns:
        格式化的检索结果字符串，含来源文件名和段落内容。
    """
    vs = get_vectorstore()
    results = vs.similarity_search(query, k=k)

    if not results:
        return "No relevant information found in the local knowledge base."

    lines = [f"Retrieved {len(results)} relevant passage(s) from local knowledge base:\n"]
    for i, doc in enumerate(results, 1):
        filename = doc.metadata.get("filename", "unknown")
        category = doc.metadata.get("category", "")
        label_map = {
            "genre_knowledge": "Genre Knowledge",
            "movie_review": "Movie Review",
            "film_books": "Film Book",
        }
        label = label_map.get(category, category)
        lines.append(f"[{i}] [{label}] {filename}")
        lines.append(doc.page_content.strip())
        lines.append("")

    return "\n".join(lines)
