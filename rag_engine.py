"""
RAG Engine - Document ingestion, embedding, retrieval, and LLM querying.
"""

import os
import glob
import requests
from typing import Optional
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS



Embeddings_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "embedding_model",
    "all-MiniLM-L6-v2"
)

class RAGEngine:
    def __init__(
        self,
        data_dir: str = "data",
        db_dir: str = "vector_db",
        model_path: Optional[str] = None
    ):
        self.data_dir = data_dir
        self.db_dir = db_dir
        self.vector_db: Optional[FAISS] = None

        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(db_dir, exist_ok=True)

        # Use local model path
        resolved_path = model_path or Embeddings_MODEL_PATH

        if not os.path.exists(resolved_path):
            raise FileNotFoundError(
                f"Embedding model not found at: {resolved_path}"
            )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=resolved_path,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    def ingest_docs(self) -> int:
        """Load, split, and index documents."""
        documents = []

        for pdf_path in glob.glob(f"{self.data_dir}/**/*.pdf", recursive=True):
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

        for ext in [".txt", ".md"]:
            loader = DirectoryLoader(
                self.data_dir,
                glob=f"**/*{ext}",
                loader_cls=TextLoader,
                loader_kwargs={"autodetect_encoding": True}
            )
            documents.extend(loader.load())

        if not documents:
            raise ValueError("No documents found. Supported: .pdf, .txt, .md")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)

        self.vector_db = FAISS.from_documents(chunks, self.embeddings)
        self.vector_db.save_local(self.db_dir)
        return len(chunks)

    def load_db(self) -> bool:
        """Load existing FAISS index from disk."""
        index_path = os.path.join(self.db_dir, "index.faiss")
        if os.path.exists(index_path):
            self.vector_db = FAISS.load_local(
                self.db_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return True
        return False

    def has_db(self) -> bool:
        """Check if a FAISS index exists on disk."""
        return os.path.exists(os.path.join(self.db_dir, "index.faiss"))

    def retrieve(self, query: str, k: int = 3) -> Optional[str]:
        """Retrieve top-k relevant chunks."""
        if not self.vector_db:
            raise RuntimeError("Vector DB not loaded.")

        docs = self.vector_db.similarity_search(query, k=k)
        if not docs:
            return None

        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def query_llm(
        self,
        query: str,
        context: str,
        api_url: str = "http://llama:8080/v1/chat/completions"
    ) -> str:
        """Send context + query to LLM server."""
        if not context:
            return "No relevant context found in documents."

        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        # Detect API style from URL
        if any(p in api_url for p in ["/chat/completions", "/api/chat"]):
            return self._query_chat_style(prompt, api_url)
        else:
            return self._query_completion_style(prompt, api_url)

    def _query_chat_style(self, prompt: str, api_url: str) -> str:
        """Query via chat completions endpoint."""
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a technical assistant. Use the provided context to answer."
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1
        }

        try:
            response = requests.post(api_url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return (
                data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "No content returned.")
            )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to {api_url}")
        except requests.exceptions.Timeout:
            raise TimeoutError("LLM request timed out.")

    def _query_completion_style(self, prompt: str, api_url: str) -> str:
        """Query via native completion endpoint."""
        payload = {
            "prompt": prompt,
            "temperature": 0.1,
            "n_predict": 512,
            "stop": ["Question:", "\n\n"]
        }

        try:
            response = requests.post(api_url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("content", "No content returned.")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to {api_url}")
        except requests.exceptions.Timeout:
            raise TimeoutError("LLM request timed out.")