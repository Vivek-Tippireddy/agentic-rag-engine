import os
import sys
import glob
from typing import List, Dict, Any
from pathlib import Path
from pypdf import PdfReader

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.documents import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

import config

class VectorDBRetriever:
    """Vector Store Retriever managing ChromaDB index over corporate annual reports."""
    
    def __init__(self, data_dir: str = None, persist_dir: str = None):
        self.data_dir = data_dir or config.DEFAULT_DATA_DIR
        self.persist_dir = persist_dir or config.CHROMA_PERSIST_DIR
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            print(f"[VectorDB] HuggingFaceEmbeddings unavailable ({e}), using GoogleGenerativeAIEmbeddings fallback.", flush=True)
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=config.GEMINI_API_KEY)
        self.vector_store = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Loads existing ChromaDB index or builds a new one from the data directory."""
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            print(f"[VectorDB] Loading existing Chroma vector store from {self.persist_dir}", flush=True)
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
        else:
            print(f"[VectorDB] Building new vector index from {self.data_dir}...", flush=True)
            self.build_index()

    def build_index(self):
        """Scans PDFs & TXT files in data directory, chunks them, and creates Chroma index."""
        documents = []
        data_path = Path(self.data_dir)

        if not data_path.exists():
            print(f"[VectorDB Warning] Data directory {self.data_dir} does not exist.", flush=True)
            return

        file_paths = list(data_path.rglob("*.pdf")) + list(data_path.rglob("*.txt"))
        print(f"[VectorDB] Found {len(file_paths)} files to process.", flush=True)

        for file_path in file_paths:
            company = file_path.parent.name
            try:
                if file_path.suffix.lower() == ".pdf":
                    reader = PdfReader(str(file_path))
                    max_p = min(len(reader.pages), 15)
                    for page_num in range(max_p):
                        text = reader.pages[page_num].extract_text() or ""
                        if text.strip():
                            doc = Document(
                                page_content=text,
                                metadata={
                                    "company": company,
                                    "file_name": file_path.name,
                                    "page": page_num + 1,
                                    "source_path": str(file_path)
                                }
                            )
                            documents.append(doc)
                    print(f"  - Loaded {file_path.name} ({max_p} pages processed)", flush=True)
                else:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    doc = Document(
                        page_content=text,
                        metadata={
                            "company": company,
                            "file_name": file_path.name,
                            "page": 1,
                            "source_path": str(file_path)
                        }
                    )
                    documents.append(doc)
                    print(f"  - Loaded {file_path.name} (text file)", flush=True)

            except Exception as e:
                print(f"  - [Error] Failed loading {file_path.name}: {e}", flush=True)

        if not documents:
            print("[VectorDB Warning] No valid documents found to index.", flush=True)
            return

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"[VectorDB] Created {len(chunks)} text chunks across all documents.", flush=True)

        # Build & persist Chroma database
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        print(f"[VectorDB] Successfully saved vector index to {self.persist_dir}", flush=True)

    def query(self, query_text: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Retrieves top_k relevant document chunks matching query_text."""
        if not self.vector_store:
            return [{"source": "VectorDB", "content": "Vector DB index is empty.", "score": 0.0}]

        try:
            results_with_scores = self.vector_store.similarity_search_with_score(query_text, k=top_k)
            retrieved = []
            for doc, score in results_with_scores:
                retrieved.append({
                    "source": "VectorDB",
                    "content": doc.page_content,
                    "company": doc.metadata.get("company", "Unknown"),
                    "file_name": doc.metadata.get("file_name", "Unknown"),
                    "page": doc.metadata.get("page", 0),
                    "score": round(float(score), 4)
                })
            return retrieved
        except Exception as e:
            print(f"[VectorDB Query Error] {e}", flush=True)
            return [{"source": "VectorDB", "content": f"Query error: {e}", "score": 0.0}]


if __name__ == "__main__":
    db = VectorDBRetriever()
    res = db.query("What is Tata Motors EV strategy and revenue?")
    for idx, r in enumerate(res, 1):
        print(f"\n--- Chunk {idx} ({r['company']} - {r['file_name']}) ---", flush=True)
        print(r['content'][:250] + "...", flush=True)
