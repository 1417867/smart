import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ML_RAG_ENABLED = os.getenv("ML_RAG_ENABLED", "true").lower() == "true"
    ML_KB_DIR = os.getenv("ML_KB_DIR", "data")
    ML_CHUNK_SIZE = int(os.getenv("ML_CHUNK_SIZE", "1024"))
    ML_CHUNK_OVERLAP = int(os.getenv("ML_CHUNK_OVERLAP", "200"))
    ML_RETRIEVAL_TOP_K = int(os.getenv("ML_RETRIEVAL_TOP_K", "10"))
    ML_RERANK_ENABLED = os.getenv("ML_RERANK_ENABLED", "true").lower() == "true"
    ML_QUERY_REWRITE_ENABLED = os.getenv("ML_QUERY_REWRITE_ENABLED", "true").lower() == "true"
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "faiss_index")
    
    @property
    def textbooks_dir(self):
        return os.path.join(self.ML_KB_DIR, "textbooks")
    
    @property
    def papers_dir(self):
        return os.path.join(self.ML_KB_DIR, "papers")
    
    @property
    def docs_dir(self):
        return os.path.join(self.ML_KB_DIR, "docs")
    
    @property
    def competitions_dir(self):
        return os.path.join(self.ML_KB_DIR, "competitions")

settings = Settings()
