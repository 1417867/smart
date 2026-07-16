import os
import pickle
from typing import Dict
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from config.settings import settings
from collector.document_loader import DocumentLoader
from chunker.ml_chunker import MlChunker

class RAGIndexer:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY)
        self.chunker = MlChunker()
        self.index_dir = settings.FAISS_INDEX_DIR
        
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
    
    def build_index(self) -> Dict[str, FAISS]:
        print("Loading documents from all knowledge bases...")
        kb_docs = DocumentLoader.load_all_kb()
        
        vector_stores = {}
        
        for kb_name, docs in kb_docs.items():
            if not docs:
                print(f"No documents found for {kb_name}, skipping...")
                continue
            
            print(f"Processing {len(docs)} documents for {kb_name}...")
            
            all_chunks = []
            for i, doc in enumerate(docs):
                chunks = self.chunker.split_text(doc.page_content, doc.metadata)
                for j, chunk in enumerate(chunks):
                    all_chunks.append(Document(
                        page_content=chunk.content,
                        metadata={
                            'chunk_id': f"{kb_name}_{i}_{j}",
                            'document_id': str(i),
                            'chunk_index': j,
                            'source_type': chunk.source_type,
                            'title': doc.metadata.get('title', doc.metadata.get('file_name', '')),
                            'file_name': doc.metadata.get('file_name', ''),
                            'difficulty': chunk.difficulty,
                            'ml_term_density': chunk.ml_term_density,
                            'formula_count': chunk.formula_count,
                            'code_snippet_count': chunk.code_snippet_count
                        }
                    ))
            
            print(f"Creating FAISS index for {kb_name} ({len(all_chunks)} chunks)...")
            vector_store = FAISS.from_documents(all_chunks, self.embeddings)
            
            index_path = os.path.join(self.index_dir, f"{kb_name}_faiss_index")
            vector_store.save_local(index_path)
            print(f"Saved index to {index_path}")
            
            vector_stores[kb_name] = vector_store
        
        self._save_vector_stores_info(vector_stores)
        print("Index building complete!")
        
        return vector_stores
    
    def load_index(self) -> Dict[str, FAISS]:
        vector_stores = {}
        
        for kb_name in ['textbooks', 'papers', 'docs', 'competitions']:
            index_path = os.path.join(self.index_dir, f"{kb_name}_faiss_index")
            
            if os.path.exists(index_path):
                print(f"Loading FAISS index for {kb_name}...")
                vector_store = FAISS.load_local(
                    index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                vector_stores[kb_name] = vector_store
                print(f"Loaded {kb_name} index")
            else:
                print(f"No index found for {kb_name}")
        
        return vector_stores
    
    def _save_vector_stores_info(self, vector_stores: Dict[str, FAISS]):
        info = {kb: vs.index.ntotal for kb, vs in vector_stores.items()}
        info_path = os.path.join(self.index_dir, "index_info.pkl")
        with open(info_path, 'wb') as f:
            pickle.dump(info, f)
    
    def get_index_info(self) -> Dict[str, int]:
        info_path = os.path.join(self.index_dir, "index_info.pkl")
        if os.path.exists(info_path):
            with open(info_path, 'rb') as f:
                return pickle.load(f)
        return {}

if __name__ == "__main__":
    indexer = RAGIndexer()
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'build':
        indexer.build_index()
    else:
        vector_stores = indexer.load_index()
        info = indexer.get_index_info()
        print("Loaded vector stores:")
        for kb, count in info.items():
            print(f"  {kb}: {count} vectors")
