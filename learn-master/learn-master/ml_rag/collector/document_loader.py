import os
import glob
from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from config.settings import settings

class DocumentLoader:
    SUPPORTED_EXTENSIONS = ['.txt', '.md', '.py', '.json']
    
    @classmethod
    def load_from_dir(cls, dir_path: str, source_type: str) -> List[Document]:
        documents = []
        if not os.path.exists(dir_path):
            return documents
        
        for ext in cls.SUPPORTED_EXTENSIONS:
            pattern = os.path.join(dir_path, f'**/*{ext}')
            files = glob.glob(pattern, recursive=True)
            
            for file_path in files:
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    
                    for doc in docs:
                        doc.metadata['source_type'] = source_type
                        doc.metadata['file_path'] = file_path
                        doc.metadata['file_name'] = os.path.basename(file_path)
                        documents.append(doc)
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")
        
        return documents
    
    @classmethod
    def load_all_kb(cls) -> Dict[str, List[Document]]:
        kb_docs = {}
        
        kb_config = [
            ('textbooks', settings.textbooks_dir),
            ('papers', settings.papers_dir),
            ('docs', settings.docs_dir),
            ('competitions', settings.competitions_dir),
        ]
        
        for kb_name, kb_dir in kb_config:
            docs = cls.load_from_dir(kb_dir, kb_name)
            kb_docs[kb_name] = docs
            print(f"Loaded {len(docs)} documents from {kb_name}")
        
        return kb_docs
    
    @classmethod
    def load_from_text(cls, text: str, source_type: str, title: str) -> List[Document]:
        return [Document(
            page_content=text,
            metadata={'source_type': source_type, 'title': title}
        )]
