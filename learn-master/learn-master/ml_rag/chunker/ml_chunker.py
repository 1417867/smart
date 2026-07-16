import re
from typing import List, Dict, Optional, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.settings import settings
from retrieval.ml_terms import ML_TERMS

class MlChunkConfig:
    def __init__(self, chunk_size=None, chunk_overlap=None, 
                 protect_formulas=True, protect_code_blocks=True,
                 protect_definitions=True, term_density_threshold=0.05):
        self.chunk_size = chunk_size or settings.ML_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.ML_CHUNK_OVERLAP
        self.protect_formulas = protect_formulas
        self.protect_code_blocks = protect_code_blocks
        self.protect_definitions = protect_definitions
        self.term_density_threshold = term_density_threshold

class MlChunk:
    def __init__(self, content: str, chunk_index: int, metadata: Dict = None):
        self.content = content
        self.chunk_index = chunk_index
        self.metadata = metadata or {}
        self.formula_count = 0
        self.code_snippet_count = 0
        self.ml_term_density = 0.0
        self.definition_blocks = []
        self.source_type = metadata.get("source_type", "unknown")
        self.difficulty = metadata.get("difficulty")
        
    def analyze(self):
        self.formula_count = self._count_formulas()
        self.code_snippet_count = self._count_code_snippets()
        self.ml_term_density = self._calculate_term_density()
        self.definition_blocks = self._extract_definitions()
        
    def _count_formulas(self) -> int:
        block_formulas = len(re.findall(r'\$\$[\s\S]*?\$\$', self.content))
        inline_formulas = len(re.findall(r'(?<!\$)\$[^$\n]+\$(?!\$)', self.content))
        return block_formulas + inline_formulas
    
    def _count_code_snippets(self) -> int:
        return len(re.findall(r'```[\s\S]*?```', self.content))
    
    def _calculate_term_density(self) -> float:
        words = re.findall(r'\w+', self.content.lower())
        if not words:
            return 0.0
        term_count = 0
        for term in ML_TERMS.keys():
            if term.lower() in self.content.lower():
                term_count += 1
        return term_count / len(words)
    
    def _extract_definitions(self) -> List[str]:
        patterns = [
            r'定义[：:]',
            r'定义\s+\d+[：:]',
            r'定理[：:]',
            r'定理\s+\d+[：:]',
            r'引理[：:]',
            r'性质[：:]',
            r'公理[：:]',
            r'法则[：:]',
        ]
        definitions = []
        for pattern in patterns:
            matches = re.findall(pattern + r'\s*([^\n]{1,200})', self.content)
            definitions.extend(matches)
        return definitions

class MlChunker:
    def __init__(self, config: MlChunkConfig = None):
        self.config = config or MlChunkConfig()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
    
    def split_text(self, text: str, metadata: Dict = None) -> List[MlChunk]:
        raw_chunks = self.splitter.split_text(text)
        
        protected_chunks = []
        if self.config.protect_formulas or self.config.protect_code_blocks:
            protected_chunks = self._protect_special_blocks(raw_chunks)
        else:
            protected_chunks = raw_chunks
        
        ml_chunks = []
        for i, chunk_content in enumerate(protected_chunks):
            ml_chunk = MlChunk(chunk_content, i, metadata)
            ml_chunk.analyze()
            
            if ml_chunk.ml_term_density >= self.config.term_density_threshold:
                ml_chunks.append(ml_chunk)
        
        return ml_chunks
    
    def split_documents(self, documents: List[Document]) -> List[MlChunk]:
        all_chunks = []
        for doc in documents:
            chunks = self.split_text(doc.page_content, doc.metadata)
            all_chunks.extend(chunks)
        return all_chunks
    
    def _protect_special_blocks(self, chunks: List[str]) -> List[str]:
        merged = []
        i = 0
        while i < len(chunks):
            current = chunks[i]
            
            has_open_formula = self._has_open_formula(current)
            has_open_code = self._has_open_code(current)
            
            if has_open_formula or has_open_code:
                j = i + 1
                while j < len(chunks):
                    current += chunks[j]
                    if has_open_formula and not self._has_open_formula(current):
                        break
                    if has_open_code and not self._has_open_code(current):
                        break
                    j += 1
                i = j
            else:
                i += 1
            
            merged.append(current)
        
        return merged
    
    def _has_open_formula(self, text: str) -> bool:
        block_count = text.count("$$")
        inline_count = text.count("$") - 2 * block_count
        return (block_count % 2 == 1) or (inline_count % 2 == 1)
    
    def _has_open_code(self, text: str) -> bool:
        return text.count("```") % 2 == 1
