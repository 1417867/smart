import re
from typing import List, Dict

class Citation:
    def __init__(self, index: int, source_type: str, document_title: str,
                 chunk_index: int, chunk_id: str = '', authors: List = None,
                 year: int = None, bibtex: str = '', verified: bool = False):
        self.index = index
        self.source_type = source_type
        self.document_title = document_title
        self.chunk_index = chunk_index
        self.chunk_id = chunk_id
        self.authors = authors or []
        self.year = year
        self.bibtex = bibtex
        self.verified = verified

class CitationEngine:
    CITATION_PATTERN = re.compile(r'\[Source:\s*(.+?),\s*Chunk\s*(\d+)\]')
    
    @classmethod
    def parse_citations(cls, text: str) -> List[Citation]:
        citations = []
        seen = set()
        
        for match in cls.CITATION_PATTERN.finditer(text):
            doc_title = match.group(1).strip()
            chunk_idx = int(match.group(2))
            
            key = (doc_title, chunk_idx)
            if key not in seen:
                seen.add(key)
                citations.append(Citation(
                    index=len(citations) + 1,
                    source_type='unknown',
                    document_title=doc_title,
                    chunk_index=chunk_idx
                ))
        
        return citations
    
    @classmethod
    def verify_citations(cls, citations: List[Citation], sources: List) -> List[Citation]:
        for citation in citations:
            for source in sources:
                if (citation.document_title in source.document_title or
                    source.document_title in citation.document_title):
                    citation.verified = True
                    citation.source_type = source.metadata.get('source_type', 'unknown')
                    citation.chunk_id = source.chunk_id
                    break
        return citations
    
    @classmethod
    def format_bibliography(cls, citations: List[Citation]) -> str:
        if not citations:
            return ""
        
        lines = ["## 📚 参考文献"]
        for citation in citations:
            authors = ", ".join(citation.authors) if citation.authors else "匿名作者"
            year = f"({citation.year})" if citation.year else ""
            source_type = cls._get_source_type_label(citation.source_type)
            
            line = f"[{citation.index}] {authors} {year}. 《{citation.document_title}》"
            if citation.chunk_index:
                line += f", Chunk {citation.chunk_index}"
            line += f" ({source_type})"
            lines.append(line)
        
        return "\n".join(lines)
    
    @classmethod
    def _get_source_type_label(cls, source_type: str) -> str:
        labels = {
            'textbooks': '教材',
            'papers': '论文',
            'docs': '官方文档',
            'competitions': '竞赛方案',
            'unknown': '未知来源'
        }
        return labels.get(source_type, source_type)
