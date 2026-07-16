from typing import List, Dict, Optional
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from config.settings import settings
from .query_rewriter import QueryRewriteResult

class RAGSource:
    def __init__(self, chunk_id: str, document_id: str, document_title: str,
                 content: str, score: float, chunk_index: int, metadata: Dict = None):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.document_title = document_title
        self.content = content
        self.score = score
        self.chunk_index = chunk_index
        self.metadata = metadata or {}

class MultiKBSearchOptions:
    def __init__(self, kb_names: List[str] = None, kb_weights: Dict[str, float] = None,
                 top_k: int = 10, dedup: bool = True, diversity_lambda: float = 0.7,
                 include_formulas: bool = True, include_code: bool = True):
        self.kb_names = kb_names or ['textbooks', 'papers', 'docs', 'competitions']
        self.kb_weights = kb_weights or {
            'textbooks': 1.0,
            'papers': 1.0,
            'docs': 0.8,
            'competitions': 0.5
        }
        self.top_k = top_k
        self.dedup = dedup
        self.diversity_lambda = diversity_lambda
        self.include_formulas = include_formulas
        self.include_code = include_code

class MultiKBSearchResult:
    def __init__(self, results: List[RAGSource] = None, dedup_removed: int = 0,
                 diversity_score: float = 0.0, per_kb_stats: Dict = None):
        self.results = results or []
        self.dedup_removed = dedup_removed
        self.diversity_score = diversity_score
        self.per_kb_stats = per_kb_stats or {}

class MultiKBSearcher:
    def __init__(self, vector_stores: Dict[str, FAISS] = None):
        self.vector_stores = vector_stores or {}
        self.embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY)
    
    def search(self, query: str, options: MultiKBSearchOptions = None) -> MultiKBSearchResult:
        options = options or MultiKBSearchOptions()
        
        all_results = []
        per_kb_stats = {}
        
        for kb_name in options.kb_names:
            if kb_name not in self.vector_stores:
                continue
            
            vs = self.vector_stores[kb_name]
            weight = options.kb_weights.get(kb_name, 1.0)
            
            try:
                docs_with_scores = vs.similarity_search_with_score(query, k=options.top_k * 3)
                
                for doc, score in docs_with_scores:
                    weighted_score = score * weight
                    source = RAGSource(
                        chunk_id=doc.metadata.get('chunk_id', ''),
                        document_id=doc.metadata.get('document_id', ''),
                        document_title=doc.metadata.get('title', doc.metadata.get('file_name', '')),
                        content=doc.page_content,
                        score=weighted_score,
                        chunk_index=doc.metadata.get('chunk_index', 0),
                        metadata=doc.metadata
                    )
                    all_results.append(source)
                
                per_kb_stats[kb_name] = {
                    'found': len(docs_with_scores),
                    'contributed': len(docs_with_scores)
                }
            except Exception as e:
                print(f"Search failed for {kb_name}: {e}")
                per_kb_stats[kb_name] = {'found': 0, 'contributed': 0}
        
        if options.dedup:
            all_results, dedup_count = self._deduplicate(all_results)
        else:
            dedup_count = 0
        
        all_results.sort(key=lambda x: x.score, reverse=True)
        all_results = all_results[:options.top_k]
        
        return MultiKBSearchResult(
            results=all_results,
            dedup_removed=dedup_count,
            per_kb_stats=per_kb_stats
        )
    
    def search_with_rewrite(self, rewrite_result: QueryRewriteResult, 
                           options: MultiKBSearchOptions = None) -> MultiKBSearchResult:
        options = options or MultiKBSearchOptions()
        
        all_results = []
        
        for variant in rewrite_result.variants[:5]:
            result = self.search(variant, options)
            all_results.extend(result.results)
        
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        if options.dedup:
            all_results, _ = self._deduplicate(all_results)
        
        all_results = all_results[:options.top_k]
        
        return MultiKBSearchResult(results=all_results)
    
    def _deduplicate(self, results: List[RAGSource]) -> tuple:
        seen_contents = set()
        unique = []
        removed = 0
        
        for result in results:
            content_hash = hash(result.content[:200])
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique.append(result)
            else:
                removed += 1
        
        return unique, removed
