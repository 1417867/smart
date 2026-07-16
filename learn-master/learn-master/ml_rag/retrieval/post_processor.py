from typing import List, Dict
from .multi_kb_search import RAGSource, MultiKBSearchResult

class PostProcessor:
    def __init__(self, min_score: float = 0.3, max_context_length: int = 12000):
        self.min_score = min_score
        self.max_context_length = max_context_length
    
    def process(self, results: MultiKBSearchResult) -> MultiKBSearchResult:
        filtered = self._filter_by_score(results.results)
        deduplicated = self._semantic_dedup(filtered)
        final = self._truncate_to_context(deduplicated)
        
        return MultiKBSearchResult(
            results=final,
            dedup_removed=len(filtered) - len(final),
            per_kb_stats=results.per_kb_stats
        )
    
    def _filter_by_score(self, results: List[RAGSource]) -> List[RAGSource]:
        return [r for r in results if r.score >= self.min_score]
    
    def _semantic_dedup(self, results: List[RAGSource]) -> List[RAGSource]:
        unique = []
        for r in results:
            is_duplicate = False
            for u in unique:
                if self._similarity(r.content, u.content) > 0.9:
                    is_duplicate = True
                    if r.score > u.score:
                        unique.remove(u)
                        unique.append(r)
                    break
            if not is_duplicate:
                unique.append(r)
        return unique
    
    def _similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split()[:50])
        words2 = set(text2.lower().split()[:50])
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)
    
    def _truncate_to_context(self, results: List[RAGSource]) -> List[RAGSource]:
        total_length = 0
        truncated = []
        
        for r in results:
            header = f"[Source: {r.document_title}, Chunk {r.chunk_index + 1}]\n"
            block_length = len(header) + len(r.content)
            
            if total_length + block_length > self.max_context_length:
                remaining = self.max_context_length - total_length
                if remaining > 100:
                    truncated.append(RAGSource(
                        r.chunk_id, r.document_id, r.document_title,
                        r.content[:remaining - 20] + "\n[... truncated]",
                        r.score, r.chunk_index, r.metadata
                    ))
                break
            
            truncated.append(r)
            total_length += block_length
        
        return truncated
    
    def build_context_string(self, results: List[RAGSource]) -> str:
        parts = []
        for r in results:
            header = f"[Source: {r.document_title}, Chunk {r.chunk_index + 1}]"
            block = f"{header}\n{r.content}\n"
            parts.append(block)
        return '\n---\n\n'.join(parts)
