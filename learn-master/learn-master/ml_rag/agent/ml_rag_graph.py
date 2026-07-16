from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import settings
from retrieval.query_rewriter import QueryRewriter, QueryRewriteResult
from retrieval.multi_kb_search import MultiKBSearcher, MultiKBSearchResult, MultiKBSearchOptions
from retrieval.post_processor import PostProcessor
from .prompt_templates import GROUNDED_GENERATOR_SYSTEM
from .quality_evaluator import QualityEvaluator
from .citation_engine import CitationEngine

class MLRagState(TypedDict):
    query: str
    user_id: str
    selected_kb_names: List[str]
    
    difficulty: Optional[str]
    domains: List[str]
    rewritten_queries: List[str]
    
    search_results: Optional[MultiKBSearchResult]
    retrieval_iterations: int
    
    answer: str
    citations: List
    formulas_included: bool
    code_included: bool
    
    quality_score: int
    quality_issues: List[str]
    needs_more_retrieval: bool

class MLRagGraph:
    def __init__(self, vector_stores=None):
        self.llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0.1)
        self.query_rewriter = QueryRewriter(self.llm)
        self.searcher = MultiKBSearcher(vector_stores)
        self.post_processor = PostProcessor()
        self.quality_evaluator = QualityEvaluator(self.llm)
        self.citation_engine = CitationEngine()
        
        self.max_retrieval_iterations = 2
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(MLRagState)
        
        workflow.add_node("query_analyzer", self._query_analyzer_node)
        workflow.add_node("query_rewriter", self._query_rewriter_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("grounding_generator", self._grounding_generator_node)
        workflow.add_node("quality_reflector", self._quality_reflector_node)
        
        workflow.set_entry_point("query_analyzer")
        workflow.add_edge("query_analyzer", "query_rewriter")
        workflow.add_edge("query_rewriter", "retriever")
        workflow.add_edge("retriever", "grounding_generator")
        workflow.add_edge("grounding_generator", "quality_reflector")
        
        workflow.add_conditional_edges(
            "quality_reflector",
            self._after_reflection,
            {
                "END": END,
                "retriever": "retriever"
            }
        )
        
        return workflow.compile()
    
    def _query_analyzer_node(self, state: MLRagState) -> MLRagState:
        print(f"[Query Analyzer] Processing: {state['query']}")
        rewrite_result = self.query_rewriter.rewrite(state['query'])
        
        return {
            **state,
            "difficulty": rewrite_result.difficulty,
            "domains": rewrite_result.target_domains,
            "rewritten_queries": rewrite_result.variants
        }
    
    def _query_rewriter_node(self, state: MLRagState) -> MLRagState:
        print(f"[Query Rewriter] Generated {len(state['rewritten_queries'])} variants")
        return state
    
    def _retriever_node(self, state: MLRagState) -> MLRagState:
        iteration = state.get("retrieval_iterations", 0) + 1
        print(f"[Retriever] Iteration {iteration}")
        
        options = MultiKBSearchOptions(
            kb_names=state['selected_kb_names'],
            top_k=settings.ML_RETRIEVAL_TOP_K
        )
        
        if state['rewritten_queries']:
            search_result = self.searcher.search_with_rewrite(
                QueryRewriteResult(
                    original_query=state['query'],
                    standardized_query=state['query'],
                    variants=state['rewritten_queries'],
                    sub_questions=[],
                    difficulty=state['difficulty'] or 'beginner',
                    target_domains=state['domains'],
                    has_formula_request=False,
                    has_code_request=False,
                    expected_answer_type="definition"
                ),
                options
            )
        else:
            search_result = self.searcher.search(state['query'], options)
        
        processed_result = self.post_processor.process(search_result)
        
        return {
            **state,
            "search_results": processed_result,
            "retrieval_iterations": iteration
        }
    
    def _grounding_generator_node(self, state: MLRagState) -> MLRagState:
        print("[Grounding Generator] Generating answer")
        
        if not state['search_results'] or not state['search_results'].results:
            return {
                **state,
                "answer": "抱歉，没有找到相关的知识库内容来回答您的问题。",
                "citations": [],
                "formulas_included": False,
                "code_included": False
            }
        
        context = self.post_processor.build_context_string(state['search_results'].results)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", GROUNDED_GENERATOR_SYSTEM),
            ("human", """[知识库上下文]
{context}

[用户问题]
{query}

[难度级别]
{difficulty}

请基于知识库内容回答问题，遵循重要规则。"""),
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "query": state['query'],
            "difficulty": state['difficulty'] or "intermediate"
        })
        
        citations = self.citation_engine.parse_citations(answer)
        citations = self.citation_engine.verify_citations(
            citations, state['search_results'].results
        )
        
        has_formulas = "$$" in answer or "$" in answer
        has_code = "```" in answer
        
        return {
            **state,
            "answer": answer,
            "citations": citations,
            "formulas_included": has_formulas,
            "code_included": has_code
        }
    
    def _quality_reflector_node(self, state: MLRagState) -> MLRagState:
        print("[Quality Reflector] Evaluating answer quality")
        
        context = ""
        if state['search_results']:
            context = self.post_processor.build_context_string(state['search_results'].results)
        
        quality_result = self.quality_evaluator.evaluate(
            state['query'],
            state['answer'],
            context
        )
        
        print(f"[Quality Reflector] Score: {quality_result.total_score}/10")
        
        return {
            **state,
            "quality_score": quality_result.total_score,
            "quality_issues": quality_result.issues,
            "needs_more_retrieval": quality_result.needs_more_retrieval and 
                                   state['retrieval_iterations'] < self.max_retrieval_iterations
        }
    
    def _after_reflection(self, state: MLRagState) -> str:
        if state['needs_more_retrieval']:
            print("[Quality Reflector] Needs more retrieval, retrying...")
            return "retriever"
        return "END"
    
    def run(self, query: str, user_id: str = "anonymous", 
            selected_kb_names: List[str] = None) -> dict:
        if selected_kb_names is None:
            selected_kb_names = ['textbooks', 'papers', 'docs', 'competitions']
        
        initial_state: MLRagState = {
            "query": query,
            "user_id": user_id,
            "selected_kb_names": selected_kb_names,
            "difficulty": None,
            "domains": [],
            "rewritten_queries": [],
            "search_results": None,
            "retrieval_iterations": 0,
            "answer": "",
            "citations": [],
            "formulas_included": False,
            "code_included": False,
            "quality_score": 0,
            "quality_issues": [],
            "needs_more_retrieval": False
        }
        
        final_state = self.graph.invoke(initial_state)
        
        return {
            "query": final_state["query"],
            "answer": final_state["answer"],
            "sources": [
                {
                    "document_title": r.document_title,
                    "chunk_index": r.chunk_index,
                    "content": r.content,
                    "score": r.score,
                    "metadata": r.metadata
                }
                for r in (final_state["search_results"].results if final_state["search_results"] else [])
            ],
            "citations": [
                {
                    "index": c.index,
                    "document_title": c.document_title,
                    "chunk_index": c.chunk_index,
                    "verified": c.verified,
                    "source_type": c.source_type
                }
                for c in final_state["citations"]
            ],
            "quality_score": final_state["quality_score"],
            "quality_issues": final_state["quality_issues"],
            "retrieval_iterations": final_state["retrieval_iterations"],
            "difficulty": final_state["difficulty"],
            "domains": final_state["domains"]
        }
