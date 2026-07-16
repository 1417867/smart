from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from config.settings import settings
from .ml_terms import find_ml_terms, detect_difficulty, detect_domains

class QueryRewriteResult(BaseModel):
    original_query: str = Field(description="原始查询")
    standardized_query: str = Field(description="术语标准化后的查询")
    variants: List[str] = Field(description="检索变体列表")
    sub_questions: List[str] = Field(description="子问题分解")
    difficulty: str = Field(description="难度级别")
    target_domains: List[str] = Field(description="匹配的ML子领域")
    has_formula_request: bool = Field(description="是否包含公式需求")
    has_code_request: bool = Field(description="是否包含代码需求")
    expected_answer_type: str = Field(description="期望的回答类型")

class QueryRewriter:
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model=settings.LLM_MODEL, temperature=0)
        self.parser = JsonOutputParser(pydantic_object=QueryRewriteResult)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个机器学习查询分析专家。
分析用户的查询，生成多个检索变体和子问题分解。

【术语标准化规则】
- 将中文术语转换为中英文对照形式
- "反向传播" → "反向传播 (Backpropagation)"
- "梯度下降" → "梯度下降 (Gradient Descent)"

【生成规则】
1. variants: 生成5-8个检索变体，包含中英文形式
2. sub_questions: 如果查询复杂，分解为2-4个子问题
3. difficulty: beginner | intermediate | advanced
4. target_domains: dl, nlp, cv, rl, optimization, ml
5. has_formula_request: 包含"公式"、"推导"、"证明"等关键词为true
6. has_code_request: 包含"代码"、"实现"、"示例"等关键词为true
7. expected_answer_type: definition | derivation | comparison | implementation | application

【输出格式】JSON格式，不要包含其他文字。"""),
            ("human", "用户查询: {query}\n\n请分析并生成检索变体。"),
        ])
        self.chain = self.prompt | self.llm | self.parser

    def rewrite(self, query: str) -> QueryRewriteResult:
        if not settings.ML_QUERY_REWRITE_ENABLED:
            difficulty = detect_difficulty(query)
            domains = detect_domains(query)
            return QueryRewriteResult(
                original_query=query,
                standardized_query=query,
                variants=[query],
                sub_questions=[],
                difficulty=difficulty,
                target_domains=domains,
                has_formula_request=False,
                has_code_request=False,
                expected_answer_type="definition"
            )
        
        try:
            result = self.chain.invoke({"query": query})
            if not isinstance(result, QueryRewriteResult):
                result = QueryRewriteResult(**result)
            return result
        except Exception as e:
            print(f"Query rewriting failed, falling back to basic analysis: {e}")
            difficulty = detect_difficulty(query)
            domains = detect_domains(query)
            return QueryRewriteResult(
                original_query=query,
                standardized_query=query,
                variants=[query],
                sub_questions=[],
                difficulty=difficulty,
                target_domains=domains,
                has_formula_request="公式" in query or "推导" in query or "证明" in query,
                has_code_request="代码" in query or "实现" in query or "示例" in query,
                expected_answer_type="definition"
            )
