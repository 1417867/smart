from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Dict, List
from config.settings import settings

class QualityScore(BaseModel):
    accuracy: int = Field(description="事实准确性 0-2")
    citations: int = Field(description="引用完整性 0-2")
    formulas: int = Field(description="公式正确性 0-2")
    code: int = Field(description="代码可用性 0-2")
    teaching: int = Field(description="教学适配性 0-2")

class QualityResult(BaseModel):
    scores: QualityScore = Field(description="各维度评分")
    total_score: int = Field(description="总分 0-10")
    issues: List[str] = Field(description="问题列表")
    suggestions: List[str] = Field(description="改进建议")
    needs_more_retrieval: bool = Field(description="是否需要补充检索")

class QualityEvaluator:
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model=settings.LLM_MODEL, temperature=0)
        self.parser = JsonOutputParser(pydantic_object=QualityResult)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个 ML 问答质量评估专家。
评估以下问答的质量，返回 JSON 格式的评估结果。

【评估维度】(每项 0-2 分)
1. 事实准确性: 回答是否与知识库内容一致?
2. 引用完整性: 关键陈述是否都有引用?
3. 公式正确性: 公式是否完整且正确?
4. 代码可用性: 代码是否可运行?
5. 教学适配性: 解释难度是否适合用户?

总分 = 各维度得分之和 (0-10)

如果总分 < 7 且有明确的信息缺失，needs_more_retrieval = true"""),
            ("human", """查询: {query}
回答: {answer}
知识库上下文: {context}

请评估回答质量。"""),
        ])
        self.chain = self.prompt | self.llm | self.parser
    
    def evaluate(self, query: str, answer: str, context: str) -> QualityResult:
        try:
            result = self.chain.invoke({
                "query": query,
                "answer": answer,
                "context": context[:8000]
            })
            if not isinstance(result, QualityResult):
                result = QualityResult(**result)
            return result
        except Exception as e:
            print(f"Quality evaluation failed: {e}")
            return QualityResult(
                scores=QualityScore(accuracy=1, citations=0, formulas=0, code=0, teaching=1),
                total_score=2,
                issues=["评估失败"],
                suggestions=["建议重新生成"],
                needs_more_retrieval=True
            )
    
    def is_passed(self, result: QualityResult) -> bool:
        return result.total_score >= 7
