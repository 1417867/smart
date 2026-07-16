import re
from typing import List, Dict, Tuple

class FormulaExtractor:
    BLOCK_FORMULA_PATTERN = re.compile(r'\$\$([\s\S]*?)\$\$')
    INLINE_FORMULA_PATTERN = re.compile(r'(?<!\$)\$([^$\n]+)\$(?!\$)')
    
    @classmethod
    def extract_formulas(cls, text: str) -> Tuple[List[str], List[str]]:
        block_formulas = cls.BLOCK_FORMULA_PATTERN.findall(text)
        inline_formulas = cls.INLINE_FORMULA_PATTERN.findall(text)
        return block_formulas, inline_formulas
    
    @classmethod
    def extract_all(cls, text: str) -> List[Dict]:
        results = []
        
        for match in cls.BLOCK_FORMULA_PATTERN.finditer(text):
            results.append({
                'type': 'block',
                'formula': match.group(1),
                'start': match.start(),
                'end': match.end(),
                'full_match': match.group(0)
            })
        
        for match in cls.INLINE_FORMULA_PATTERN.finditer(text):
            results.append({
                'type': 'inline',
                'formula': match.group(1),
                'start': match.start(),
                'end': match.end(),
                'full_match': match.group(0)
            })
        
        return results
    
    @classmethod
    def replace_with_placeholders(cls, text: str) -> Tuple[str, List[Dict]]:
        formulas = cls.extract_all(text)
        placeholders = []
        
        for i, formula in enumerate(formulas):
            placeholder = f"[FORMULA_{i}]"
            text = text.replace(formula['full_match'], placeholder, 1)
            placeholders.append({
                'placeholder': placeholder,
                'formula': formula['formula'],
                'type': formula['type']
            })
        
        return text, placeholders
    
    @classmethod
    def restore_from_placeholders(cls, text: str, placeholders: List[Dict]) -> str:
        for placeholder in placeholders:
            wrapper = '$$' if placeholder['type'] == 'block' else '$'
            text = text.replace(placeholder['placeholder'], 
                               f"{wrapper}{placeholder['formula']}{wrapper}")
        return text
    
    @classmethod
    def has_formula_request(cls, query: str) -> bool:
        formula_keywords = ['公式', '推导', '证明', '表达式', '数学表达式',
                            '损失函数', '激活函数', '梯度', '偏导', '导数']
        return any(keyword in query for keyword in formula_keywords)
