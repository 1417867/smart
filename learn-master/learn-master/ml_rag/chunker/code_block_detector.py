import re
from typing import List, Dict

class CodeBlockDetector:
    CODE_BLOCK_PATTERN = re.compile(r'```(\w*)\n?([\s\S]*?)```')
    IMPORT_PATTERN = re.compile(r'(?:^|\n)import\s+(\w+)')
    API_CALL_PATTERN = re.compile(r'(sklearn|torch|tf|numpy|pandas)\.\w+')
    
    @classmethod
    def detect_code_blocks(cls, text: str) -> List[Dict]:
        blocks = []
        
        for match in cls.CODE_BLOCK_PATTERN.finditer(text):
            lang = match.group(1) or 'text'
            code = match.group(2)
            
            imports = cls.IMPORT_PATTERN.findall(code)
            api_calls = list(set(cls.API_CALL_PATTERN.findall(code)))
            
            blocks.append({
                'language': lang,
                'code': code,
                'imports': imports,
                'api_calls': api_calls,
                'start': match.start(),
                'end': match.end()
            })
        
        return blocks
    
    @classmethod
    def has_code_request(cls, query: str) -> bool:
        code_keywords = ['代码', '实现', '示例', '编程', 'python', '代码示例',
                         '代码实现', '如何实现', '写一个']
        return any(keyword in query.lower() for keyword in code_keywords)
    
    @classmethod
    def extract_api_names(cls, text: str) -> List[str]:
        api_calls = cls.API_CALL_PATTERN.findall(text)
        return list(set(api_calls))
    
    @classmethod
    def analyze_code_complexity(cls, code: str) -> str:
        lines = code.strip().split('\n')
        if len(lines) <= 5:
            return 'simple'
        elif len(lines) <= 20:
            return 'medium'
        else:
            return 'complex'
