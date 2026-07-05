import ast
with open('src/decision/llm_reasoner.py', encoding='utf-8') as f:
    ast.parse(f.read())
print('Syntax OK')
