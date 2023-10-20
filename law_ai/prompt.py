# coding: utf-8
from langchain.prompts import PromptTemplate

law_prompt_template = """你是一个专业的律师，请你结合以下内容回答问题:
{law_context}

{web_context}

问题: {question}
"""
LAW_PROMPT = PromptTemplate(
    template=law_prompt_template, input_variables=["law_context", "web_context", "question"]
)

check_law_prompt_template = """你是一个专业律师，请判断下面问题是否和法律相关，相关请回答YES，不想关请回答NO，不允许其它回答，不允许在答案中添加编造成分。
问题: {question}
"""

CHECK_LAW_PROMPT = PromptTemplate(
    template=check_law_prompt_template, input_variables=["question"]
)
