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
