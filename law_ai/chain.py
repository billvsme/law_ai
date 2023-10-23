# coding: utf-8
from typing import Any, Optional, List
from collections import defaultdict
from operator import itemgetter

from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.schema.language_model import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import Callbacks
from langchain.chains.question_answering.stuff_prompt import PROMPT_SELECTOR
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document
from langchain.schema import format_document
from langchain.schema import BaseRetriever
from langchain.pydantic_v1 import Field
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers import BooleanOutputParser
from langchain.schema.runnable import RunnableMap
from langchain.chains.base import Chain
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)

from .utils import get_vectorstore, get_model
from .retriever import LawWebRetiever, get_multi_query_law_retiever
from .prompt import LAW_PROMPT, CHECK_LAW_PROMPT, HYPO_QUESTION_PROMPT
from .combine import combine_law_docs, combine_web_docs


class LawStuffDocumentsChain(StuffDocumentsChain):
    def _get_inputs(self, docs: List[Document], **kwargs: Any) -> dict:
        """Construct inputs from kwargs and docs.

        Format and the join all the documents together into one input with name
        `self.document_variable_name`. The pluck any additional variables
        from **kwargs.

        Args:
            docs: List of documents to format and then join into single input
            **kwargs: additional inputs to chain, will pluck any other required
                arguments from here.

        Returns:
            dictionary of inputs to LLMChain
        """
        # Join the documents together to put them in the prompt.
        law_book = defaultdict(list)
        law_web = defaultdict(list)
        for doc in docs:
            metadata = doc.metadata
            if 'book' in metadata:
                law_book[metadata["book"]].append(
                    format_document(doc, self.document_prompt).strip("\n"))
            elif 'link' in metadata:
                law_web[metadata["title"]].append(
                    format_document(doc, self.document_prompt).strip("\n"))

        law_str = ""
        for book, page_contents in law_book.items():
            law_str += f"《{book}》\n"
            law_str += "\n".join(page_contents)
            law_str += "\n\n"

        for web, page_contents in law_web.items():
            law_str += f"网页：{web}\n"
            law_str += "\n".join(page_contents)
            law_str += "\n\n"

        inputs = {
            k: v
            for k, v in kwargs.items()
            if k in self.llm_chain.prompt.input_variables
        }
        inputs[self.document_variable_name] = law_str
        return inputs


class LawQAChain(BaseRetrievalQA):
    vs_retriever: BaseRetriever = Field(exclude=True)
    web_retriever: BaseRetriever = Field(exclude=True)

    def _get_docs(
        self,
        question: str,
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        vs_docs = self.vs_retriever.get_relevant_documents(
            question, callbacks=run_manager.get_child()
        )

        web_docs = self.web_retriever.get_relevant_documents(
            question, callbacks=run_manager.get_child()
        )

        return vs_docs + web_docs

    async def _aget_docs(
        self,
        question: str,
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        vs_docs = await self.vs_retriever.aget_relevant_documents(
            question, callbacks=run_manager.get_child()
        )

        web_docs = await self.web_retriever.aget_relevant_documents(
            question, callbacks=run_manager.get_child()
        )

        return vs_docs + web_docs

    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return "law_qa"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[PromptTemplate] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> BaseRetrievalQA:
        """Initialize from LLM."""
        _prompt = prompt or PROMPT_SELECTOR.get_prompt(llm)
        llm_chain = LLMChain(llm=llm, prompt=_prompt, callbacks=callbacks)
        document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}"
        )

        combine_documents_chain = LawStuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context",
            document_prompt=document_prompt,
            callbacks=callbacks,
        )

        return cls(
            combine_documents_chain=combine_documents_chain,
            callbacks=callbacks,
            **kwargs,
        )


def get_check_law_chain(config: Any) -> Chain:
    model = get_model()

    check_chain = CHECK_LAW_PROMPT | model | BooleanOutputParser()

    return check_chain


def get_law_chain(config: Any, out_callback: AsyncIteratorCallbackHandler) -> Chain:
    law_vs = get_vectorstore(config.LAW_VS_COLLECTION_NAME)
    web_vs = get_vectorstore(config.WEB_VS_COLLECTION_NAME)

    vs_retriever = law_vs.as_retriever(search_kwargs={"k": config.LAW_VS_SEARCH_K})
    web_retriever = LawWebRetiever(
        vectorstore=web_vs,
        search=DuckDuckGoSearchAPIWrapper(),
        num_search_results=config.WEB_VS_SEARCH_K
    )

    multi_query_retriver = get_multi_query_law_retiever(vs_retriever, get_model())

    chain = (
        RunnableMap(
            {
                "law_docs": itemgetter("question") | multi_query_retriver,
                'web_docs': itemgetter("question") | web_retriever,
                "question": lambda x: x["question"]}
        )
        | RunnableMap(
            {
                "law_docs": lambda x: x["law_docs"],
                "web_docs": lambda x: x["web_docs"],
                "law_context": lambda x: combine_law_docs(x["law_docs"]),
                "web_context": lambda x: combine_web_docs(x["web_docs"]),
                "question": lambda x: x["question"]}
        )
        | RunnableMap({
                "law_docs": lambda x: x["law_docs"],
                "web_docs": lambda x: x["web_docs"],
                "law_context": lambda x: x["law_context"],
                "web_context": lambda x: x["web_context"],
                "prompt": LAW_PROMPT
            }
        )
        | RunnableMap({
            "law_docs": lambda x: x["law_docs"],
            "web_docs": lambda x: x["web_docs"],
            "law_context": lambda x: x["law_context"],
            "web_context": lambda x: x["web_context"],
            "answer": itemgetter("prompt") | get_model(callbacks=[out_callback]) | StrOutputParser()
        })
    )

    return chain


def get_hypo_questions_chain(config: Any) -> Chain:
    model = get_model()

    functions = [
        {
            "name": "hypothetical_questions",
            "description": "Generate hypothetical questions",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                    },
                },
                "required": ["questions"]
            }
        }
    ]

    chain = (
        {"context": lambda x: f"《{x.metadata['book']}》{x.page_content}"}
        | HYPO_QUESTION_PROMPT
        | model.bind(functions=functions, function_call={"name": "hypothetical_questions"})
        | JsonKeyOutputFunctionsParser(key_name="questions")
    )

    return chain
