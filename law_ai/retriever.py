# coding: utf-8
from typing import List

from langchain.schema.vectorstore import VectorStore
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.pydantic_v1 import Field
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter


class LawWebRetiever(BaseRetriever):
    # Inputs
    vectorstore: VectorStore = Field(
        ..., description="Vector store for storing web pages"
    )

    search: DuckDuckGoSearchAPIWrapper = Field(..., description="DuckDuckGo Search API Wrapper")
    num_search_results: int = Field(1, description="Number of pages per Google search")

    text_splitter: TextSplitter = Field(
        RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50),
        description="Text splitter for splitting web pages into chunks",
    )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:

        results = self.search.results(query, self.num_search_results)

        docs = []
        for res in results:
            docs.append(Document(
                page_content=res["snippet"],
                metadata={"link": res["link"], "title": res["title"]}
            ))

        docs = self.text_splitter.split_documents(docs)

        return docs
