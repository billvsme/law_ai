# coding: utf-8
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.docstore.document import Document
from typing import Any, Iterable, List


class LawSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, **kwargs: Any) -> None:
        """Initialize a LawSplitter."""
        separators = [r"第\S*条 "]
        is_separator_regex = True

        headers_to_split_on = [
            ("#", "header1"),
            ("##", "header2"),
            ("###", "header3"),
            ("####", "header4"),
        ]

        self.md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        super().__init__(separators=separators, is_separator_regex=is_separator_regex, **kwargs)

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents:
            md_docs = self.md_splitter.split_text(doc.page_content)
            for md_doc in md_docs:
                texts.append(md_doc.page_content)

                metadatas.append(
                    md_doc.metadata | doc.metadata | {"book": md_doc.metadata.get("header1")})

        return self.create_documents(texts, metadatas=metadatas)
