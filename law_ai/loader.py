# coding: utf-8
from typing import Any
from langchain.document_loaders import TextLoader, DirectoryLoader


class LawLoader(DirectoryLoader):
    """Load law books."""
    def __init__(self, path: str, **kwargs: Any) -> None:
        loader_cls = TextLoader
        glob = "**/*.md"
        super().__init__(path, loader_cls=loader_cls, glob=glob, **kwargs)
