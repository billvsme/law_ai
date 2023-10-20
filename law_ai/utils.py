# coding: utf-8
from collections import defaultdict

from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain.vectorstores import Chroma
from langchain.indexes._api import _batch
from langchain.chat_models import ChatOpenAI


def get_cached_embedder():
    fs = LocalFileStore("./.cache/embeddings")
    underlying_embeddings = OpenAIEmbeddings()

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, fs, namespace=underlying_embeddings.model
    )
    return cached_embedder


def get_record_manager():
    return SQLRecordManager(
        "chroma/law", db_url="sqlite:///law_record_manager_cache.sql"
    )


def get_vectorstore(collection_name="law"):
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=get_cached_embedder(),
        collection_name=collection_name)

    return vectorstore


def clear_vectorstore():
    index([], get_record_manager(), get_vectorstore("law"), cleanup="full", source_id_key="source")


def get_model():
    model = ChatOpenAI(streaming=True)
    return model


def law_index(docs, show_progress=True):
    info = defaultdict(int)

    pbar = None
    if show_progress:
        from tqdm import tqdm
        pbar = tqdm(total=len(docs))

    for docs in _batch(100, docs):
        result = index(
            docs,
            get_record_manager(),
            get_vectorstore("law"),
            cleanup=None,
            # cleanup="full",
            source_id_key="source",
        )
        for k, v in result.items():
            info[k] += v

        if pbar:
            pbar.update(len(docs))

    if pbar:
        pbar.close()

    return info
