# coding: utf-8
from collections import defaultdict

from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.indexes._api import _batch
from dotenv import load_dotenv

load_dotenv()


def get_cached_embedder():
    fs = LocalFileStore("./.cache/embeddings")
    underlying_embeddings = OpenAIEmbeddings()

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, fs, namespace=underlying_embeddings.model
    )
    return cached_embedder


record_manager = SQLRecordManager(
    "chroma/law", db_url="sqlite:///law_record_manager_cache.sql"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=get_cached_embedder(),
    collection_name="law")


def get_vectorstore(collection_name="law"):
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=get_cached_embedder(),
        collection_name=collection_name)

    return vectorstore


def clear_vectorstore():
    index([], record_manager, vectorstore, cleanup="full", source_id_key="source")


def get_llm():
    llm = OpenAI(temperature=0, streaming=True)
    return llm


def law_index(docs, show_progress=True):
    info = defaultdict(int)

    pbar = None
    if show_progress:
        from tqdm import tqdm
        pbar = tqdm(total=len(docs))

    for docs in _batch(100, docs):
        result = index(
            docs,
            record_manager,
            vectorstore,
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


def source_text(docs):
    text = ""
    for doc in docs:
        if "book" in doc.metadata:
            text += f"相关法律：《{doc.metadata['book']}》\n"
            text += doc.page_content.strip("\n") + "\n"
        elif 'link' in doc.metadata:
            text += f"相关网页：{doc.metadata['title']}\n"
            text += f"地址：{doc.metadata['link']}\n"
            text += doc.page_content.strip("\n") + "\n"

        text += "\n"

    return text
