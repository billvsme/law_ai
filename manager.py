# coding: utf-8
import asyncio
import sys
from pprint import pprint

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.utilities import DuckDuckGoSearchAPIWrapper

from law_ai.loader import LawLoader
from law_ai.splitter import LawSplitter
from law_ai.utils import get_vectorstore, get_llm, law_index, \
        clear_vectorstore, record_manager, source_text
from law_ai.chain import LawQAChain
from law_ai.retriever import LawWebRetiever

from config import config


def init_vectorstore():
    record_manager.create_schema()

    clear_vectorstore()

    text_splitter = LawSplitter.from_tiktoken_encoder(
        chunk_size=config.LAW_BOOK_CHUNK_SIZE, chunk_overlap=config.LAW_BOOK_CHUNK_OVERLAP
    )
    docs = LawLoader(config.LAW_BOOK_PATH).load_and_split(text_splitter=text_splitter)
    info = law_index(docs)
    pprint(dict(info))


async def run_shell():
    llm = get_llm()
    law_vs = get_vectorstore(config.LAW_VS_COLLECTION_NAME)
    web_vs = get_vectorstore(config.WEB_VS_COLLECTION_NAME)

    web_retriever = LawWebRetiever(
        vectorstore=web_vs,
        search=DuckDuckGoSearchAPIWrapper(),
        num_search_results=config.WEB_VS_SEARCH_K
    )

    chain = LawQAChain.from_llm(
        llm,
        vs_retriever=law_vs.as_retriever(search_kwargs={"k": config.LAW_VS_SEARCH_K}),
        web_retriever=web_retriever,
        return_source_documents=True,
    )

    while True:
        query = input("\n用户:")
        if query.strip() == "stop":
            break
        print("\n法律小助手:", end="")
        callback = AsyncIteratorCallbackHandler()
        task = asyncio.create_task(
            chain.ainvoke({"query": query}, config={"callbacks": [callback]}))
        async for t in callback.aiter():
            print(t, end="", flush=True)

        print("\n")
        res = await task
        _, docs = res['result'], res['source_documents']

        print(f"{source_text(docs)}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description="please specify only one operate method once time.")
    parser.add_argument(
        "-i",
        "--init",
        action="store_true",
        help=('''
            init vectorstore
        ''')
    )
    parser.add_argument(
        "-s",
        "--shell",
        action="store_true",
        help=('''
            run shell
        ''')
    )

    if len(sys.argv) <= 1:
        parser.print_help()
        exit()

    args = parser.parse_args()
    if args.init:
        init_vectorstore()
    if args.shell:
        asyncio.run(run_shell())
