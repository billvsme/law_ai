# coding: utf-8
from dotenv import load_dotenv

import sys
import asyncio
from pprint import pprint

from langchain.callbacks import AsyncIteratorCallbackHandler

from law_ai.loader import LawLoader
from law_ai.splitter import LawSplitter
from law_ai.utils import law_index, clear_vectorstore, get_record_manager
from law_ai.chain import get_law_chain

from config import config


load_dotenv()


def init_vectorstore():
    record_manager = get_record_manager()
    record_manager.create_schema()

    clear_vectorstore()

    text_splitter = LawSplitter.from_tiktoken_encoder(
        chunk_size=config.LAW_BOOK_CHUNK_SIZE, chunk_overlap=config.LAW_BOOK_CHUNK_OVERLAP
    )
    docs = LawLoader(config.LAW_BOOK_PATH).load_and_split(text_splitter=text_splitter)
    info = law_index(docs)
    pprint(dict(info))


async def run_shell():
    chain = get_law_chain(config)

    while True:
        query = input("\n用户:")
        if query.strip() == "stop":
            break
        print("\n法律小助手:", end="")
        callback = AsyncIteratorCallbackHandler()
        task = asyncio.create_task(
            chain.ainvoke({"question": query}, config={"callbacks": [callback]}))
        async for new_token in callback.aiter():
            print(new_token, end="", flush=True)

        print("\n\n")
        res = await task
        print(res["law_context"] + "\n" + res["web_context"])


async def run_web():
    import gradio as gr

    chain = get_law_chain(config)

    async def chat(message, history):
        callback = AsyncIteratorCallbackHandler()
        task = asyncio.create_task(
            chain.ainvoke({"question": message}, config={"callbacks": [callback]}))

        response = ""
        async for new_token in callback.aiter():
            response += new_token
            yield response

        res = await task
        for new_token in ["\n\n", res["law_context"], "\n", res["web_context"]]:
            response += new_token
            yield response

    demo = gr.ChatInterface(
        fn=chat, examples=["故意杀了一个人，会判几年？", "杀人自首会减刑吗？"], title="法律AI小助手")

    demo.queue()
    demo.launch(
        server_name=config.WEB_HOST, server_port=config.WEB_PORT,
        auth=(config.WEB_USERNAME, config.WEB_PASSWORD),
        auth_message="默认用户名密码: username / password")


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
    parser.add_argument(
        "-w",
        "--web",
        action="store_true",
        help=('''
            run web
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
    if args.web:
        asyncio.run(run_web())
