# coding: utf-8


class Config:
    LAW_BOOK_PATH = "./Law-Book"
    LAW_BOOK_CHUNK_SIZE = 100
    LAW_BOOK_CHUNK_OVERLAP = 20
    LAW_VS_COLLECTION_NAME = "law"
    LAW_VS_SEARCH_K = 2

    WEB_VS_COLLECTION_NAME = "web"
    WEB_VS_SEARCH_K = 2


config = Config()
