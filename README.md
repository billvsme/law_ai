法律AI助手
=========
法律AI助手，法律RAG，通过倒入全部200+本法律手册、网页搜索内容结合LLM回答你的问题，并且给出对应的法规和网站，基于langchain，openai，chroma，duckduckgo-search

## 原理
基于langchain链式调用，先按条切分法律条文，导入向量数据Chroma。
问答相关问题时，先通过相似度搜索向量数据，获取相关法律条文，然后通过DuckDuckGo互联网搜索相关网页，然后合并法律条文和网页内容，对问题进行回答。

**初始化init**
```
LawLoader -> LawSplitter -> 向量数据库(Chroma)
```

**提问**
```
LawQAChain -> 向量数据库(Chroma) -> DuckDuckGo互联网搜索 -> stuff合并(LawStuffDocumentsChain) -> LLM -> callback异步输出
```
  
## 初始化运行环境
```
# 创建.env 文件
cp .env.example .env

# 修改.env 中的内容
vim .env

# 安装venv环境
python -m venv ~/.venv/law
. ~/.venv/law
pip install -r requirements.txt
```

## 初始化向量数据库
```
# 加载和切分法律手册，初始化向量数据库
python manager.py --init
```

## 运行对话
```
python manager.py --shell
```

<a href="https://sm.ms/image/7E4zMpbafCPvNxX" target="_blank"><img src="https://s2.loli.net/2023/10/19/7E4zMpbafCPvNxX.png"></a>
