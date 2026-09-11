import milvus_config as config
import jieba.analyse
from openai import OpenAI
import os
from dotenv import load_dotenv
import random

load_dotenv()

# 初始化 OpenAI 客户端（用于生成向量）
embedding_client = OpenAI(
    api_key=os.getenv('API_KEY'),
    base_url=os.getenv('BASE_URL')
)
  
def generate_embedding(text):
    """生成文本向量"""
    try:
        response = embedding_client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=text,
            dimensions=config.EMBEDDING_DIM,
            encoding_format="float"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"生成向量时出错: {e}")
        return None
    
def extract_keywords(text, top_n=10):
    """提取关键词（用于混合检索）"""
    words = jieba.analyse.extract_tags(text, topK=top_n)
    return " ".join(words)


def insert_data(db_client, chunks):
    """插入数据到 Milvus"""
    # 准备数据
    data = []

    print("正在生成向量...")
    for i, chunk in enumerate(chunks):
        # 生成向量
        embedding = generate_embedding(chunk['text'])
        # embedding = [random.random() for _ in range(config.EMBEDDING_DIM)]
        if embedding is None:
            print(f"跳过第 {i+1} 个片段（向量生成失败）")
            continue

        # 提取关键词
        keywords = extract_keywords(chunk['text'])

        text = chunk['text'][:256]
        data.append(
            {"text": text, "embedding": embedding, "source":chunk['source'], "section": chunk['heading'], "keywords": keywords}
        )

        if (i + 1) % 10 == 0:
            print(f"  已处理 {i+1}/{len(chunks)} 个片段")
    if data:
        res = db_client.insert(
            collection_name=config.COLLECTION_NAME,
            data=data
        )
        if res:
            print(f"✓ 成功插入 {len(data)} 条数据")
            db_client.flush(config.COLLECTION_NAME)
        else:
            print(f"❌️ 插入数据失败")
    else:
        print("没有数据需要插入")
