from milvus_insert import *
from pymilvus import Function, FunctionType, AnnSearchRequest
import numpy as np

def hybrid_search(db_client, query_text, top_k=5):
    # embedding = np.array([random.random() for _ in range(config.EMBEDDING_DIM)], dtype=np.float32).tolist()
    # text semantic search (dense)
    search_param_1 = {
        "data": [generate_embedding(query_text)],
        # "data": [embedding],
        "anns_field": "embedding",
        "param": {"nprobe": 10},
        "limit": 10
    }
    request_1 = AnnSearchRequest(**search_param_1)
    
    # full-text search (sparse)
    search_param_2 = {
        "data": [query_text],
        "anns_field": "bm25",
        "param": {
        "metric_type": "BM25",
        "params": {          # 这里放置索引专用的搜索参数
                "drop_ratio_search": 0.2  # 忽略查询向量中 20% 的最小值
            }
        },
        "limit": 10
    }
    request_2 = AnnSearchRequest(**search_param_2)
    
    reqs = [request_1, request_2]
    
    ranker = Function(
        name="rrf",
        input_field_names=[], # Must be an empty list
        function_type=FunctionType.RERANK,
        params={
            "reranker": "rrf", 
            "k": 60  # Optional
        }
    )
    
    res = db_client.hybrid_search(
        collection_name=config.COLLECTION_NAME,
        reqs=reqs,
        ranker=ranker,
        limit=top_k
    )
    return res

def get_topk_ans(db_client, query, top_k=5):
    db_client.load_collection(config.COLLECTION_NAME)
    
    search_res = hybrid_search(db_client, query, top_k=top_k)
    ids = []
    if search_res:
        for hit in search_res[0]:
            ids.append(hit['id'])
    else:
        return
    text = f'user query: {query}, 以下是RAG检索出的片段：\n'
    
    res = db_client.query(
        collection_name=config.COLLECTION_NAME,
        ids=ids,
        output_fields=["text", "source", "section", "keywords"],
        limit=top_k
    )
    for i, data in enumerate(res):
        text += f'检索的第{i}个片段：原文:\n'
        text += data.get('text', '') + '\n'
        text += f'source file: {data.get('source', '')}\n'
        text += f'section in source file: {data.get('section', '')}\n'
        text += f'keywords in chunk: {data.get('keywords', '')}\n'
        text += '=' * 20 + '\n'
    return text

def generate_user_answer(llm_client, db_client, query, top_k=5):
    text = get_topk_ans(db_client, query, top_k=top_k)
    prompt = f'''你是一个回答问题的智能体，现在给你用户问题（user query），以及检索出来的片段（text）,要求根据检索出来的片段回答。
        输入：
        user query: {query},
        text: {text}
        
        如果没有答案，则回答：暂未检索到答案。如果有，请严格按照以下格式回答：
                
        回答：你生成出来的答案。
        相关文档：source file - section in source file
    '''
    completion = llm_client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": prompt}
        ]
    )
    return completion.choices[0].message.content