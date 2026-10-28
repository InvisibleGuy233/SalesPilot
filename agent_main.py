import file_extraction, milvus_create, milvus_insert, langgraph_agent
import re
from dotenv import load_dotenv
import json
from global_arg import *
from logs import Log

load_dotenv()

def load_file_set() -> set[str]:
    if not REGISTRY_FILE_PATH.exists():
        with REGISTRY_FILE_PATH.open("w", encoding="utf-8") as f:
            f.write("[]")
        return set()

    with REGISTRY_FILE_PATH.open("r", encoding="utf-8") as f:
        return set(json.load(f))


def save_file_set(file_set: set[str]) -> None:
    # 先写临时文件，再替换，降低写到一半留下损坏文件的风险
    temp_path = REGISTRY_FILE_PATH.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(sorted(file_set), f, ensure_ascii=False, indent=2)

    temp_path.replace(REGISTRY_FILE_PATH)

def normalize(filename):
    name, ext = filename.rsplit('.', 1)  
    name = re.sub(r'\(\d+\)$', '', name) 
    return f"{name}.{ext}"

def offline_doc_processing(file_list):
    # 先离线处理files into chunks
    llama_cloud_client = get_llama_cloud_client()
    db_client = get_db_client()
    milvus_create.create_collection(db_client)
    
    file_set = load_file_set()
    
    for file in file_list:
        norm_file = PROJECT_PATH / normalize(file)
        if norm_file.name in file_set:
            # 写log还不知道咋写，就是说还是先等重构吧
            Log(f"文件已存在，跳过：{norm_file.name}")
            continue
        md_file = file_extraction.extract_text_into_md(llama_cloud_client, norm_file)
        chunks = file_extraction.split_chunks(md_file)
        
        # with open(PROJECT_PATH / 'chunks.txt', 'a', encoding='utf-8') as f:
        #     f.write(f"\n\n======这是一个新的markdown文件: {md_file.name} =========\n\n")
        #     for i, chunk in enumerate(chunks): 
        #         f.write(f"\n======这是第 {i} 个 chunk =========\n")
        #         f.write(f"chunk text: {chunk['text']}\n heading: {chunk['heading']}\n source: {chunk['source']}")
                
        milvus_insert.insert_data(db_client, chunks)
        file_set.add(norm_file.name)
        save_file_set(file_set)

        
if __name__ == '__main__':
    files_upload = ['docs/A User Manual.pdf', 'docs/B.pdf']
    offline_doc_processing(files_upload)
    
    query_text = "your question?"
    res = langgraph_agent.run_agent_langgraph(query_text)

    
    
    
    
    
    