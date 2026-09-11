from llama_cloud import LlamaCloud
import file_extraction, milvus_create, milvus_insert, langgraph_agent
from dotenv import load_dotenv
import os

load_dotenv()

def offline_doc_processing(file_list):
    # 先离线处理files into chunks
    client = LlamaCloud(api_key=os.getenv('LLAMA_CLOUD_API_KEY'))
    db_client = langgraph_agent.get_db_client()
    milvus_create.create_collection(db_client)
    
    for file in file_list:
        md_file = file_extraction.extract_text_into_md(client, file)
        chunks = file_extraction.split_chunks(md_file)
        
        milvus_insert.insert_data(db_client, chunks)

        
if __name__ == '__main__':
    file_list = ['./星辰电动ES9·未来旗舰电动SUV产品介绍.pdf']
    offline_doc_processing(file_list)
    
    query_text = "未来旗舰电动SUV产品经济版要多少钱？"
    # res = run_agent(query_text)
    res = langgraph_agent.run_agent_langgraph(query_text)

    
    
    
    
    
    