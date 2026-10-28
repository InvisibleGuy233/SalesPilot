from pathlib import Path
from llama_cloud import LlamaCloud
import os
import milvus_config as config
from pymilvus import MilvusClient
from openai import OpenAI
from langchain_openai import ChatOpenAI

PROJECT_PATH = Path(__file__).resolve().parent
REGISTRY_FILE_PATH = PROJECT_PATH / "docs" / "processed_files.json"

llama_cloud_client = None
def get_llama_cloud_client():
    global llama_cloud_client
    if llama_cloud_client is None:
        llama_cloud_client = LlamaCloud(api_key=os.getenv('LLAMA_CLOUD_API_KEY'))
        print(f"✓ 已创建llama_cloud_client")
    return llama_cloud_client


db_client = None
def get_db_client():
    global db_client
    if db_client is None:
        db_client = MilvusClient(
            uri=f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}",
            token="root:Milvus"
        )
        print(f"✓ 已连接到 Milvus ({config.MILVUS_HOST}:{config.MILVUS_PORT})")
    return db_client

llm_client = None
def get_llm_client():
    global llm_client
    if llm_client is None:
        llm_client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL")
        )
        print(f"✓ 已创建llm_client")
    return llm_client

chat_client = None
def get_chat_client():
    global chat_client
    if chat_client is None:
        chat_client = ChatOpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
            model = os.getenv("LLM_MODEL"),
            streaming=True
        )
        print(f"✓ 已创建llm_client")
    return chat_client