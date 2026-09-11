from pymilvus import DataType

MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
COLLECTION_NAME = "milvus_db"
EMBEDDING_DIM = 1024
EMBEDDING_MODEL = "text-embedding-v4"

# index info
METRIC_TYPE = "L2"
INDEX_TYPE = "IVF_FLAT"
NPROBE = 10
NLIST = 128



