import milvus_config as config
from pymilvus import MilvusClient, DataType, Function, FunctionType

def create_schema():
    # 创建集合
    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )
    
    # index field，类似于其他数据库的字段规定
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1024, enable_analyzer=True)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIM)
    schema.add_field(field_name="bm25", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="keywords", datatype=DataType.VARCHAR, max_length=256)
    
    # Add function to schema
    bm25_function = Function(
        name="text_bm25_emb",
        input_field_names=["text"],
        output_field_names=["bm25"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_function)
    
    print(f"✓ 创建集合 {config.COLLECTION_NAME}")
    return schema


def create_index(db_client):
    """创建索引"""
    # Prepare index parameters
    index_params = db_client.prepare_index_params()

    # Add indexes
    index_params.add_index(
        field_name="embedding",
        index_name="embedding_index",
        index_type=config.INDEX_TYPE,
        metric_type=config.METRIC_TYPE,
        params={
            "nlist": config.NLIST,
        } 
    )
    print("✓ 创建向量索引")

    index_params.add_index(
        field_name="bm25",
        index_name="bm25_sparse_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE"}, # or "DAAT_WAND" or "TAAT_NAIVE"
    )
    print("✓ 创建向bm25索引")
    return index_params

    
def create_collection(db_client):
    """创建 Milvus 集合"""
    # 检查集合是否存在
    if db_client.has_collection(config.COLLECTION_NAME):
        print(f"集合 {config.COLLECTION_NAME} 已存在")
        # db_client.drop_collection(config.COLLECTION_NAME)
        return
    
    db_client.create_collection(
        collection_name=config.COLLECTION_NAME,
        schema=create_schema(),
        index_params=create_index(db_client)
    )
