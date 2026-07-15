import sys

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task
from pymilvus import DataType

def _create_collection(client: get_milvus_client):

    schema = client.create_schema(
        auto_id=True,
        enable_dynamic_fields=True,
    )

    schema.add_field('chunk_id', datatype=DataType.INT64, is_primary=True)
    schema.add_field('file_title', datatype=DataType.VARCHAR, max_length=4092)
    schema.add_field('item_name', datatype=DataType.VARCHAR, max_length=4092)
    schema.add_field('content', datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field('title', datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field('parent_title', datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field('part', datatype=DataType.INT8)
    schema.add_field('dense_vector', datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field('sparse_vector', datatype=DataType.SPARSE_FLOAT_VECTOR)

    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name='dense_vector',
        index_type='AUTOINDEX',
        metric_type="COSINE",
    )

    index_params.add_index(
        field_name='sparse_vector',
        index_type='SPARSE_INVERTED_INDEX',
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE" }
    )

    client.create_collection(
        collection_name = milvus_config.chunks_collection,
        schema=schema,
        index_params=index_params,
    )


@step_log("step_1_insert_data 幂等清理并写入chunks")
def step_1_insert_data(state: ImportGraphState):
    chunks = state["chunks"]
    item_name = state["item_name"]
    if not chunks or len(chunks) == 0:
        msg = "chunks为空，错误"
        logger.error(msg)
        raise ValueError(msg)

    client = get_milvus_client()

    if not client.has_collection(collection_name=milvus_config.chunks_collection):
        _create_collection(client)

    client.delete(collection_name=milvus_config.chunks_collection, filter=f"item_name=='{item_name}'")



    insert_data = []

    for chunk in chunks:
        insert_data.append({
            "file_title": chunk["file_title"],
            "item_name": chunk["item_name"],
            "content": chunk["content"],
            "title": chunk["title"],
            "parent_title": chunk["parent_title"],
            "part": chunk["part"],
            "dense_vector": chunk["dense_vector"],
            "sparse_vector": chunk["sparse_vector"],
        })

    client.insert(collection_name=milvus_config.chunks_collection, data=insert_data)


@node_log("node_import_milvus")
def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    为什么叫这个名字: 将处理好的向量数据写入 Milvus 数据库。
    未来要实现:
    1. 连接 Milvus。
    2. 根据 item_name 删除旧数据 (幂等性)。
    3. 批量插入新的向量数据。
    """
    add_running_task(state["task_id"], "node_import_milvus")
    step_1_insert_data(state)
    add_done_task(state["task_id"], "node_import_milvus")
    return state

if __name__ == '__main__':
    # --- 单元测试 ---
    # 目的：验证 Milvus 导入节点的完整流程，包括连接、创建集合、清理旧数据和插入新数据。
    import sys
    import os
    from dotenv import load_dotenv

    # 加载环境变量 (自动寻找项目根目录的 .env)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造测试数据
    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "item_name":"测试项目_Milvus",
        "chunks": [
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title":"test.pdf",
                "part":1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
,
            {
                "content": "Milvus 测试文本 2",
                "title": "测试标题2",
                "item_name": "测试项目_Milvus2",  # 必须有 item_name，用于幂等清理
                "parent_title": "test.pdf2",
                "part": 1,
                "file_title": "test.pdf2",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        # 检查必要的环境变量
        if not os.getenv("MILVUS_URL"):
            print("❌ 未设置 MILVUS_URL，无法连接 Milvus")
        elif not os.getenv("CHUNKS_COLLECTION"):
            print("❌ 未设置 CHUNKS_COLLECTION")
        else:
            # 执行节点函数
            result_state = node_import_milvus(test_state)

            # 验证结果
            chunks = result_state.get("chunks", [])
            if chunks and chunks[0].get("chunk_id"):
                print(f"✅ Milvus 导入测试通过，生成 ID: {chunks[0]['chunk_id']}")
            else:
                print("❌ 测试失败：未能获取 chunk_id")

    except Exception as e:
        print(f"❌ 测试失败: {e}")