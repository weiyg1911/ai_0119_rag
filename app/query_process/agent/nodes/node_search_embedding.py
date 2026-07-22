import time
import sys
from app.utils.task_utils import  add_done_task,add_running_task

from app.core.logger import logger, node_log, step_log
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
from app.clients.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from app.conf.milvus_config import milvus_config


def step_1_data_validates(state):
    item_names = state["item_names"]
    rewritten_query=state["rewritten_query"]
    if not item_names or not rewritten_query:
        msg = "rewritten_query 或者 item_names 为空"
        logger.error(msg)
        raise ValueError(msg)

def step_2_query_embedding(state):
    rewritten_query=state["rewritten_query"]
    item_names = state["item_names"]
    result = generate_embeddings([rewritten_query])


    dense_vector = result['dense'][0]
    sparse_vector = result["sparse"][0]

    filter_expr = f"item_name IN {item_names}"
    # filter_expr = f"ARRAY_CONTAINS_ANY(item_name, {item_names})"


    reqs = create_hybrid_search_requests(dense_vector, sparse_vector, expr=filter_expr)
    milvus_client = get_milvus_client()

    response = hybrid_search(
            client=milvus_client,
            collection_name=milvus_config.chunks_collection,
            reqs=reqs,
            limit=5,
            ranker_weights=(0.8, 0.2),
            output_fields=["chunk_id", "item_name", "content", "title", "parent_title", "part", "file_title"],
            norm_score=True
        )
    return response[0] if len(response) > 0 else []


def node_search_embedding(state):
    """
    节点功能：进行向量内容检索
    """
    print("---量内容检索 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 搜索假设性答案
    print("量内容检索答案！！")
    time.sleep(7)
    step_1_data_validates(state)
    chunks = step_2_query_embedding(state)
    # ...
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    print("---量内容检索 处理结束---")
    return {"embedding_chunks":chunks}


    
if __name__ == "__main__":
    # 模拟测试数据
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "HAK 180 烫金机# 对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。",  # 模拟改写后的查询
        "item_names": ["HAK 180 烫金机"],  # 模拟已确认的商品名
        "is_stream": False
    }

    print("\n>>> 开始测试 node_search_embedding 节点...")
    try:
        # 执行节点函数
        result = node_search_embedding(test_state)
        logger.info(f"检索结果汇总：{result}")
        # 验证结果
        chunks = result.get("embedding_chunks", [])
        print(f"\n>>> 测试完成！检索到 {len(chunks)} 条结果")
        print(f"\n>>> 测试完成！检索到 {chunks} 条结果")
    except Exception as e:
        logger.error(f"测试运行失败: {e}", exc_info=True)