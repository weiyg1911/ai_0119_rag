import time
import sys
from app.clients.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from app.conf.milvus_config import milvus_config
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
from langchain_core.output_parsers import StrOutputParser
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger, node_log, step_log
from app.clients.mongo_history_utils import save_chat_message, get_recent_messages
from langchain_core.messages import SystemMessage, HumanMessage


# 大模型工具：获取大模型客户端，统一模型调用入口
from app.lm.lm_utils import get_llm_client
# 提示词工具：加载本地prompt模板，实现提示词与代码解耦
from app.core.load_prompt import load_prompt

def step_1_data_validates(state):
    item_names = state["item_names"]
    rewritten_query=state["rewritten_query"]
    if not item_names or not rewritten_query:
        msg = "rewritten_query 或者 item_names 为空"
        logger.error(msg)
        raise ValueError(msg)

def step_2_get_llm_answer(state):
    rewritten_query=state["rewritten_query"]
    llm = get_llm_client("qwen3-vl-flash")
    prompt =  load_prompt("hyde_prompt", rewritten_query=rewritten_query)
    msg = HumanMessage(content=[prompt])
    
    chain = llm | StrOutputParser()

    llm_ans = chain.invoke(input=[msg])

    return  llm_ans

def step_3_query_hyde_embedding(llm_ans, state):
    item_names = state["item_names"]
    rewritten_query=state["rewritten_query"]

    embedding_str = f"{rewritten_query}{llm_ans}"
    result = generate_embeddings([embedding_str])
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

    pass 


def node_search_embedding_hyde(state):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """
    print("---HyDE 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 搜索假设性答案
    print("搜索架设性答案！！")
    step_1_data_validates(state)
    llm_ans = step_2_get_llm_answer(state)
    chunks = step_3_query_hyde_embedding(llm_ans, state)

    # ...
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    print("---HyDE 处理结束---")
    return {"hyde_embedding_chunks":chunks}

    
if __name__ == "__main__":
    # 本地测试代码
    print("\n" + "=" * 50)
    print(">>> 启动 node_search_embedding_hyde 本地测试")
    print("=" * 50)

    # 模拟输入状态
    mock_state = {
        "session_id": "test_hyde_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "item_names": ["HAK 180 烫金机"],
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_search_embedding_hyde(mock_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"HyDE Doc Generated: {bool(result.get('hyde_doc'))}")
        if result.get("hyde_doc"):
            print(f"Doc Preview: {result.get('hyde_doc')[:50]}...")

        chunks = result.get("hyde_embedding_chunks", [])
        print(f"Chunks Found: {len(chunks)} , chunks内容：{chunks}")
        if chunks:
            print(f"Top Chunk Score: {chunks[0].get('distance')}")
        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")