import time
import sys
from urllib import response
import json

from app.clients.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from app.conf.milvus_config import milvus_config
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
from langchain_core.output_parsers import JsonOutputParser
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger, node_log, step_log
from app.clients.mongo_history_utils import save_chat_message, get_recent_messages
from langchain_core.messages import SystemMessage, HumanMessage

# 大模型工具：获取大模型客户端，统一模型调用入口
from app.lm.lm_utils import get_llm_client
# 提示词工具：加载本地prompt模板，实现提示词与代码解耦
from app.core.load_prompt import load_prompt

CHAT_HISTORY_LIMIT = 20

def step_1_data_validates(state):
    session_id = state["session_id"]
    original_query = state["original_query"]
    if not session_id or not original_query:
        msg = f"session_id 和 original_query 为空"
        logger.error(msg)
        raise ValueError(msg)

def step_2_chat_history(state):
    session_id = state["session_id"]
    messages = get_recent_messages(session_id, limit=CHAT_HISTORY_LIMIT)
    state["history"] = messages

def step_3_llm_item_names_and_rewrite(state):
    history_message_list = state["history"]
    original_query = state["original_query"]
    llm = get_llm_client("qwen3-vl-flash")

    prompt =  load_prompt("rewritten_query_and_itemnames", history_text=history_message_list, query=original_query)
    msg = HumanMessage(content=[prompt])
    chain = llm | JsonOutputParser()

    res = chain.invoke(input=[msg])


    item_names = res["item_names"]
    rewritten_query = res["rewritten_query"]
    state["item_names"] = item_names
    state["rewritten_query"] = rewritten_query
    logger.info(f"item_names: {item_names}")
    logger.info(f"rewritten_query: {rewritten_query}")
    
def step_4_vector_query_item_name(state):
    item_names = state["item_names"]
    vector_dict = {}
    result = generate_embeddings(item_names)
    milvus_client = get_milvus_client()


    for i in range(0, len(item_names)):
        dense_vector = result['dense'][i]
        sparse_vector = result["sparse"][i]
        reqs = create_hybrid_search_requests(dense_vector, sparse_vector)
        response = hybrid_search(
            client=milvus_client,
            collection_name=milvus_config.item_name_collection,
            reqs=reqs,
            ranker_weights=(0.8, 0.2),
            norm_score=True
        )
        current_item_name_list = []
        for item in response[0]:
            current_item_name_list.append({
                "item_name": item.get("entity", {}).get("item_name", ""),
                "score": item.get("distance", 0)
            })
        vector_dict[item_names[i]] = current_item_name_list

    logger.info(f"vector_dict: {vector_dict}")

    return vector_dict


def step_5_select_item_name_list(vector_dict):
    confirmed_item_name_list = []
    options_item_name_list = []
    for item_name, list in vector_dict.items():
        list.sort(key= lambda x: x["score"], reverse=True)
        high_list = [item for item in list if item["score"]>=0.65]
        low_list = [item for item in list if (item["score"]<0.65 and item["score"]>=0.5)]
        if len(high_list) > 0:
            confirmed_item_name_list.append(high_list[0]["item_name"])
            continue
        if len(low_list) > 0:
            options_item_name_list.extend([item["item_name"] for item in low_list[0:2]])

    return {
        "confirmed_item_name_list": confirmed_item_name_list,
        "options_item_name_list": options_item_name_list
    }

def step_6_deal_state(state, final_result):

    confirmed_item_name_list = final_result.get("confirmed_item_name_list", [])
    options_item_name_list = final_result.get("options_item_name_list", [])
    if len(confirmed_item_name_list) > 0:
        state["item_names"] = confirmed_item_name_list
        if "answer" in state:
            del state["answer"]
        return
    if len(options_item_name_list) > 0:
        option_names_str = '、'.join(options_item_name_list)
        state["answer"] = f"您想要咨询的是:{option_names_str}吗？"
        return 
    
    state["answer"] = "对不起，未找到相关的产品，请您明确后再尝试"



def node_item_name_confirm(state):
    """
    节点功能：确认用户问题中的核心商品名称。
    输入：state['original_query']
    输出：更新 state['item_names']
    """
    print(f"---node_item_name_confirm---开始处理")
    # 记录任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    # 后面会调用大模型，进行逻辑处理
    time.sleep(7)
    # 记录任务结束

    step_1_data_validates(state)
    step_2_chat_history(state)
    step_3_llm_item_names_and_rewrite(state)
    vector_dict = step_4_vector_query_item_name(state)
    final_result = step_5_select_item_name_list(vector_dict)
    step_6_deal_state(state, final_result)


    save_chat_message(
        session_id=state["session_id"], 
        role="user", 
        text=state["original_query"],
        rewritten_query=state["rewritten_query"],
        item_names = state["item_names"]
    )

    add_done_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    return state

if __name__ == "__main__":
    # 模拟输入状态
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "烫金机 和 苹果手机  好不好用?",
        "is_stream": False
    }

    print(">>> 开始测试 node_item_name_confirm...")
    try:
        # 运行节点
        result_state = node_item_name_confirm(mock_state)

        print("\n>>> 测试完成！最终状态:")
        print(json.dumps(result_state, indent=2, ensure_ascii=False))

        # 简单验证
        if result_state.get("item_names"):
            print(f"\n[PASS] 成功提取并确认商品名: {result_state['item_names']}")
        else:
            print(f"\n[WARN] 未确认到商品名 (可能是向量库无匹配或LLM未提取)")

    except Exception as e:
        print(f"\n[FAIL] 测试运行出错: {e}")
