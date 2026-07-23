import json
import time
import sys
import asyncio
from app.utils.task_utils import add_done_task,add_running_task
from agents.mcp import MCPServerStreamableHttp # pip install openai-agents
from app.core.logger import logger, node_log, step_log
from app.conf.bailian_mcp_config import mcp_config
from dotenv import load_dotenv

DASHSCOPE_BASE_URL_STREAM_ABLE_HTTP = "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp"
DASHSCOPE_API_KEY = mcp_config.api_key

def step_1_data_validates(state):
    rewritten_query = state["rewritten_query"]
    if not rewritten_query:
        msg = f"rewritten_query  为空"
        logger.error(msg)
        raise ValueError(msg)


async def step_2_web_search_mcp(rewritten_query, limit=5):

    mcp_server = MCPServerStreamableHttp(
        name="search_mcp", # 随便写
        params={
            "url": DASHSCOPE_BASE_URL_STREAM_ABLE_HTTP, #
            "headers": {"Authorization": DASHSCOPE_API_KEY},
            "timeout": 300,
            "sse_read_timeout": 300
        })
    try:
        await mcp_server.connect()
        tool_list =  await mcp_server.list_tools()
        logger.info(f"工具列表:{tool_list}")
        mcp_result = await mcp_server.call_tool(
            tool_name="bailian_web_search",
            arguments={
                "query": rewritten_query,
                "count": limit
            }
        )
        return mcp_result
    finally:
        # 4.释放本次链接资源
        await mcp_server.cleanup()


def step_3_get_web_search_docs(mcp_result):
    text_dict = json.loads(mcp_result.content[0].text)
    pages = text_dict["pages"]
    return pages



def node_web_search_mcp(state):
    """
    节点功能，调用外部搜索引擎补充信息
    :param state:
    :return:
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])
    print("---node-web-search-mcp处理---")

    add_done_task(state["session_id"],sys._getframe().f_code.co_name,state["is_stream"])
    step_1_data_validates(state)
    mcp_result = asyncio.run(step_2_web_search_mcp(state["rewritten_query"]))
    pages = step_3_get_web_search_docs(mcp_result=mcp_result)
    # 调用mcp外部引擎
    print(f"调用外部mcp引擎")

    print("---node-web-search-mcp处理结束---")
    return {"web_search_docs": pages}

if __name__ == '__main__':
    load_dotenv()
    test_state = {
        "session_id":"xxxx",
        "is_stream":False,
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置"
    }

    # 调用 websearch_node 函数
    result_state = node_web_search_mcp(test_state)

    # 验证结果
    print("测试结果:")
    print(f"查询内容: {test_state.get('rewritten_query')}")
    print(f"查询内容: {result_state}")