import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType

from app.conf.milvus_config import milvus_config
# 导入自定义模块：
# 1. 流程状态载体：ImportGraphState为LangGraph流程的统一状态管理对象
from app.import_process.agent.state import ImportGraphState
# 2. Milvus工具：获取单例Milvus客户端，实现连接复用
from app.clients.milvus_utils import get_milvus_client
# 3. 大模型工具：获取大模型客户端，统一模型调用入口
from app.lm.lm_utils import get_llm_client
# 4. 向量工具：BGE-M3模型实例、向量生成方法（稠密+稀疏向量）
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
# 5. 稀疏向量工具：归一化处理，保证向量长度为1，提升检索准确性
from app.utils.normalize_sparse_vector import normalize_sparse_vector
# 6. 任务工具：更新任务运行状态，用于任务监控和管理
from app.utils.task_utils import add_running_task, add_done_task
# 7. 日志工具：项目统一日志入口，分级输出（info/warning/error）
from app.core.logger import logger,node_log,step_log
# 8. 提示词工具：加载本地prompt模板，实现提示词与代码解耦
from app.core.load_prompt import load_prompt

from app.utils.escape_milvus_string_utils import escape_milvus_string

# --- 配置参数 (Configuration) ---
# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 10000

def step1_check_content(state: ImportGraphState):
    chunks = state["chunks"]
    file_title = state["file_title"]

    if not chunks:
        msg = "chunks没有内容，无法进行操作"
        logger.error(msg)
        raise ValueError(msg)

    if not file_title:
        logger.warning("fileTitle不存在")
        file_title = 'default fileTitle'
        state["file_title"] = file_title

    return chunks, file_title

def step2_get_item_name_from_modal(state: ImportGraphState):
    file_title = state["file_title"]

    part_chunks = state["chunks"][0: 5]
    prompt_list = []
    for idx, chunk in enumerate(part_chunks, start=1):
        prompt_list.append(f"切片:{idx}, 标题:{chunk['title']},内容：{chunk['content']}")
    final_chunks_str = '\n'.join(prompt_list)[0: CONTEXT_TOTAL_MAX_CHARS]

    system_prompt = load_prompt("product_recognition_system")
    system_msg = SystemMessage(content=[system_prompt])
    prompt =  load_prompt("item_name_recognition", file_title=file_title, context=final_chunks_str)
    msg = HumanMessage(content=[prompt])

    item_name_modal = get_llm_client("qwen3-vl-flash")

    chain = item_name_modal | StrOutputParser()

    item_name = chain.invoke(input=[system_msg,msg])

    if not item_name:
        return file_title
    return item_name

def _create_collection(client: get_milvus_client):

    schema = client.create_schema(
        auto_id=True,
        enable_dynamic_fields=True,
    )

    schema.add_field('pk', datatype=DataType.INT64, is_primary=True)
    schema.add_field('file_title', datatype=DataType.VARCHAR, max_length=512)
    schema.add_field('item_name', datatype=DataType.VARCHAR, max_length=512)
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
        collection_name = milvus_config.item_name_collection,
        schema=schema,
        index_params=index_params,
    )

def step_4_invert_item_name(item_name:str, file_title:str, dense_vector, sparse_vector):
    # 连接milvus的客户端
    # 创建表对应的schema
    # 创建列对应的索引

    client = get_milvus_client()

    if not client.has_collection(collection_name=milvus_config.item_name_collection):
        _create_collection(client)

    client.delete(collection_name=milvus_config.item_name_collection, filter=f"item_name=='{item_name}'")

    data = [
        {
            "file_title": file_title,
            "item_name": item_name,
            "dense_vector": dense_vector,
            "sparse_vector": sparse_vector,
        }
    ]

    client.insert(collection_name=milvus_config.item_name_collection, data=data)




    pass

@node_log("node_item_name_recognition")
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    为什么叫这个名字: 识别文档核心描述的物品/商品名称 (Item Name)。
    未来要实现:
    1. 取文档前几段内容。
    2. 调用 LLM 识别这篇文档讲的是什么东西 (如: "Fluke 17B+ 万用表")。
    3. 存入 state["item_name"] 用于后续数据幂等性清理。
    """

    step1_check_content(state)
    item_name = step2_get_item_name_from_modal(state)
    state["item_name"] = item_name
    chunks = state["chunks"]
    for chunk in chunks:
        chunk['item_name'] = item_name

    result = generate_embeddings([item_name])

    dense_vector = result["dense"][0]
    sparse_vector = result["sparse"][0]

    step_4_invert_item_name(item_name=item_name, file_title=item_name, dense_vector=dense_vector, sparse_vector=sparse_vector)

    return state


# ===================== 本地测试方法（直接运行调试，无需启动LangGraph） =====================
def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
        2. 确保大模型、Milvus、BGE-M3服务均可正常访问
        3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
        直接运行该函数：if __name__ == "__main__": test_node_item_name_recognition()
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

        # 4. 验证Milvus存储（可选）

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()