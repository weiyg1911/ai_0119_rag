import json
import os
import re
import sys

from pathlib import Path

from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ====================== 全局配置（可根据模型调整）======================
# 单个文本块最大长度（控制不超过模型上下文）
CHUNK_SIZE = 200 # 小值方便测试切割
# 块之间重叠长度（保证语义不丢失）
CHUNK_OVERLAP = 20

CHUNK_MAX_SIZE = 500

def is_title(inputStr:str):
    reg = re.compile(r"^\s*#{1,6}\s.+")
    return bool(reg.match(inputStr))

def is_code_tag(inputStr:str):
    return  inputStr.startswith("```") or inputStr.startswith("~~~")

@step_log("step_1_check_value 校验并规范化md内容")
def step_1_check_value(state: ImportGraphState):
    md_content = state["md_content"]
    file_title = state["file_title"]
    if not md_content:
        logger.warning("当前md_content为空")
        md_path = state["md_path"]
        md_content = Path(md_path).read_text()
        if not md_content:
            msg = f"md_content为空，从{md_path}中读取content也为空"
            logger.error(msg)
            raise ValueError(msg)

    if not file_title:
        file_title = Path(md_path).stem
        if not file_title:
            file_title = 'default'

    md_content = md_content.replace('\r\n', '\n').replace('\r', '\n')

    state["file_title"] = file_title
    state["md_content"] = md_content



@step_log("step_2_content_to_chunks 按标题切分chunks")
def step_2_content_to_chunks(state: ImportGraphState):
    file_title = state["file_title"]
    content_list = state["md_content"].split("\n")
    chunk_list = []
    current_title = ""
    current_content_list = []
    is_in_code = False
    for content in content_list:
        if is_title(content) and not is_in_code:
            if not current_title:
                current_title = content
                continue
            chunk_dist = {
                "title": current_title,
                "content": '\n'.join(current_content_list),
                'file_title': file_title
            }
            chunk_list.append(chunk_dist)
            current_title = content
            current_content_list = [current_title]
        else:
            current_content_list.append(content)
            if is_code_tag(content):
                is_in_code = not is_in_code
    if not current_title:
        chunk_dist = {
            "title": file_title if len(current_title) == 0 else current_title,
            "content": ''.join(current_content_list),
            'file_title': file_title
        }
        chunk_list.append(chunk_dist)

    if len(chunk_list) == 0:
        chunk_dist = {
            "title": file_title if len(file_title) == 0 else current_title,
            "content": state["md_content"],
            "file_title": file_title
        }
        chunk_list.append(chunk_dist)

    state["chunks"] = chunk_list
    logger.info(f"chunk切割完成, chunk-size: {len(chunk_list)}, 前三个chunk: {chunk_list[0:3]}")
    return chunk_list


@step_log("step_3_chunks_detail 过长chunk二次切分")
def step_3_chunks_detail(chunk_list, state: ImportGraphState):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # 切割优先级：段落 → 换行 → 句子 → 空格
        separators=["\n\n", "\n", "。", "！", "；", " "]
    )
    final_chunk_list = []
    for chunk in chunk_list:
        content = chunk["content"]
        file_title = chunk["file_title"]
        title = chunk["title"]
        if len(content) > CHUNK_MAX_SIZE:
            res = text_splitter.split_text(content)
            for idx, item in enumerate(res, start=1):
                chunk_dist = {
                    "file_title": file_title,
                    "parent_title": title,
                    "part": idx,
                    "content": item,
                    "title": f"{title}-{idx}",
                }
                final_chunk_list.append(chunk_dist)
        else:
            chunk_dist = {
                "title": title,
                "file_title": file_title,
                "parent_title": title,
                "part": 1,
                "content": content
            }
            final_chunk_list.append(chunk_dist)

    state["chunks"] = final_chunk_list

    return final_chunk_list

@step_log("step_4_save_chunks 落盘保存chunks")
def step_4_save_chunks(chunk_list, state: ImportGraphState):
    md_path = state["md_path"]
    md_path_obj = Path(md_path)
    new_chunks_obj = md_path_obj.with_name(f"chunks_{md_path_obj.stem}.json")
    # 备份
    new_chunks_obj.write_text(json.dumps(chunk_list, ensure_ascii=False, indent=4))

@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    未来要实现:
    1. 基于 Markdown 标题层级进行递归切分。
    2. 对过长的段落进行二次切分。
    3. 生成包含 Metadata (标题路径) 的 Chunk 列表。
    """
    add_running_task(state['task_id'], "node_document_split")
    step_1_check_value(state)
    chunk_list = step_2_content_to_chunks(state)
    final_chunk_list = step_3_chunks_detail(chunk_list, state)
    step_4_save_chunks(final_chunk_list, state)

    add_done_task(state['task_id'], "node_document_split")
    return state


if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output/hak180产品安全手册", "hak180产品安全手册_new.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir":os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")