import mimetypes
from pathlib import Path
import sys
import re
from typing import Tuple,List, Dict
import os
import base64
from langchain_core.messages import HumanMessage

from app.conf import lm_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log
from app.import_process.agent.state import ImportGraphState
from app.lm.lm_utils import get_llm_client

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

def get_content(state: ImportGraphState) ->Tuple[str, Path, Path]:
    md_content = state["md_content"]
    md_path = state["md_path"]

    if not md_path:
        msg = f" md_path为空"
        logger.err(msg)
        return ValueError(msg)


    
    md_path_obj = Path(md_path)
    images_path_obj = md_path_obj.parent / "images"

    if not md_content:
        logger.warning("md_content为空")
        state["md_content"] = md_path_obj.read_text(encoding="utf-8")

    return (md_content, md_path_obj, images_path_obj)

def scan_images(md_content, images_path_obj)-> List[Tuple[str, str, Tuple[str, str]]]:

    
    image_context_list = []
    for image in images_path_obj.iterdir():
        image_name = image.name
        if not is_supported_image(image_name):
            logger.warning(f"{image_name}不是图片，无需处理")
            continue
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
        match_obj = pattern.search(md_content)

        if not match_obj:
            logger.warning(f"{image_name}不在本次的md中使用，跳过本次处理")
            continue
        start, end = match_obj.span()
        pre_context = md_content[max(start - 100, 0): start]
        pos_context = md_content[end: min(len(md_content), end + 100)]
        image_context_list.append((image_name, str(image), pre_context, pos_context))

    return image_context_list

def image_summary(image_context_list, stem) -> Dict[str, str]:

    image_summary_dict = {}

    print(lm_config.lv_model)
    vm = get_llm_client(lm_config.lv_model)

    for image_name, image_path_str, (pre_context, pos_context) in image_context_list:
        image_context_prompt = load_prompt("image_summary", root_folder = stem, image_content = [pre_context, pos_context])
        image_data = base64.b64encode(Path(image_path_str).read_bytes()).decode("utf-8")

        msg = HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mimetypes.guess_type(image_name)[0]};base64,{image_data}"
                    },
                },
                {"type": "text", "text": image_context_prompt},
            ])

        summary = vm.invoke(input=[msg])

        image_summary_dict[image_name] = summary

    return image_summary_dict

@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. (可选) 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    """

    (md_content, md_path_obj, images_path_obj) = get_content(state=state)

    if not images_path_obj.exists() or len(list(images_path_obj.iterdir())) == 0:
        logger.warning("md文件中没有任何图片")
        return state

    image_context_list = scan_images(md_content=md_content, images_path_obj=images_path_obj)

    image_summary_list = image_summary(image_context_list, md_path_obj.stem)

    return state


if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output/hak180产品安全手册", "hak180产品安全手册.md")
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
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")