import mimetypes
from pathlib import Path
import sys
import re
from typing import Tuple,List, Dict
import os
import base64

from langchain_community.tools.file_management import delete
from langchain_core.messages import HumanMessage
from minio import Minio
from minio.deleteobjects import DeleteObject
from sqlalchemy.orm.persistence import delete_obj

from app.conf.lm_config import lm_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.lm.lm_utils import get_llm_client
from app.utils.minio_utils import get_minio_client
from app.conf.minio_config import minio_config
from app.utils.task_utils import add_running_task, add_done_task

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

@step_log("get_content 获取md内容与图片目录")
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
        md_content = state["md_content"]

    return (md_content, md_path_obj, images_path_obj)

@step_log("scan_images 扫描md中的图片及上下文")
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

@step_log("image_summary 多模态生成图片摘要")
def image_summary(image_context_list, stem) -> Dict[str, str]:

    image_summary_dict = {}

    print(lm_config.lv_model)
    vm = get_llm_client(lm_config.lv_model)

    for (image_name, image_path_str,pre_context, pos_context) in image_context_list:
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

        image_summary_dict[image_name] = summary.content

        logger.info(f'summary:{summary}')

    return image_summary_dict

@step_log("upload_and_replace 上传图片并替换md链接")
def upload_and_replace(image_context_list, image_summery_dict, md_content, stem) -> str:
    minio_client = get_minio_client()
    list_objs = minio_client.list_objects(bucket_name=minio_config.bucket_name, prefix=f"{minio_config.minio_img_dir[1:]}/{stem}/", recursive=True)

    delete_obj_list = [DeleteObject(obj.object_name) for obj in list_objs]

    if list_objs:
        errors = minio_client.remove_objects(bucket_name=minio_config.bucket_name, delete_object_list=delete_obj_list)

        for err in errors:
            logger.warning(f"删除失败，原因是{err}")

        logger.debug("删除成功")

    image_minio_dict = {}

    for (image_name, image_path_str, pre_context, pos_context) in image_context_list:
        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=f"{minio_config.minio_img_dir}/{stem}/{image_name}",
                file_path=image_path_str,
                content_type=mimetypes.guess_type(image_name)[0])
            image_minio_url = f"http://{minio_config.endpoint}/{minio_config.bucket_name}{minio_config.minio_img_dir}/{stem}/{image_name}"
            logger.debug(f"图片{image_name}上传成功， 地址为{image_minio_url}")
            image_minio_dict[image_name] = image_minio_url
        except Exception as e:
            logger.warning(f"本次图片上传失败{image_path_str}")
            continue

    total_image_info = {}

    for image_name, minio_url in image_minio_dict.items():
        total_image_info[image_name] = (minio_url, image_summery_dict[image_name])


    for image_name,(image_url,image_summary) in total_image_info.items():
        rep = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + ".*?\)")
        logger.info(f"要替换md_content文档了image_summary:{image_summary}, image_url:{image_url}")
        md_content =  rep.sub(lambda _:f"![{image_summary}]({image_url})",md_content)
    # 6. 最终返回md_content
    return md_content



@step_log("backup_md 备份处理后的md文件")
def backup_md(new_md_content, md_path_obj):
    new_md_path_obj = md_path_obj.with_name(f"{md_path_obj.stem}_new.md")
    # 备份
    new_md_path_obj.write_text(new_md_content,encoding="utf-8")
    return str(new_md_path_obj)


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
    add_running_task(state["task_id"], "node_md_img")

    (md_content, md_path_obj, images_path_obj) = get_content(state=state)

    if not images_path_obj.exists() or len(list(images_path_obj.iterdir())) == 0:
        logger.warning("md文件中没有任何图片")
        add_done_task(state["task_id"], "node_md_img")
        return state

    image_context_list = scan_images(md_content=md_content, images_path_obj=images_path_obj)


    image_summary_dict = image_summary(image_context_list, md_path_obj.stem)

    new_md_content = upload_and_replace(image_context_list, image_summary_dict, md_content, md_path_obj.stem)

    new_md_path_str = backup_md(new_md_content, md_path_obj)

    add_done_task(state["task_id"], "node_md_img")
    return state




if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
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