import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

from typing import Tuple
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import add_running_task, add_done_task
from app.conf.mineru_config import mineru_config



"""
    节点作用: node_pdf_to_md  将pdf转成md,并且保存和存储,同时修改state相关的参数
    入参:  [pdf_path:str :Path   local_dir:str :Path 默认的存储文件地址(项目/output) ]
    出参:  [md_path:str  md_content:str]
    步骤:
       1. 日志+进行中的任务记录 add_running_task
       2. step_1_validate_paths 校验pdf和输出地址
       3. step_2_upload_and_poll minerU进行交互
       4. step_3_download_and_extract 下载提取和解压
       5. 根据md地址读取对应md_content内容,并且更新state
       6. 日志+完成的任务记录  add_done_task
"""
@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    """
    # 1. 日志+进行中的任务记录 add_running_task
    add_running_task(state["task_id"], "node_entry")

    # 2. step_1_validate_paths 校验pdf和输出地址
    pdf_path_obj, local_dir_obj = validate_paths(state)

    # 3. step_2_upload_and_poll minerU进行交互
    zip_url = upload_and_poll(pdf_path_obj)

    # 4. step_3_download_and_extract 下载提取和解压
    md_path_obj = download_and_extract(Path(zip_url), local_dir_obj, pdf_path_obj)


    # 5. 根据md地址读取对应md_content内容,并且更新state
    md_content = md_path_obj.read_text(encoding='utf-8')
    state["md_content"] = md_content

    # 6. 日志+完成的任务记录  add_done_task
    add_done_task(state["task_id"],"node_entry")
    return state


"""
    step_1_validate_paths 校验pdf和输出地址
      入参: state
      出参: pdf_path_obj [Path]  local_dir_obj [Path]
      步骤:
         1. state获取对应的地址
         2. 进行非空校验(pdf_path -> none -> 结束 | local_dir 给与默认地址)
         3. 将两个参数转成Path (str -> Path )
         4. 判断pdf_path_obj是否有文件,local_dir_path 是否存在文件夹
            没有文件->抛出异常
            没有文件夹 -> 创建文件mkdir
         5. 返回两个路径地址
"""
def validate_paths(state: ImportGraphState) -> Tuple[Path, Path]:
    
    pdf_path = state["pdf_path"]
    local_dir = state["local_dir"]

    if not pdf_path:
        logger.error("pdf_path 为空，无法读取文件，")
        raise ValueError("pdf_path 为空，无法读取文件，")
    
    if not local_dir:
        logger.warning("没有传入local_dir, 给予默认值")
        local_dir = PROJECT_ROOT / "output"
    
    state["local_dir"] = str(local_dir)

    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)

    if not pdf_path_obj.exists():
        logger.error(f"pdf_path {pdf_path} 不存在")
        raise FileNotFoundError(f"pdf_path {pdf_path} 不存在")


    if not local_dir_obj.exists():
        logger.warning("没有传入local_dir_obj, 默认创建")
        local_dir_obj.mkdir(parents=True, exist_ok=True)

    return (pdf_path_obj, local_dir_obj)

"""
    step_2_upload_and_poll minerU进行交互
       入参: pdf_path_obj
       出参: zip_url (str)
       步骤:
         1. 参数校验 (minerU -> 检查下miner url和key)
         2. 申请上传地址 (minerU) [batch_id]
         3. 向执行地址进行上传文件
         4. 轮询获取返回结果(zip_url) [batch_id]
         5. 返回zip_url
"""
def upload_and_poll(pdf_path_obj)-> str:
    base_url = mineru_config.base_url
    api_key = mineru_config.api_key
    if not base_url or not api_key:
        logger.error("minerU配置错误，请检查minerU配置")
        raise ValueError("minerU配置错误，请检查minerU配置")
   
    url =  f"{base_url}/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "files": [
            {"name":f"{pdf_path_obj.name}"}
        ],
        "model_version":"vlm"
    }

    response = requests.post(url,headers=header,json=data)

    http_status_code = response.status_code
    if not http_status_code == 200:
        msg = f"申请上传地址失败，返回状态码为:{http_status_code}, 请检查minerU的配置"
        logger.error(msg)
        raise RuntimeError(msg)

    result_dict = response.json()

    code = result_dict['code']

    if not code == 0:
        err_msg = result_dict['msg']
        msg = f"申请地址网络状态成功，但是业务失败，错误码为:{code},报错信息:{err_msg}"
        logger.error(msg)
        raise RuntimeError(msg)

    batch_id = result_dict['data']['batch_id']
    file_upload_url = result_dict['data']['file_urls'][0]

    # with open(pdf_path_obj, 'rb') as f:
    
    file_bytes = pdf_path_obj.read_bytes()


    # 避免代理污染网络请求
    with requests.Session() as session:
        session.trust_env = False
        upload_res = session.put(file_upload_url, data=file_bytes)
        http_status_code = upload_res.status_code
        if not http_status_code == 200:
            msg = f"上传文件失败，返回状态码为:{http_status_code}, 请检查minerU的配置"
            logger.error(msg)
            raise RuntimeError(msg)


    poll_url =  f"{base_url}/extract-results/batch/{batch_id}"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    timeout = 600 # 超时时间
    interval_time = 3 # 轮询时间

    start_time = time.time()

    while True:
        now_time = time.time()
        if now_time - start_time >= timeout:
            logger.error("minerU，文件状态轮询超时")
            raise TimeoutError("minerU，文件状态轮询超时")
        try:
            poll_res = requests.get(poll_url, headers=header)
        except Exception as e:
            logger.warning('轮询请求异常')
            time.sleep(interval_time)
            continue

        http_poll_res_code = poll_res.status_code

        if http_poll_res_code != 200:
            if 500 < http_poll_res_code and http_poll_res_code < 600:
                logger.warning("minerU，文件状态轮询错误，可恢复的状态码{http_poll_res_code}")
                time.sleep(interval_time)
            else:
                msg = f"minerU，文件状态轮询错误，状态码{http_poll_res_code}"
                logger.error(msg)
                raise RuntimeError(msg)
                
        else:
            poll_res_dict = poll_res.json()
            if poll_res_dict['code'] != 0:
                msg = f"minerU，文件状态轮询错误，状态码{poll_res_dict['code'] }"
                logger.error(msg)
                raise RuntimeError(msg)
            else:
                extract_result = poll_res_dict["data"]["extract_result"][0]
                if extract_result["state"] == 'done':
                    extract_result_url = extract_result["full_zip_url"]
                    if not extract_result_url:
                        msg = f"已经完成解析，但是url为空"
                        logger.error(msg)
                        raise RuntimeError(msg)
                    return extract_result_url
                elif extract_result['state'] == 'failed':
                    msg = f"已经完成了解析，但是失败了，失败信息：{extract_result}"
                    logger.error(msg)
                    raise RuntimeError("msg")
                    break
                else:
                    logger.info(f"解析进行中，{extract_result['state']}")
                    time.sleep(interval_time)
                    continue
            pass
        pass
    pass


"""
 step_3_download_and_extract 下载提取和解压
       入参: zip_url local_dir_path  pdf_path_obj/pdf_path_obj.stem
       出参: 新的md_path_obj [Path]
       步骤:
         1. 向指定zip地址发起请求获取响应response
         2. 将响应数据写到本地磁盘 [local_dir_path/pdf_path_obj.stem/stem.zip]
         3. 先清空解压文件夹的原文件
         4. 再次解压即可(避免出现脏数据)
         5. 检查是否存在md文件
         6. 进行md文件的命名确定 [xx.pdf -> full.md -> xx.md]
         7. 返回md_path_obj地址
"""
def download_and_extract(zip_url, local_dir_path_obj, stem):

    response = requests.get(zip_url, timeout=30)

    md_path_dir = local_dir_path_obj / f"{stem}_result.zip"

    md_path_dir.write_bytes(response.content)

    extract_path_obj = local_dir_path_obj / stem
    if extract_path_obj.exits():
        shutil.rmtree(extract_path_obj)

    extract_path_obj.mkdir(parents=True, exist_ok=True)

    shutil.unpack_archive(md_path_dir, extract_path_obj)





    pass


if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")
