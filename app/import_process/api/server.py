from mimetypes import guess_type
from pathlib import Path
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette import status
from starlette.middleware.cors import CORSMiddleware
from app.core.logger import logger, PROJECT_ROOT
from app.import_process.agent.state import get_default_state

from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.import_process.agent.main_graph import kb_import_app


app = FastAPI(title="import service", description="导入文件处理")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def invoke_import_graph(task_id: str, local_dir: str, local_file_path: str):
    try:
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        state = get_default_state()
        state["task_id"] = task_id
        state["local_dir"] = local_dir
        state["local_file_path"] = local_file_path

        kb_import_app.invoke(state)
        update_task_status(task_id, TASK_STATUS_COMPLETED)

    except Exception as e:
        update_task_status(task_id, TASK_STATUS_FAILED)
        logger.error(e)

@app.get("/import/html")
def return_import_html():

    html_file_path = PROJECT_ROOT / "app" / "import_process" / "page" / "import.html"

    if not html_file_path.exists():
        logger.error(f"{html_file_path} does not exist")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="页面文件不存在")

    return FileResponse(
        path= html_file_path,
        media_type=guess_type(html_file_path)[0],
    )


@app.post("/upload")
async def upload_file(files: List[UploadFile], background_tasks: BackgroundTasks):
    task_ids = []
    base_dir_obj = PROJECT_ROOT / "output" / datetime.now().strftime("%Y%m%d%")
    for file in files:
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        local_dir_obj = base_dir_obj / task_id
        local_file_path = local_dir_obj / file.filename
        local_file_path.parent.mkdir(parents=True, exist_ok=True)
        content = await file.read()
        local_file_path.write_bytes(content)
        background_tasks.add_task(invoke_import_graph,
                                  task_id=task_id,
                                  local_dir=str(local_dir_obj),
                                  local_file_path=str(local_file_path)
                                  )

    return {
        "code": 200,
        "message": "success",
        "task_ids": task_ids,
    }

@app.get("/status/{task_id}")
def return_status(task_id: str):
    task_status_info = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    }

    logger.info(task_status_info)

    return task_status_info

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)