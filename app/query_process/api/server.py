from pathlib import Path
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from app.core.logger import logger, PROJECT_ROOT

from starlette import status

from mimetypes import guess_type

from app.query_process.agent.state import create_query_default_state
from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.query_process.agent.main_graph import query_app

class Query(BaseModel):
    query: str
    session_id: str
    is_stream: bool

# 定义fastapi对象
app = FastAPI(title="query service", description="掌柜智库查询服务！")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/query/html")
def return_query_html():
    html_file_path = PROJECT_ROOT / "app" / "query_process" / "page" / "chat.html"

    if not html_file_path.exists():
        logger.error(f"{html_file_path} does not exist")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="页面文件不存在")

    return FileResponse(
        path= html_file_path,
        media_type=guess_type(html_file_path)[0],
    )

@app.get('/health')
def health():
    return {"ok": True}



def run_query_graph(query_req: Query):
    try:
        query = query_req.query
        session_id = query_req.session_id
        is_stream = query_req.is_stream

        # 清空原有的任务列表
        clear_task(session_id)

        update_task_status(session_id, TASK_STATUS_PROCESSING, push_queue=is_stream)

        initial_state = create_query_default_state(session_id=session_id,
                                                   original_query=query,
                                                   is_stream=is_stream
                                                   )
        query_app.invoke(initial_state)
        update_task_status(session_id, TASK_STATUS_COMPLETED, push_queue=is_stream)


        image_urls = [""]
        push_to_session(
            session_id,
            SSEEvent.FINAL,
            {
                "answer": get_task_result(session_id,"answer"),
                "status": "completed",
                "image_urls": image_urls
            }
        )

    except Exception as e:
        update_task_status(session_id, TASK_STATUS_FAILED, push_queue=True)

        logger.error(e)



@app.post("/query")
async def query(query_req: Query, background_tasks: BackgroundTasks):
    query = query_req.query
    session_id = query_req.session_id
    is_stream = query_req.is_stream


    if is_stream:

        create_sse_queue(session_id)
        background_tasks.add_task(run_query_graph, query_req)
        return {
            "message": "结果处理中",
            "session_id": session_id,
        }
        pass
    else:

        run_query_graph(query_req)
        answer = get_task_result(session_id, "answer")
        return {
            "message": "结果处理完毕",
            "session_id": session_id,
            "answer": answer,
            "done_list": get_done_task_list(session_id)
        }
        pass

    logger.info(query)
    pass



@app.get("/stream/{session_id}")
async def sse(session_id: str, request: Request):
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type = "text/event-stream",
    )


    pass


@app.get('/history/{session_id}')
def get_history(session_id: str, limit: int=10):
    msgList = get_recent_messages(session_id, limit)
    items = []
    for r in msgList:
        items.append({
            "_id": str(r.get("_id")) if r.get("_id") is not None else "",
            "session_id": r.get("session_id", ""),
            "role": r.get("role", ""),
            "text": r.get("text", ""),
            "rewritten_query": r.get("rewritten_query", ""),
            "item_names": r.get("item_names", []),
            "ts": r.get("ts")
        })
    return {
        "session_id": session_id,
        "items": items,
    }

@app.delete('/history/{session_id}')
def delete_history(session_id: str, request: Request):
    delete_count = clear_history(session_id)
    return {
        "message": f"删除:{session_id}, 删除了{delete_count}条",
        "session_id": session_id
    }

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8001)