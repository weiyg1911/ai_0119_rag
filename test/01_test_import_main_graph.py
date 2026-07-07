import json

from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import create_default_state
import sys
from app.core.logger import logger

state = create_default_state(task_id="001", local_file_path="xxx.pdf")

result = kb_import_app.invoke(state)
logger.info("执行结果",json.dumps(result, indent=4, ensure_ascii=False))
logger.info("图结构:")

logger.info(kb_import_app.get_graph().print_ascii())