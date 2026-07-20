from langgraph.graph import StateGraph, END

from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger

query_graph = StateGraph(QueryGraphState)
query_graph.add_node("node_item_name_confirm", node_item_name_confirm)
query_graph.add_node("node_rerank", node_rerank)
query_graph.add_node("node_rrf", node_rrf)
query_graph.add_node("node_search_embedding", node_search_embedding)
query_graph.add_node("node_search_embedding_hyde", node_search_embedding_hyde)
query_graph.add_node("node_web_search_mcp", node_web_search_mcp)
query_graph.add_node("node_answer_output", node_answer_output)

query_graph.set_entry_point("node_item_name_confirm")

def node_item_name_confirm_after_router(state: QueryGraphState):
    if not state["answer"]:
        return "node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"
    else:
        logger.warning(f"无法继续向后执行")
        return "node_answer_output"

    pass

query_graph.add_conditional_edges("node_item_name_confirm", node_item_name_confirm_after_router, {
    "node_search_embedding": "node_search_embedding",
    "node_search_embedding_hyde": "node_search_embedding_hyde",
    "node_web_search_mcp": "node_web_search_mcp",
})

query_graph.add_edge("node_search_embedding", "node_rrf")
query_graph.add_edge("node_search_embedding_hyde", "node_rrf")
query_graph.add_edge("node_web_search_mcp", "node_rrf")

query_graph.add_edge("node_rrf", "node_rerank")
query_graph.add_edge("node_rerank", "node_answer_output")

query_graph.add_edge("node_answer_output", END)

query_app = query_graph.compile()