from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from vectrix_graphs.graphs.nodes.multi_modal_rag import (
    MultiModalRetrievalState,
    RAGNodes,
)
from vectrix_graphs.logger import setup_logger

logger = setup_logger(__name__, level="INFO")


# Define the config
class GraphConfig(TypedDict):
    mode: str = "online"
    collection_name: str
    include_images: bool = False


graph_nodes = RAGNodes(logger)


graph = StateGraph(MultiModalRetrievalState, config_schema=GraphConfig)

graph.add_node("multi_modal_retrieval", graph_nodes.multi_modal_retrieval)
graph.add_node("answer_question", graph_nodes.answer_question)

graph.add_edge(START, "multi_modal_retrieval")
graph.add_edge("multi_modal_retrieval", "answer_question")
graph.add_edge("answer_question", END)

multi_modal_graph = graph.compile()

__all__ = ["multi_modal_graph"]
