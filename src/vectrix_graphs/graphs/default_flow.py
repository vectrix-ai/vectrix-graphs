from typing import TypedDict, Literal
from vectrix_graphs.graphs.utils.nodes import GraphNodes
from vectrix_graphs.graphs.utils.state import OverallState
from vectrix_graphs.db.vectordb import VectorDB
from langgraph.graph import StateGraph, START, END
import pathlib
from vectrix_graphs.logger import setup_logger

logger = setup_logger(name="LangGraph Flow")
vectordb = VectorDB(setup_logger(name="VectorDB"))

# Define the config
class GraphConfig(TypedDict):
    internet_search: bool

graph_nodes = GraphNodes(logger, vectordb, mode="online")

# Create a new workflow
workflow = StateGraph(OverallState, config_schema=GraphConfig)

# Define the nodes
workflow.add_node("detect_intent", graph_nodes.detect_intent)
workflow.add_node("llm_answer", graph_nodes.llm_answer)
workflow.add_node("split_questions", graph_nodes.split_question_list)
workflow.add_node("retrieve", graph_nodes.retrieve)
workflow.add_node("rag_answer", graph_nodes.rag_answer)
workflow.add_node("hallucination_grader", graph_nodes.hallucination_grader)
workflow.add_node("final_answer", graph_nodes.final_answer)
workflow.add_node("rewrite_question", graph_nodes.rewrite_question)


# Define the flow
workflow.set_entry_point("detect_intent")
workflow.add_edge(START, "detect_intent")
workflow.add_conditional_edges(
    "detect_intent",
    graph_nodes.decide_answering_path,
    {
        "greeting": "llm_answer",
        "specific_question": "split_questions",
        "metadata_query": END,
        "follow_up_question": END
    }
)
workflow.add_edge("llm_answer", END)
workflow.add_conditional_edges("split_questions", graph_nodes.retrieve_documents, ["retrieve"])
workflow.add_edge("retrieve", "rag_answer")
workflow.add_edge("rag_answer", "hallucination_grader")
workflow.add_conditional_edges(
    "hallucination_grader",
    graph_nodes.grade,
    {
        "no_hallucinations": "final_answer",
        "hallucinations": "rewrite_question"
    }
)
workflow.add_edge("rewrite_question", "split_questions")
workflow.add_edge("final_answer", END)

default_flow = workflow.compile()

__all__ = ['default_flow']