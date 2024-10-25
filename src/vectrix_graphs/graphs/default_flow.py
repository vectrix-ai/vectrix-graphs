from typing import TypedDict, Literal
from vectrix_graphs.graphs.utils.nodes import GraphNodes
from vectrix_graphs.graphs.utils.state import OverallState
from vectrix_graphs import vectordb
from langgraph.graph import StateGraph, START, END
import pathlib
from vectrix_graphs.logger import setup_logger

logger = setup_logger(name="LangGraph Flow")

# Define the config
class GraphConfig(TypedDict):
    internet_search: bool

graph_nodes = GraphNodes(logger, vectordb, mode="online")

subgraph = StateGraph(OverallState, config_schema=GraphConfig)

subgraph.add_node("split_questions", graph_nodes.split_question_list)
subgraph.add_node("retrieve", graph_nodes.retrieve)
subgraph.add_node("rag_answer", graph_nodes.rag_answer)
subgraph.add_node("filter_docs", graph_nodes.filter_docs)
subgraph.add_node("hallucination_grader", graph_nodes.hallucination_grader)
subgraph.add_node("final_answer", graph_nodes.final_answer)
subgraph.add_node("rewrite_question", graph_nodes.rewrite_question)

subgraph.add_edge(START, "split_questions")
subgraph.add_conditional_edges("split_questions", graph_nodes.retrieve_documents, ["retrieve"])
subgraph.add_edge("retrieve", "filter_docs")
subgraph.add_edge("filter_docs", "rag_answer")
subgraph.add_edge("rag_answer", "hallucination_grader")
subgraph.add_conditional_edges("hallucination_grader", graph_nodes.grade, {"no_hallucinations": "final_answer", "hallucinations": "rewrite_question"})
subgraph.add_edge("rewrite_question", "split_questions")
subgraph.add_edge("final_answer", END)

subgraph = subgraph.compile()

workflow = StateGraph(OverallState, config_schema=GraphConfig)

# Define the nodes
workflow.add_node("detect_intent", graph_nodes.detect_intent)
workflow.add_node("llm_answer", graph_nodes.llm_answer)
workflow.add_node("question_subgraph", subgraph)
workflow.add_node("metadata_query", graph_nodes.metadata_query)
workflow.add_node("default_flow_end", graph_nodes.default_flow_end)
# Define the flow
workflow.add_edge(START, "detect_intent")
workflow.add_conditional_edges(
    "detect_intent",
    graph_nodes.decide_answering_path,
    {
        "greeting": "llm_answer",
        "specific_question": "question_subgraph",
        "metadata_query": "metadata_query",
        "follow_up_question": "default_flow_end"
    }
)
workflow.add_edge("llm_answer", END)
workflow.add_edge("question_subgraph", END)
workflow.add_edge("metadata_query", END)
workflow.add_edge("default_flow_end", END)
default_flow = workflow.compile()

__all__ = ['default_flow']