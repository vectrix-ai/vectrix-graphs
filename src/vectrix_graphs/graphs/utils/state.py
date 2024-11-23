import operator
from typing import Annotated, List, Literal, Sequence

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from .models.tools import CitedSources


class QuestionState(TypedDict):
    question: str


class OverallState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    temporary_answer: str
    intent: Literal[
        "specific_question", "greeting", "metadata_query", "follow_up_question"
    ]
    question_list: List[str]
    documents: Annotated[List[Document], operator.add]
    cited_sources: List[CitedSources]
    hallucination_grade: bool


class SubgraphState(TypedDict):
    answer: str
    question: str
    documents: Annotated[List[Document], operator.add]
