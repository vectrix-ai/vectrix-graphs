from typing import List, Annotated, Sequence, Literal, Optional
from typing_extensions import TypedDict
from pydantic import Field, BaseModel
import operator
from enum import Enum
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages



class IntentEnum(str, Enum):
    GREETING = "greeting"
    SPECIFIC_QUESTION = "specific_question"
    METADATA_QUERY = "metadata_query"
    FOLLOW_UP_QUESTION = "follow_up_question"
    
class Intent(BaseModel):
    intent: IntentEnum

class QuestionList(BaseModel):
    questions : List[str]

class QuestionState(TypedDict):
    question: str

class CitedSources(BaseModel):
    source: str = Field(
        description="The source of the information"
    )
    url: str = Field(
        description="The URL associated with the source"
    )
    source_type: str = Field(
        description="The type of source"
    )

class OverallState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    temporary_answer: str
    intent: Literal["specific_question", "greeting", "metadata_query", "follow_up_question"]
    question_list: List[str]
    documents: Annotated[List[Document], operator.add]
    cited_sources: List[CitedSources]
    hallucination_grade: bool