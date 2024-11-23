from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class IntentEnum(str, Enum):
    GREETING = "greeting"
    SPECIFIC_QUESTION = "specific_question"
    METADATA_QUERY = "metadata_query"
    FOLLOW_UP_QUESTION = "follow_up_question"


class Intent(BaseModel):
    intent: IntentEnum


class QuestionList(BaseModel):
    questions: List[str]


class CitedSources(BaseModel):
    source: str = Field(description="The source of the information")
    url: str = Field(description="The URL associated with the source")
    source_type: str = Field(description="The type of source")
