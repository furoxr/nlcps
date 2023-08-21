from typing import List, Optional

from pydantic.v1 import BaseModel, Field, validator


class AnalysisResult(BaseModel):
    entities: List[str] = Field(description="Entities including in the user utterance")
    thoughts: str = Field(
        description="Thoughts is the reasoning steps you made to decide whether context is needed"
    )
    need_context: bool = Field(
        description="Whether context is needed to fulfill the task described in the utterance"
    )

    @validator("entities")
    def entities_validator(cls, v):
        if len(v) == 0:
            raise ValueError("Entities cannot be empty")
        return v


class AnalysisExample(BaseModel):
    utterance: str = Field(description="User utterance")
    analysis_result: AnalysisResult = Field(description="Analysis result")


class BaseIdModel(BaseModel):
    id: Optional[str] = Field(description="Unique id of the point in qdrant")


class RetrieveExample(BaseIdModel):
    user_utterance: str = Field(description="User utterance")
    code: str = Field(description="Generated DSL code")
    context: Optional[str] = Field(description="Context of the example")
    entities: List[str] = Field(description="Entities of the user utterance")


class DSLSyntaxExample(BaseIdModel):
    entities: List[str]
    code: str


class DSLRuleExample(BaseIdModel):
    entities: List[str]
    rule: str
