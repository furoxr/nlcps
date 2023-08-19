from typing import List, Optional, NewType

from pydantic import BaseModel, Field, field_validator


Entity = NewType("Entity", str)


class AnalysisResult(BaseModel):
    entities: List[Entity] = Field(
        description="Entities including in the user utterance"
    )
    thoughts: str = Field(
        description="Thoughts is the reasoning steps you made to decide whether context is needed"
    )
    need_context: bool = Field(
        description="Whether context is needed to fulfill the task described in the utterance"
    )

    @field_validator("entities")
    def entities_validator(cls, v):
        if len(v) == 0:
            raise ValueError("Entities cannot be empty")
        return v


class AnalysisExample(BaseModel):
    utterance: str = Field(description="User utterance")
    analysis_result: AnalysisResult = Field(description="Analysis result")


class RelatedSample(BaseModel):
    score: Optional[float] = Field(description="Score of the sample")
    entities: List[Entity] = Field(description="Entities in the sample")
    code: str = Field(description="DSL code")
    context: Optional[str] = Field(description="Context of the sample")
