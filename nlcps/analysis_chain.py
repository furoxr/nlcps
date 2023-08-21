from langchain import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic.v1 import BaseModel

from nlcps.types import AnalysisExample, AnalysisResult

from typing import Dict, List


ANALYSIS_PROMPT = (
    "There are {entities_len} entities in the utterance: {entities}."
    "You need to perform the following tasks:\n"
    "1. Categorize a given sentence into entity categories. Each sentence can have more than one category.\n"
    "2. Classify whether a sentence requires context."
    "Context is required when additional information about the content of a presentation is required"
    "to fulfill the task described in the sentence\n"
    "{context_rules}\n\n"
    "{format_instructions}"
    "Let's think step by step."
)


class AnalysisChain(BaseModel):
    llm: ChatOpenAI
    entities: List[str]
    context_rules: List[str]
    examples: List[AnalysisExample]

    def init_chain(self) -> None:
        """Initialize an analysis chain.

        This chain will analyze user utterance and returns a AnalysisResult.
        """
        example_prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template("{input}"),
                AIMessagePromptTemplate.from_template("{output}"),
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt, examples=self.format_examples(self.examples)
        )
        final_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(ANALYSIS_PROMPT),
                few_shot_prompt,
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )
        parser = PydanticOutputParser(pydantic_object=AnalysisResult)
        self.prompt_template = final_prompt.partial(
            entities_len=len(self.entities),  # type: ignore
            entities=",".join(self.entities),
            context_rules="\n".join(self.context_rules),
            format_instructions=parser.get_format_instructions(),
        )

        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            output_parser=parser,
        )

    def format_examples(self, examples: List[AnalysisExample]) -> List[Dict[str, str]]:
        """Handle the format of examples

        Args:
            examples (List[AnalysisExample]): Examples

        Returns:
            List[Dict[str, str]]: Formated examples required by the prompt template.
        """
        return [
            {"input": e.utterance, "output": e.analysis_result.model_dump_json()}  # type: ignore
            for e in examples
        ]

    def run(self, user_utterance: str) -> AnalysisResult:
        self.init_chain()
        return self.chain.run(user_utterance)
