from langchain import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from nlcps.selector import FilterExampleSelector
from nlcps.types import (
    AnalysisResult,
    DSLRuleExample,
    DSLSyntaxExample,
    RetrieveExample,
)

from typing import List, Optional


RETRIEVE_PROMPT = (
    "{system_instruction}\n"
    "Here are examples of this DSL's syntax:\n{dsl_syntax}"
    "Generate an DSL progarm to fulfill the given user utterance. Remember to follow the following rules when generating DSL:"
    "{dsl_rules}"
)


class RetrieveChain:
    def __init__(
        self,
        llm: ChatOpenAI,
        user_utterance: str,
        system_instruction: str,
        analysis_result: AnalysisResult,
        dsl_syntax_selector: FilterExampleSelector[DSLSyntaxExample],
        dsl_rules_selector: FilterExampleSelector[DSLRuleExample],
        dsl_examples_selector: FilterExampleSelector[RetrieveExample],
        context: Optional[str] = None,
    ) -> None:
        """Initialize an analysis chain.

        This chain will analyze user utterance and returns a AnalysisResult.

        Args:
            llm (ChatOpenAI): Interface with openai
            entities (List[Entity]): All entities DSL supported
            context_rules (List[str]): Rules inster into prompt, helping AI decide whether context is needed
            examples (List[AnalysisExample]): Few-shot prompt examples
        """
        few_shot_prompt = self.few_shot_exmaple_template(
            user_utterance, analysis_result.entities, dsl_examples_selector
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(RETRIEVE_PROMPT),
                few_shot_prompt,
                HumanMessagePromptTemplate.from_template("{input}\n{context}"),
            ]
        )
        self.prompt_template = self.prompt_template.partial(
            system_instruction=system_instruction
        )
        self.format_dsl_syntax(dsl_syntax_selector, analysis_result.entities)
        self.format_dsl_rules(dsl_rules_selector, analysis_result.entities)
        if context:
            self.prompt_template = self.prompt_template.partial(
                context=f"Context is:\n{context}"
            )

        self.chain = LLMChain(
            llm=llm,
            prompt=self.prompt_template,
        )

    def format_dsl_syntax(
        self,
        dsl_syntax_selector: FilterExampleSelector[DSLSyntaxExample],
        entities: List[str],
    ):
        """Format all DSL syntax code examples of which entities is a subset of entities"""
        key = f"{dsl_syntax_selector.qdrant.metadata_payload_key}.entities"
        dsl_syntax_examples = dsl_syntax_selector.select(
            Filter(must=[FieldCondition(key=key, match=MatchAny(any=entities))])
        )
        self.prompt_template = self.prompt_template.partial(
            dsl_syntax="\n".join(["- " + i.code for i in dsl_syntax_examples])
        )

    def format_dsl_rules(
        self,
        dsl_rules_selector: FilterExampleSelector[DSLRuleExample],
        entities: List[str],
    ):
        """Format all DSL rules related to entities"""
        key = f"{dsl_rules_selector.qdrant.metadata_payload_key}.entities"
        dsl_rules_examples = dsl_rules_selector.select(
            Filter(must=[FieldCondition(key=key, match=MatchAny(any=entities))])
        )
        self.prompt_template = self.prompt_template.partial(
            dsl_rules="\n".join(["- " + i.rule for i in dsl_rules_examples])
        )

    def few_shot_exmaple_template(
        self,
        user_utterance: str,
        entities: List[str],
        dsl_examples_selector: FilterExampleSelector[RetrieveExample],
    ) -> FewShotChatMessagePromptTemplate:
        """Get all examples of this DSL related to user utterance"""
        key = f"{dsl_examples_selector.qdrant.metadata_payload_key}.entities"
        condition = Filter(
            must=[
                FieldCondition(key=key, match=MatchValue(value=entity))
                for entity in entities
            ]
        )
        similarity_examples = dsl_examples_selector.similarity_select(
            {"user_utterance": user_utterance}, filter=condition
        )
        similarity_examples.sort(key=lambda x: x[1])
        similarity_examples.reverse()

        final_examples = []
        for index in range(max([dsl_examples_selector.k, len(entities)])):
            example = similarity_examples[index][0]
            final_examples.append(
                {
                    "input": example.user_utterance,
                    "context": f"Context is:{example.context}"
                    if example.context
                    else "",
                    "output": example.code,
                }
            )

        example_prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template("{input}\n{context}"),
                AIMessagePromptTemplate.from_template("{output}"),
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt, examples=final_examples
        )
        return few_shot_prompt

    def run(self, user_utterance: str) -> str:
        return self.chain.run(user_utterance)
