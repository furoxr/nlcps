from typing import Any, List, Optional

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from pydantic.v1 import BaseModel, PrivateAttr
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from nlcps.selector import FilterExampleSelector
from nlcps.types import (
    DSLRuleExample,
    DSLSyntaxExample,
    RetrieveExample,
)

RETRIEVE_PROMPT = (
    "{system_instruction}\n"
    "Here are examples of this DSL's syntax:\n{dsl_syntax}"
    "Generate an DSL progarm to fulfill the given user utterance. Remember to follow the following rules when generating DSL:"
    "{dsl_rules}"
)


class RetrieveChain(BaseModel):
    llm: ChatOpenAI
    system_instruction: str
    dsl_examples_selector: FilterExampleSelector[RetrieveExample]

    _prompt_template: ChatPromptTemplate = PrivateAttr()
    _chain: LLMChain = PrivateAttr()

    async def format_dsl_syntax(
        self,
        entities: List[str],
    ):
        """Format all DSL syntax code examples of which entities is a subset of entities"""
        dsl_syntax_examples = await DSLSyntaxExample.scroll(
            Filter(must=[FieldCondition(key='entities', match=MatchAny(any=entities))])
        )
        self._prompt_template = self._prompt_template.partial(
            dsl_syntax="\n".join(["- " + i.code for i in dsl_syntax_examples])
        )

    async def format_dsl_rules(
        self,
        entities: List[str],
    ):
        """Format all DSL rules related to entities"""
        dsl_rules_examples = await DSLRuleExample.scroll(
            Filter(must=[FieldCondition(key='entities', match=MatchAny(any=entities))])
        )
        self._prompt_template = self._prompt_template.partial(
            dsl_rules="\n".join(["- " + i.rule for i in dsl_rules_examples])
        )

    def retrieve_few_shot_examples(
        self, user_utterance: str, entities: List[str]
    ) -> List[tuple[RetrieveExample, float]]:
        """Retrieve related samples from sample bank."""
        key = f"{self.dsl_examples_selector.qdrant.metadata_payload_key}.entities"
        condition = Filter(
            must=[
                FieldCondition(key=key, match=MatchValue(value=entity))
                for entity in entities
            ]
        )
        similarity_examples = self.dsl_examples_selector.similarity_select(
            {"user_utterance": user_utterance}, filter=condition
        )
        similarity_examples.sort(key=lambda x: x[1])
        similarity_examples.reverse()
        return similarity_examples

    def few_shot_exmaple_template(
        self,
        user_utterance: str,
        entities: List[str],
    ) -> FewShotChatMessagePromptTemplate:
        """Format template leveraging examples"""
        similarity_examples = self.retrieve_few_shot_examples(user_utterance, entities)
        final_examples = []

        # Select max(k, length(entities)) samples, such that each associated entities is represented at least once
        for index in range(min(max(self.dsl_examples_selector.k, len(entities)), len(similarity_examples))):
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

    async def init_chain(
        self, user_utterance: str, entities: List[str], context: Optional[str] = None
    ):
        """Format template with rules, syntax and examples"""
        few_shot_prompt = self.few_shot_exmaple_template(user_utterance, entities)
        self._prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(RETRIEVE_PROMPT),
                few_shot_prompt,
                HumanMessagePromptTemplate.from_template("{input}\n{context}"),
            ]
        )
        self._prompt_template = self._prompt_template.partial(
            system_instruction=self.system_instruction
        )
        await self.format_dsl_syntax(entities)
        await self.format_dsl_rules(entities)
        if context:
            self._prompt_template = self._prompt_template.partial(
                context=f"Context is:\n{context}"
            )

        self._chain = LLMChain(
            llm=self.llm,
            prompt=self._prompt_template,
        )

    async def run(
        self, user_utterance: str, entities: List[str], context: Optional[str] = None
    ) -> str:
        await self.init_chain(user_utterance, entities, context)
        return await self._chain.arun(input=user_utterance, context=context)
