from typing import List, Optional

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from pydantic.v1 import BaseSettings
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams

from nlcps.analysis_chain import AnalysisChain, AnalysisResult
from nlcps.retrieve_chain import RetrieveChain
from nlcps.selector import FilterExampleSelector
from nlcps.types import (
    AnalysisExample,
    DSLRuleExample,
    DSLSyntaxExample,
    RetrieveExample,
)
from nlcps.util import logger


class NlcpsConfig(BaseSettings):
    openai_api_key: str
    openai_api_base: str = "https://api.openai.com/v1"

    entities: List[str]
    context_rules: List[str]
    analysis_examples: List[AnalysisExample]
    system_instruction: str

    dsl_syntax_collection_name: str
    dsl_syntax_k: int = 5

    dsl_rules_collection_name: str
    dsl_rules_k: int = 5

    dsl_examples_collection_name: str
    dsl_examples_k: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class NlcpsExecutor:
    def __init__(self, config: NlcpsConfig):
        self.config = config

        self.llm = ChatOpenAI(
            openai_api_key=self.config.openai_api_key,
            openai_api_base=self.config.openai_api_base,
        )
        self.qdrant_client = QdrantClient()

        embeddings = OpenAIEmbeddings(  # type: ignore
            openai_api_key=self.config.openai_api_key,
            openai_api_base=self.config.openai_api_base,
        )

        dsl_syntax_selector = FilterExampleSelector(
            qdrant=Qdrant(
                self.qdrant_client, self.config.dsl_syntax_collection_name, embeddings
            ),
            model_cls=DSLSyntaxExample,
            k=self.config.dsl_syntax_k,
            input_keys=["code"],
        )

        dsl_rules_selector = FilterExampleSelector(
            qdrant=Qdrant(
                self.qdrant_client, self.config.dsl_rules_collection_name, embeddings
            ),
            k=self.config.dsl_rules_k,
            model_cls=DSLRuleExample,
            input_keys=["rule"],
        )

        dsl_examples_selector = FilterExampleSelector(
            qdrant=Qdrant(
                self.qdrant_client, self.config.dsl_examples_collection_name, embeddings
            ),
            k=self.config.dsl_examples_k,
            model_cls=RetrieveExample,
            input_keys=["user_utterance"],
        )
        self.analysis_chain = AnalysisChain(
            llm=self.llm,
            entities=self.config.entities,
            context_rules=self.config.context_rules,
            examples=self.config.analysis_examples,
        )
        self.retrieve_chain = RetrieveChain(
            llm=self.llm,
            system_instruction=self.config.system_instruction,
            dsl_syntax_selector=dsl_syntax_selector,
            dsl_rules_selector=dsl_rules_selector,
            dsl_examples_selector=dsl_examples_selector,
        )

    def init_vectorstore(self):
        """Create collections if not exists."""
        collections = [
            self.config.dsl_syntax_collection_name,
            self.config.dsl_rules_collection_name,
            self.config.dsl_examples_collection_name,
        ]
        for collection_name in collections:
            try:
                self.qdrant_client.get_collection(collection_name)
                logger.info(f"Collection '{collection_name}' already exists")
            except UnexpectedResponse:
                self.qdrant_client.create_collection(
                    collection_name, VectorParams(size=1536, distance=Distance.COSINE)
                )
                logger.info(f"Collection {collection_name} created.")

    def analysis(self, user_utterance: str) -> AnalysisResult:
        """Analysis user utterance to get entities and whether context needed."""
        return self.analysis_chain.run(user_utterance)

    def retrieve(
        self, user_utterance: str, entities: List[str]
    ) -> List[tuple[RetrieveExample, float]]:
        """Retrieve related samples from sample bank."""
        return self.retrieve_chain.retrieve_few_shot_examples(user_utterance, entities)

    def program_synthesis(
        self,
        user_utterance: str,
        context: Optional[str] = None,
    ) -> str:
        """Generate DSL program to fulfill user utterance."""
        analysis_result = self.analysis_chain.run(user_utterance)
        logger.debug(f"{analysis_result}")
        if analysis_result.need_context and context is None:
            raise ValueError(
                "User utterance requires context but no context is provided."
            )

        return self.retrieve_chain.run(
            user_utterance, analysis_result.entities, context
        )
