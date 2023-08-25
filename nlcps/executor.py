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
from nlcps.selector import (
    FilterExampleSelector,
    dsl_examples_selector_factory,
    dsl_rules_selector_factory,
    dsl_syntax_selector_factory,
)
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

    collection_name_prefix: str

    dsl_syntax_k: int = 5
    dsl_rules_k: int = 5
    dsl_examples_k: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class NlcpsExecutor:
    def __init__(
        self,
        qdrant_client: QdrantClient,
        analysis_chain: AnalysisChain,
        retrieve_chain: RetrieveChain,
    ):
        self.qdrant_client = qdrant_client
        self.analysis_chain = analysis_chain
        self.retrieve_chain = retrieve_chain

    def init_vectorstore(self):
        """Create collections if not exists."""
        collections = [
            self.retrieve_chain.dsl_rules_selector.collection_name,
            self.retrieve_chain.dsl_examples_selector.collection_name,
            self.retrieve_chain.dsl_syntax_selector.collection_name,
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


def nlcps_executor_factory(config: NlcpsConfig) -> NlcpsExecutor:
    dsl_syntax_collection_name = f"{config.collection_name_prefix}_dsl_syntax"
    dsl_rules_collection_name = f"{config.collection_name_prefix}_dsl_rules"
    dsl_examples_collection_name = f"{config.collection_name_prefix}_dsl_examples"

    llm = ChatOpenAI(
        openai_api_key=config.openai_api_key,
        openai_api_base=config.openai_api_base,
    )
    qdrant_client = QdrantClient()

    embeddings = OpenAIEmbeddings(  # type: ignore
        openai_api_key=config.openai_api_key,
        openai_api_base=config.openai_api_base,
    )

    dsl_syntax_selector = dsl_syntax_selector_factory(
        qdrant_client, dsl_syntax_collection_name, embeddings, config.dsl_syntax_k
    )
    dsl_rules_selector = dsl_rules_selector_factory(
        qdrant_client, dsl_rules_collection_name, embeddings, config.dsl_rules_k
    )
    dsl_examples_selector = dsl_examples_selector_factory(
        qdrant_client, dsl_examples_collection_name, embeddings, config.dsl_examples_k
    )

    analysis_chain = AnalysisChain(
        llm=llm,
        entities=config.entities,
        context_rules=config.context_rules,
        examples=config.analysis_examples,
    )
    retrieve_chain = RetrieveChain(
        llm=llm,
        system_instruction=config.system_instruction,
        dsl_syntax_selector=dsl_syntax_selector,
        dsl_rules_selector=dsl_rules_selector,
        dsl_examples_selector=dsl_examples_selector,
    )
    return NlcpsExecutor(
        qdrant_client=qdrant_client,
        analysis_chain=analysis_chain,
        retrieve_chain=retrieve_chain,
    )
