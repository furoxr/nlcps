from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from pydantic import PrivateAttr
from pydantic.v1 import BaseModel, BaseSettings
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams

from nlcps.analysis_chain import AnalysisChain, AnalysisResult
from nlcps.retrieve_chain import RetrieveChain
from nlcps.selector import FilterExampleSelector
from nlcps.types import AnalysisExample, RetrieveExample
from nlcps.util import logger


class NlcpConfig(BaseSettings):
    openai_api_key: str
    openai_api_base: str

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


class NlcpExecutor(BaseModel):
    config: NlcpConfig

    llm: ChatOpenAI = PrivateAttr()
    analysis_chain: AnalysisChain = PrivateAttr()
    retrieve_chain: RetrieveChain = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)

        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key, openai_api_base=self.openai_api_base
        )
        self._qdrant_client = QdrantClient()

        embeddings = OpenAIEmbeddings(
            openai_api_key=self.openai_api_key, openai_api_base=self.openai_api_base
        )

        dsl_syntax_selector = FilterExampleSelector(
            Qdrant(self._qdrant_client, self.dsl_syntax_collection_name, embeddings),
            self.dsl_syntax_k,
        )

        dsl_rules_selector = FilterExampleSelector(
            Qdrant(self._qdrant_client, self.dsl_rules_collection_name, embeddings),
            self.dsl_rules_k,
        )

        dsl_examples_selector = FilterExampleSelector(
            Qdrant(self._qdrant_client, self.dsl_examples_collection_name, embeddings),
            self.dsl_examples_k,
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
        collections = [
            self.config.dsl_syntax_collection_name,
            self.config.dsl_rules_collection_name,
            self.config.dsl_examples_collection_name,
        ]
        for collection_name in collections:
            try:
                self._qdrant_client.get_collection(collection_name)
                logger.info(f"Collection '{collection_name}' already exists")
            except UnexpectedResponse:
                self._qdrant_client.create_collection(
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
        context: str,
    ) -> str:
        """Retrieve related samples from sample bank."""
        analysis_result = self.analysis_chain.run(user_utterance)
        return self.retrieve_chain.run(
            user_utterance, analysis_result.entities, context
        )
