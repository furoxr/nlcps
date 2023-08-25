from typing import Dict, Generic, List, Optional, TypeVar

from langchain.prompts.example_selector.semantic_similarity import sorted_values
from langchain.vectorstores.qdrant import Qdrant
from langchain.embeddings.base import Embeddings
from pydantic.v1 import BaseModel, Extra
from qdrant_client import QdrantClient
from qdrant_client.models import Filter

from nlcps.types import BaseIdModel, DSLRuleExample, DSLSyntaxExample, RetrieveExample

T = TypeVar("T", bound=BaseIdModel)


class FilterExampleSelector(BaseModel, Generic[T]):
    qdrant: Qdrant
    model_cls: type[T]
    k: int = 4
    input_keys: Optional[List[str]] = None

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True
    
    @property
    def collection_name(self) -> str:
        return self.qdrant.collection_name
    
    def add(self, item: T) -> str:
        """Add new point to collection."""
        assert isinstance(item, self.model_cls)
        data = item.dict()
        if "id" in data:
            data.pop("id")

        if self.input_keys:
            string_example = " ".join(
                sorted_values({key: data[key] for key in self.input_keys})
            )
        else:
            string_example = " ".join(sorted_values(data))
        ids = self.qdrant.add_texts([string_example], metadatas=[data])
        return ids[0]

    def similarity_select(
        self, input_variables: Dict[str, str], filter: Optional[Filter] = None
    ) -> List[tuple[T, float]]:
        """Select points with semantic similarity."""
        if self.input_keys:
            input_variables = {key: input_variables[key] for key in self.input_keys}
        query = " ".join(sorted_values(input_variables))
        example_docs = self.qdrant.client.search(
            self.qdrant.collection_name,
            self.qdrant._embed_query(query),
            limit=self.k,
            query_filter=filter,
        )

        examples = [
            (
                self.model_cls(  # type: ignore[arg-type]
                    id=e.id, **e.payload.get(self.qdrant.metadata_payload_key)  # type: ignore[union-attr, arg-type]
                ),
                e.score,
            )
            for e in example_docs
        ]
        return examples

    def select(self, filter: Optional[Filter] = None) -> List[T]:
        """Select points with filter"""
        points = self.qdrant.client.scroll(
            self.qdrant.collection_name,
            filter,
            limit=self.k,
        )[0]
        examples = [
            self.model_cls(  # type: ignore[arg-type]
                id=point.id, **point.payload.get(self.qdrant.metadata_payload_key)  # type: ignore[union-attr, arg-type]
            )
            for point in points
        ]

        return examples


def dsl_syntax_selector_factory(
    client: QdrantClient, collection_name: str, embeddings: Embeddings, k: int
) -> FilterExampleSelector[DSLSyntaxExample]:
    return FilterExampleSelector(
        qdrant=Qdrant(client, collection_name, embeddings),
        model_cls=DSLSyntaxExample,
        k=k,
        input_keys=["code"],
    )


def dsl_rules_selector_factory(
    client: QdrantClient, collection_name: str, embeddings: Embeddings, k: int
) -> FilterExampleSelector[DSLRuleExample]:
    return FilterExampleSelector(
        qdrant=Qdrant(client, collection_name, embeddings),
        model_cls=DSLRuleExample,
        k=k,
        input_keys=["rule"],
    )


def dsl_examples_selector_factory(
    client: QdrantClient, collection_name: str, embeddings: Embeddings, k: int
) -> FilterExampleSelector[RetrieveExample]:
    return FilterExampleSelector(
        qdrant=Qdrant(client, collection_name, embeddings),
        model_cls=RetrieveExample,
        k=k,
        input_keys=["user_utterance"],
    )
