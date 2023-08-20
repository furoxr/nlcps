from langchain.vectorstores.qdrant import Qdrant
from langchain.prompts.example_selector.semantic_similarity import sorted_values
from pydantic.v1 import Extra
from pydantic.v1 import BaseModel
from qdrant_client.models import Filter

from typing import Generic, List, Optional, Dict, TypeVar

from nlcps.types import BaseIdModel

T = TypeVar("T", bound=BaseIdModel)


class FilterExampleSelector(BaseModel, Generic[T]):
    qdrant: Qdrant
    model_cls: type[T]
    k: int = 4
    input_keys: Optional[List[str]] = None

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    def add(self, item: T) -> str:
        """Add new example to qdrant."""
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
        if self.input_keys:
            input_variables = {key: input_variables[key] for key in self.input_keys}
        query = " ".join(sorted_values(input_variables))
        example_docs = self.qdrant.client.search(
            self.qdrant.collection_name,
            self.qdrant._embed_query(query),
            k=self.k,
            filter=filter,
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
