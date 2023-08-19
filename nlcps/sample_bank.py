from typing import List, Protocol

from nlcps.types import Entity, RelatedSample


class SampleBank(Protocol):
    def query_entities(self, entities: List[Entity]) -> List[RelatedSample]:
        ...

    def query_embedding(self, embedding: float) -> List[RelatedSample]:
        ...
