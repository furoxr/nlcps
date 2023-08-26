import threading
import uuid
from typing import ClassVar, List, Optional, TypeVar

from langchain.embeddings import OpenAIEmbeddings
from pydantic import Field
from pydantic.v1 import BaseModel
from qdrant_client.http.api_client import AsyncApis
from qdrant_client.models import (
    Filter,
    PointsList,
    PointStruct,
    SearchRequest,
    ScrollRequest,
)

from nlcps.exceptions import VectorStoreNotInitialized

T = TypeVar("T", bound="BaseQdrantModel")


class LocalQdrantClient(threading.local):
    _qdrant_client: AsyncApis
    _embedding: OpenAIEmbeddings

    @property
    def qdrant_client(self) -> AsyncApis:
        try:
            return self._qdrant_client
        except AttributeError:
            raise VectorStoreNotInitialized("Qdrant client is not initialized.")

    @qdrant_client.setter
    def qdrant_client(self, value: AsyncApis):
        self._qdrant_client = value

    @property
    def embedding(self) -> OpenAIEmbeddings:
        try:
            return self._embedding
        except AttributeError:
            raise VectorStoreNotInitialized("Embedding client is not initialized.")

    @embedding.setter
    def embedding(self, value: OpenAIEmbeddings):
        self._embedding = value


def initialize(
    qdrant_client: AsyncApis,
    embedding: OpenAIEmbeddings,
):
    _local.qdrant_client = qdrant_client
    _local.embedding = embedding


_local = LocalQdrantClient()


class BaseQdrantModel(BaseModel):
    collection_name: ClassVar[Optional[str]] = None
    """The collection name utilized for interactions with Qdrant, both during search and saving operations."""
    embedding_key: ClassVar[Optional[str]] = None
    """Generating a vector based on the instance's embedding_key value during the saving process."""

    id: Optional[str] = Field(description="Unique id of the point in qdrant")

    @classmethod
    async def search(
        cls: type[T],
        embedding: List[float],
        filter: Optional[Filter] = None,
        limit: int = 5,
    ) -> List[tuple[T, float]]:
        assert (
            cls.collection_name and cls.embedding_key
        ), f"Collection name and embedding key must be set for {cls}"

        response = await _local.qdrant_client.points_api.search_points(
            collection_name=cls.collection_name,
            search_request=SearchRequest(
                vector=embedding,
                with_payload=True,
                filter=filter,
                limit=limit,
            ),
        )

        objs = [(cls(id=p.id, **p.payload), p.score) for p in response.result]  # type: ignore[union-attr, arg-type]
        return objs

    @classmethod
    async def save(cls, instance: T) -> str:
        assert (
            cls.collection_name and cls.embedding_key
        ), f"Collection name and embedding key must be set for {cls}"

        if not isinstance(instance, cls):
            raise TypeError(f"Expected instance of {cls}, got {type(instance)}")

        data = instance.dict(
            exclude={
                "id",
            }
        )

        vector = await _local.embedding.aembed_query(
            getattr(instance, cls.embedding_key)
        )
        point_id = uuid.uuid4().hex
        await _local.qdrant_client.points_api.upsert_points(
            collection_name=cls.collection_name,
            point_insert_operations=PointsList(
                points=[PointStruct(id=point_id, vector=vector, payload=data)]
            ),
        )
        return point_id

    @classmethod
    async def scroll(cls: type[T], filter: Filter, limit: int = 5) -> List[T]:
        assert (
            cls.collection_name and cls.embedding_key
        ), f"Collection name and embedding key must be set for {cls}"

        response = await _local.qdrant_client.points_api.scroll_points(
            collection_name=cls.collection_name,
            scroll_request=ScrollRequest(limit=limit, with_payload=True, filter=filter),
        )

        objs = [cls(id=p.id, **p.payload) for p in response.result]  # type: ignore[union-attr, arg-type]
        return objs
