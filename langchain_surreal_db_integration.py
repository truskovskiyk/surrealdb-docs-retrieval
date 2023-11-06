import asyncio

import uuid
from typing import Any, Iterable, List, Optional
import pandas as pd

from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore

try:
    from surrealdb import Surreal
except ImportError:
    raise ImportError(
        "Could not import surrealdb python package. "
        "Please install it with `pip install surrealdb`."
    )

class SurrealDB(VectorStore):
    def __init__(
        self,
        embedding: Embeddings,
        vector_key: Optional[str] = "vector",
        id_key: Optional[str] = "id",
        text_key: Optional[str] = "text",
        uri: Optional[str] = "ws://localhost:8000/rpc",
        user: Optional[str] = "root",
        password: Optional[str] = "root",
        workspace: Optional[str] = "test",
        database: Optional[str] = "test",
        table_name: Optional[str] = "embeddings",
        index_name: Optional[str] = "embedding_index",
    ):

        self._embedding = embedding
        self._vector_key = vector_key
        self._id_key = id_key
        self._text_key = text_key

        # SurrealDB specific
        self._user = user
        self._password = password
        self._workspace = workspace
        self._database = database
        self._uri = uri
        self._table_name = table_name
        self._index_name = index_name

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    async def _run_sql_async(self, sql: str) -> Any:
        async with Surreal(self._uri) as db:
            await db.signin({"user": self._user, "pass": self._password})
            await db.use(self._workspace, self._database)
            results = await db.query(sql=sql)
            return results

    def _run_sql(self, sql: str) -> Any:
        results = asyncio.run(self._run_sql_async(sql=sql))
        return results 
    
    async def _upsert_batch_sql(self, docs: List[Any]):
        async with Surreal(self._uri) as db:
            await db.signin({"user": self._user, "pass": self._password})
            await db.use(self._workspace, self._database)
            for idx in range(0, len(docs)):
                doc = docs[idx]
                await db.create(self._table_name, doc)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Turn texts into embedding and add it to the database

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.

        Returns:
            List of ids of the added texts.
        """
        # Embed texts and create documents
        docs = []
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        embeddings = self._embedding.embed_documents(list(texts))
        dim = len(embeddings[0])
        for idx, text in enumerate(texts):
            embedding = embeddings[idx]
            metadata = metadatas[idx] if metadatas else {}
            docs.append(
                {
                    self._vector_key: embedding,
                    self._id_key: ids[idx],
                    self._text_key: text,
                    **metadata,
                }
            )

        asyncio.run(self._upsert_batch_sql(docs=docs))
        index_sql = f"DEFINE INDEX {self._index_name} ON {self._table_name} FIELDS {self._vector_key} MTREE DIMENSION {dim} DIST COSINE;"
        result = self._run_sql(sql=index_sql)
        assert result[0]['status'] == 'OK', f"Cannot create index {index_sql}"
        return ids

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return documents most similar to the query

        Args:
            query: String to query the vectorstore with.
            k: Number of documents to return.

        Returns:
            List of documents most similar to the query.
        """
        embedding = self._embedding.embed_query(query)

        search_sql = f"""
        LET $pt = {embedding};
        SELECT *, {self._vector_key}::similarity::cosine({self._vector_key}, $pt) AS dist OMIT {self._vector_key} FROM {self._table_name} WHERE {self._vector_key} <{k}> $pt;
        """
        result = self._run_sql(sql=search_sql)
        assert result[0]['status'] == 'OK', f"Cannot define variable"
        assert result[1]['status'] == 'OK', f"Cannot performe vector search"
        docs = pd.DataFrame(result[1]['result'])
        return [
            Document(
                page_content=row[self._text_key],
                metadata=row[docs.columns != self._text_key],
            )
            for _, row in docs.iterrows()
        ]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        vector_key: Optional[str] = "vector",
        id_key: Optional[str] = "id",
        text_key: Optional[str] = "text",

        uri: Optional[str] = "ws://localhost:8000/rpc",
        user: Optional[str] = "root",
        password: Optional[str] = "root",
        workspace: Optional[str] = "test",
        table_name: Optional[str] = "embeddings",
        index_name: Optional[str] = "embedding_index",

        **kwargs: Any,
    ) -> 'SurrealDB':
        
        instance = SurrealDB(
            embedding,
            vector_key,
            id_key,
            text_key,

            uri=uri,
            user=user,
            password=password,
            workspace=workspace,
            table_name=table_name,
            index_name=index_name,
                        
        )
        instance.add_texts(texts, metadatas=metadatas, **kwargs)

        return instance
