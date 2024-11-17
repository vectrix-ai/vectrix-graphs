import os
from typing import List

import cohere
import weaviate
from langchain_core.documents import Document
from weaviate.classes.config import Configure

from ..logger import setup_logger

logger = setup_logger(name=__name__)


class Weaviate:
    def __init__(self, embeddings_model=None):
        """Initialize Weaviate vector database connection"""
        self.co = cohere.ClientV2()

        if os.environ["ENV"] == "local":
            try:
                self.client = weaviate.connect_to_local()
            except Exception:
                self.client = weaviate.connect_to_local(
                    host="host.docker.internal",
                )

    def set_collection(self, name: str):
        """Set the Weaviate collection"""
        try:
            self.collection = self.client.collections.get(name)
        except Exception:
            self.create_collection(name)

    def create_collection(self, name: str, vectorizer_config="cohere"):
        """Create a new Weaviate collection"""
        if vectorizer_config == "cohere":
            self.client.collections.create(
                name=name,
                vectorizer_config=Configure.Vectorizer.text2vec_cohere(
                    model="embed-multilingual-v3.0"
                ),
            )
            self.collection = self.client.collections.get(name)
            logger.info(f"{name} collection created")
        elif vectorizer_config == "voyage":
            try:
                self.client.collections.create(
                    name=name,
                    vectorizer_config=Configure.Vectorizer.none(),
                )
                self.collection = self.client.collections.get(name)
                logger.info(f"{name} collection created")
            except Exception:
                self.collection = self.client.collections.get(name)
                logger.warning(f"{name} collection already exists")

    def add_documents(self, documents: List[Document]):
        """
        This function adds documents to the vector database.
        """
        with self.collection.batch.dynamic() as batch:
            for doc in documents:
                batch.add_object(
                    properties={
                        "metadata": doc.metadata,
                        "content": doc.page_content,
                    }
                )

        logger.info(f"Added {len(documents)} documents to the vector database")

    def similarity_search(self, query: str, k: int = 3):
        """Query the Weaviate database and return Langchain Documents with cosine distances"""
        results = self.collection.query.near_text(
            query=query,
            limit=k,
        )

        documents = []
        for obj in results.objects:
            metadata = obj.properties.get("metadata", {})
            metadata["uuid"] = str(obj.uuid)
            if obj.metadata.distance is not None:
                metadata["cosine_distance"] = obj.metadata.distance

            doc = Document(
                page_content=obj.properties.get("content", ""), metadata=metadata
            )
            documents.append(doc)

        return documents

    def remove_collection(self, name: str):
        """Remove a Weaviate collection"""
        self.client.collections.delete(name)
        logger.info(f"{name} collection deleted")

    def close(self):
        """Close the Weaviate client connection"""
        self.client.close()
