import os
from typing import Any, Dict, List, Literal

import cohere
import voyageai
import weaviate
from langchain_core.documents import Document
from weaviate.classes.config import Configure
from weaviate.classes.query import MetadataQuery

from ..logger import setup_logger

logger = setup_logger(name=__name__, level="INFO")


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
            logger.error(f"{name} collection does not exist")
            raise ValueError(f"{name} collection does not exist")

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
                try:
                    self.collection = self.client.collections.get(name)
                except Exception:
                    logger.error(f"{name} collection creation failed")
                    raise ValueError(f"{name} collection creation failed")
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

    def add_multi_modal_documents(
        self, documents: List[List[Any]], metadatas: List[Dict[str, Any]]
    ):
        """
        This function adds multi-modal documents to the vector database.
        """
        logger.info(
            f"Adding {len(documents)} multi-modal documents to the vector database"
        )
        if len(documents) != len(metadatas):
            logger.error("The number of documents and metadatas must be the same")
            raise ValueError("The number of documents and metadatas must be the same")

        vo = voyageai.Client()

        # Initialize a list to store all embeddings
        all_embeddings = []

        # Process this in chunks for 100 documents at a time
        for i in range(0, len(documents), 100):
            result = vo.multimodal_embed(
                inputs=documents[i : i + 100],
                model="voyage-multimodal-3",
                truncation=True,
            )
            all_embeddings.extend(result.embeddings)

        logger.info(f"Embeddings created for {len(all_embeddings)} documents")

        with self.collection.batch.dynamic() as batch:
            for i, data_row in enumerate(documents):
                batch.add_object(properties=metadatas[i], vector=all_embeddings[i])
        logger.info(f"Added {len(documents)} documents to the vector database")

    def similarity_search(
        self, query: str, k: int = 3, type: Literal["text", "multimodal"] = "text"
    ):
        """Query the Weaviate database and return Langchain Documents with cosine distances"""
        if type == "text":
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

        elif type == "multimodal":
            vo = voyageai.Client()
            vector = vo.multimodal_embed(
                [[query]], model="voyage-multimodal-3", truncation=False
            )
            results = self.collection.query.near_vector(
                near_vector=vector.embeddings[0],
                limit=k,
                return_metadata=MetadataQuery(distance=True),
            )

            documents = []

            for obj in results.objects:
                metadata = obj.properties.copy()
                metadata.pop("text", None)
                if obj.metadata.distance is not None:
                    metadata["cosine_distance"] = obj.metadata.distance
                content = obj.properties["text"]

                documents.append(Document(page_content=content, metadata=metadata))
            return documents

    def remove_collection(self, name: str):
        """Remove a Weaviate collection"""
        try:
            self.client.collections.delete(name)
            logger.info(f"{name} collection deleted")
        except Exception:
            logger.warning(f"{name} collection does not exist")

    def list_collections(self):
        """List all Weaviate collections"""
        return self.client.collections.list_all(simple=False)

    def close(self):
        """Close the Weaviate client connection"""
        self.client.close()
