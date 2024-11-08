from chromadb.config import Settings
import cohere
from langchain_core.documents import Document
from typing import List
import uuid
import os

class VectorDB():
    def __init__(self, logger, embeddings_model=None, type: str = "weaviate"):
        """
        :param type: The type of vector database to use. Choices are "weaviate" or "chromadb".
        """
        self.logger = logger
        self.co = cohere.ClientV2()
        self.type = type
        if type == "chromadb":
            import chromadb
            if os.environ["ENV"] == "local":
                try:
                    # Try localhost first
                    self.logger.warning("Connecting using localhost")
                    self.client = chromadb.HttpClient(host='localhost', port=7777)

                except Exception:
                    # Run from a docker container
                    self.logger.warning("Connecting using Docker network")
                    self.client = chromadb.HttpClient(
                        host='host.docker.internal',
                        port=7777
                        )
            else:
                self.logger.warning("Connecting to hosted ChromaDB service")
                self.client = chromadb.HttpClient(
                    host=os.environ["CHROMA_URL"], 
                    port=8000, 
                    settings=Settings(
                        chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                        chroma_client_auth_credentials=os.environ["CHROMA_SERVER_AUTHN_CREDENTIALS"]
                    )
                )
            try:
                self.collection = self.client.create_collection(
                    name="demo",
                    metadata={"hnsw:space": "cosine"} # Instead of the default squared L2 distance
                    )
                self.logger.info("Demo collection created")
            except Exception:
                self.logger.warning("Demo collection already exists")
                self.collection = self.client.get_collection(name="demo")

        elif type == "weaviate":
            import weaviate
            from weaviate.classes.config import Configure
            if os.environ["ENV"] == "local":
                try:
                    self.client = weaviate.connect_to_local()
                except Exception:
                    self.client = weaviate.connect_to_local(
                        host="host.docker.internal",
                    )
                try:
                    self.client.collections.create(
                        "demo",
                        vectorizer_config=Configure.Vectorizer.text2vec_cohere(
                            model="embed-multilingual-v3.0"
                        ))
                    self.collection = self.client.collections.get("demo")
                except Exception:
                    self.collection = self.client.collections.get("demo")
                    self.logger.warning("Demo collection already exists")

    def create_collection(self, name: str):
        if self.type == "chromadb":
            self.collection = self.client.create_collection(name=name)
            self.logger.info(f"{name} collection created")
        elif self.type == "weaviate":
            self.client.collections.create(name=name)
            self.collection = self.client.collections.get(name)
            self.logger.info(f"{name} collection created")

    def add_documents(self, documents: List[Document]):
        '''
        This function adds documents to the vector database.
        '''
        if self.type == "chromadb":
            embeddings = []

            response = self.co.embed(
                texts=[doc.page_content for doc in documents], 
                model="embed-english-v3.0", 
                input_type="search_document", 
                embedding_types=["float"]
            )

            embeddings = response.embeddings.float_

            self.collection.add(
                documents=[doc.page_content for doc in documents],
                embeddings=embeddings,
                metadatas=[doc.metadata for doc in documents],
                ids=[str(uuid.uuid4()) for _ in documents]
            )

        elif self.type == "weaviate":
            with self.collection.batch.dynamic() as batch:
                for doc in documents:
                    batch.add_object(
                        properties={"metadata": doc.metadata, "content": doc.page_content}
                    )

        self.logger.info(f"Added {len(documents)} documents to the vector database")

    def similarity_search(self, query: str, k: int = 3):
        '''
        This function queries the vector database and returns Langchain Documents with cosine distances.
        '''
        if self.type == "chromadb":
            response = self.co.embed(
                texts=[query], 
                model="embed-english-v3.0", 
                input_type="search_query", 
                embedding_types=["float"]
            )
            embeddings = response.embeddings.float_

            print(embeddings)


            results = self.collection.query(
                query_embeddings=embeddings,
                n_results=k,
                include=['documents', 'metadatas', 'distances'],
                where={"metadata_field": "is_equal_to_this"}
                )

            documents = []
            for content, metadata, distance, id in zip(results['documents'][0], results['metadatas'][0], results['distances'][0], results['ids'][0]):
                metadata['cosine_distance'] = distance
                metadata['uuid'] = id
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)

            return documents

        elif self.type == "weaviate":
            results = self.collection.query.near_text(
                query=query,
                limit=k,
            )

            documents = []
            for obj in results.objects:
                metadata = obj.properties.get('metadata', {})
                metadata['uuid'] = str(obj.uuid)
                # Add distance/certainty if available
                if obj.metadata.distance is not None:
                    metadata['cosine_distance'] = obj.metadata.distance
                
                doc = Document(
                    page_content=obj.properties.get('content', ''),
                    metadata=metadata
                )
                documents.append(doc)

            return documents
    
    def remove_collection(self, name: str):
        if self.type == "chromadb":
            self.client.delete_collection(name=name)
        elif self.type == "weaviate":
            self.client.collections.delete(name)
        self.logger.info(f"{name} collection deleted")