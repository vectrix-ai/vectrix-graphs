import chromadb
from chromadb.config import Settings
import cohere
from langchain_core.documents import Document
from typing import List
import uuid
import os

class VectorDB():
    def __init__(self, logger, embeddings_model = None):
        self.logger = logger
        self.co = cohere.ClientV2()
        try:
            # Try localhost first
            self.client = chromadb.HttpClient(
                host='localhost',
                port=7777
                )
        except Exception:
            self.logger.warning("Connecting to ChromaDB on localhost failed, using hosted service instead")
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

    def create_collection(self, name: str):
        self.collection = self.client.create_collection(name=name)
        self.logger.info(f"{name} collection created")

    def add_documents(self, documents: List[Document]):
        '''
        This function adds documents to the vector database.
        '''
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

        self.logger.info(f"Added {len(documents)} documents to the vector database")

    def similarity_search(self, query: str, k: int = 3):
        '''
        This function queries the vector database and returns Langchain Documents with cosine distances.
        '''
        response = self.co.embed(
            texts=[query], 
            model="embed-english-v3.0", 
            input_type="search_query", 
            embedding_types=["float"]
        )
        embeddings = response.embeddings.float_


        results = self.collection.query(
            query_embeddings=embeddings,
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )

        documents = []
        for content, metadata, distance, id in zip(results['documents'][0], results['metadatas'][0], results['distances'][0], results['ids'][0]):
            metadata['cosine_distance'] = distance
            metadata['uuid'] = id
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        return documents
    
    def remove_collection(self, name: str):
        self.client.delete_collection(name=name)
        self.logger.info(f"{name} collection deleted")