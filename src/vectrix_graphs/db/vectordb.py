import chromadb
import ollama
from ollama import Client
from langchain_core.documents import Document
from typing import List
import uuid

class VectorDB():
    def __init__(self, logger, embeddings_model = None):
        self.logger = logger
        try:
            self.client = chromadb.HttpClient(host='localhost', port=7777)
        except Exception:
            self.logger.warning("Connecting to ChromaDB on localhost failed, using Docker networking instead")
            self.client = chromadb.HttpClient(host='host.docker.internal', port=7777)
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
        for doc in documents:
            response = ollama.embeddings(
                model="bge-m3",
                prompt=doc.page_content
            )
            embeddings.append(response['embedding'])

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
        try:
            embeddings = ollama.embeddings(
                model="bge-m3",
                prompt=query
            )
        except Exception:
            self.logger.warning("Connecting to Ollama on localhost failed, using Docker networking instead")
            client = Client(host="host.docker.internal")
            embeddings = client.embeddings(
                model="bge-m3",
                prompt=query
            )


        results = self.collection.query(
            query_embeddings=[embeddings['embedding']],
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