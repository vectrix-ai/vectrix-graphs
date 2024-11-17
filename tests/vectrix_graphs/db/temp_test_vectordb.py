from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from vectrix_graphs.db import Weaviate


@pytest.fixture
def mock_logger():
    return Mock()


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="This is the first test document", metadata={"source": "test1"}
        ),
        Document(
            page_content="This is the second test document",
            metadata={"source": "test2"},
        ),
    ]


class TestVectorDB:
    @pytest.fixture(autouse=True)
    def setup(self, mock_logger, monkeypatch):
        # Mock environment variables
        monkeypatch.setenv("ENV", "local")

        # Mock Cohere client
        self.mock_cohere = Mock()
        with patch("cohere.ClientV2") as mock_cohere_client:
            mock_cohere_client.return_value = self.mock_cohere
            self.db = Weaviate()
            self.db.set_collection("test_collection")

    def test_create_collection(self, mock_logger):
        with patch("cohere.ClientV2"):
            db = Weaviate()
            db.create_collection("test_collection")
            # Add assertions for Weaviate collection creation

    def test_add_documents(self, mock_logger, sample_documents):
        with patch("cohere.ClientV2"):
            db = Weaviate()
            db.set_collection("test_collection")
            db.add_documents(sample_documents)
            # Add assertions to verify documents were added correctly to Weaviate

    def test_similarity_search(self, mock_logger):
        with patch("cohere.ClientV2"):
            db = Weaviate()
            db.set_collection("test_collection")
            results = db.similarity_search("test query", k=1)
            assert isinstance(results, list)
            # Add more specific assertions for Weaviate search results

    def test_remove_collection(self, mock_logger):
        with patch("cohere.ClientV2"):
            db = Weaviate()
            db.remove_collection("test_collection")
            # Add assertions to verify Weaviate collection removal
