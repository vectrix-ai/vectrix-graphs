from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from vectrix_graphs.db.weaviate import VectorDB


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
            self.db = VectorDB(mock_logger, type="weaviate")

    def test_init(self, mock_logger):
        with patch("cohere.ClientV2"):
            db = VectorDB(mock_logger, type="weaviate")
            assert db.type == "weaviate"
            assert db.logger == mock_logger

    def test_create_collection(self, mock_logger):
        with patch("cohere.ClientV2"):
            db = VectorDB(mock_logger, type="weaviate")
            db.create_collection("test_collection")
            # Add assertions for Weaviate collection creation

    def test_add_documents(self, mock_logger, sample_documents):
        with patch("cohere.ClientV2"):
            db = VectorDB(mock_logger, type="weaviate")
            db.add_documents(sample_documents)
            # Add assertions to verify documents were added correctly to Weaviate

    def test_similarity_search(self, mock_logger):
        with patch("cohere.ClientV2"):
            db = VectorDB(mock_logger, type="weaviate")
            results = db.similarity_search("test query", k=1)
            assert isinstance(results, list)
            # Add more specific assertions for Weaviate search results

    def test_remove_collection(self, mock_logger):
        with patch("cohere.ClientV2"):
            db = VectorDB(mock_logger, type="weaviate")
            db.remove_collection("test_collection")
            # Add assertions to verify Weaviate collection removal
