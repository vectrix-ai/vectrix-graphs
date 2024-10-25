from .extract.documents import ExtractDocuments
from .logger import setup_logger
from .extract.ner import ExtractMetaData
from vectrix_graphs.db.vectordb import VectorDB

# Initialize the VectorDB object
vectordb = VectorDB(setup_logger(name="VectorDB"))

from .graphs.default_flow import default_flow
from .graphs.local_slm_demo import local_slm_demo

__all__ = ["ExtractDocuments", "setup_logger", "ExtractMetaData", "default_flow", "local_slm_demo", "vectordb"]