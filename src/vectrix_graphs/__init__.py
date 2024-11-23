from .extract.ner import ExtractMetaData
from .logger import setup_logger

# Initialize global logger
logger = setup_logger(__name__, "INFO")  # You can change the default level as needed

__all__ = [
    "setup_logger",
    "ExtractMetaData",
    "logger",  # Add logger to __all__
]
