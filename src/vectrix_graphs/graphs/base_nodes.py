from abc import ABC

from .utils.models.chain_factory import ChainFactory
from .utils.models.llm_factory import LLMFactory


class BaseNodes(ABC):
    def __init__(self, logger, config):
        self.logger = logger
        self.llm_factory = LLMFactory()
        self.chain_factory = ChainFactory()
