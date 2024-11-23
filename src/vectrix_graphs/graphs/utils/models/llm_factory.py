from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether


class LLMFactory:
    @staticmethod
    def create_llm(mode: Literal["local", "online"], model_type: str, **kwargs):
        """
        Factory method to create LLM instances.
        """
        if mode == "online":
            models = {
                "default": lambda: ChatOpenAI(model_name="gpt-4o", **kwargs),
                "": lambda: ChatAnthropic(
                    model_name="claude-3-5-sonnet-20241022", **kwargs
                ),
                "mini": lambda: ChatOpenAI(model="gpt-4o-mini", **kwargs),
            }
        else:
            models = {
                "default": lambda: ChatTogether(
                    model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", **kwargs
                ),
                "turbo": lambda: ChatTogether(
                    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", **kwargs
                ),
            }

        return models.get(model_type, models["default"])()
