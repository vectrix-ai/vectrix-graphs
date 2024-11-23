from typing import Any, List

from langchain import hub
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from vectrix_graphs.logger import setup_logger

logger = setup_logger(__name__, level="INFO")


class ChainFactory:
    @staticmethod
    def create_langsmith_chain(llm, prompt_uri, tools: list[Any] | None = None):
        """Create LangSmith chain."""
        prompt = hub.pull(prompt_uri)
        if tools:
            return prompt | llm.bind_tools(tools=tools)
        else:
            return prompt | llm

    @staticmethod
    def create_multi_modal_chain(
        llm, prompt, documents: List[Document], include_images: bool
    ):
        image_messages = create_image_messages(documents) if include_images else []

        human_message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Given are search results for a given query and the extracted text. "
                    "You should answer the question given the user query and the search results."
                    "The search results can contain extracted text and images.\n\n"
                    f"User Question:\n{prompt}\n\n"
                    f"Search Results:\n{[document.page_content for document in documents]}",
                },
                *image_messages,
            ],
        )

        prompt = ChatPromptTemplate.from_messages([human_message])

        chain = prompt | llm | StrOutputParser()
        return chain.with_config({"run_name": f"Order Extraction - {llm.model_name}"})


def create_image_messages(documents):
    """
    Creates image messages for all images found in the provided documents.
    Expects images to be stored as a list in document.metadata['images'].
    Args:
        documents: List of documents containing image metadata
    Returns:
        List of image message dictionaries
    """
    logger.info(f"Creating image messages for {len(documents)} documents")
    image_messages = []

    for document in documents:
        if not document.metadata:
            continue

        if not document.metadata.get("image_data"):
            continue

        images = document.metadata["image_data"]
        for image_data in images:
            image_messages.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"},
                }
            )

    return image_messages
