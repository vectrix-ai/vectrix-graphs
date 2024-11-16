from typing import List, Literal

from langchain import hub
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether
from pytz import UTC


class ExtractMetaData:
    """
    This class is used to extract information from a document.

    args:
        llm: langchain llm object
    """

    def __init__(self, logger, model):
        self.model = model
        if model == "gpt-4o-mini":
            self.llm_with_tools = ChatOpenAI(model=model, temperature=0)
        elif model == "llama3.1-8B":
            self.llm_with_tools = ChatOllama(model="llama3.1", temperature=0)
        elif model == "llama3.1-70B":
            self.llm_with_tools = ChatTogether(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", temperature=0
            )
        else:
            raise ValueError(f"Model {self.model} not supported")
        self.prompt = hub.pull("entity_extraction")
        self.logger = logger

    @staticmethod
    def _calculate_word_count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def _calculate_read_time(word_count: int) -> float:
        return word_count / 200

    @staticmethod
    def _format_last_modified(last_modified):
        if not last_modified:
            return ""
        if isinstance(last_modified, str):
            return last_modified
        try:
            return last_modified.replace(tzinfo=UTC).isoformat()
        except AttributeError:
            return str(last_modified)

    def extract(
        self,
        documents: List[Document],
        source: Literal[
            "webpage", "uploaded_file", "OneDrive", "Notion", "chrome_extension"
        ],
    ) -> List[Document]:
        self.logger.info(
            f"Extracting metadata from {len(documents)} documents, using {self.model}"
        )
        ner_chain = self.prompt | self.llm_with_tools
        content_list = [{"content": doc.page_content} for doc in documents]

        try:
            responses = ner_chain.batch(content_list)
        except Exception as e:
            self.logger.error(f"Error during batch processing: {str(e)}")
            responses = [None] * len(documents)

        results = []

        # Merge the responses with the documents
        for doc, response in zip(documents, responses):
            doc_metadata = {
                "filename": doc.metadata["filename"],
                "filetype": doc.metadata["filetype"],
                "author": response.get("author", "") if response else "",
                "source": source,
                "word_count": self._calculate_word_count(doc.page_content),
                "language": response.get("language", "") if response else "",
                "content_type": response.get("content_type", "") if response else "",
                "tags": str(response.get("tags", "")) if response else "",
                "summary": response.get("summary", "") if response else "",
                "read_time": self._calculate_read_time(
                    self._calculate_word_count(doc.page_content)
                ),
                "last_modified": self._format_last_modified(
                    doc.metadata["last_modified"]
                ),
            }
            results.append(
                Document(page_content=doc.page_content, metadata=doc_metadata)
            )
        return results
