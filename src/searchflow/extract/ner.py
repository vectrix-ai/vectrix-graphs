from typing import Literal, Optional, List
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain import hub
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from pytz import UTC


class ExtractionObject(BaseModel):
    title: str
    content: str
    url: str
    project_name: str
    file_type: Literal["webpage", "pdf", "docx", "txt", "csv", "other"]
    source: Literal["webpage", "uploaded_file", "OneDrive", "Notion", "chrome_extension"]
    filename: Optional[str] = ""
    creation_date: Optional[datetime] = None
    last_modified_date: Optional[datetime] = None
    

class ExtractMetaData:
    '''
    This class is used to extract information from a document.

    args:
        llm: langchain llm object
    '''
    def __init__(self, logger, model):
        self.model = model
        if model == "gpt-4o-mini":
            self.llm_with_tools = ChatOpenAI(model=model, temperature=0)
        elif model == "llama3.1":
            self.llm_with_tools = ChatOllama(model=model, temperature=0)
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

    def extract(self, documents: List[ExtractionObject], source: Literal["webpage", "uploaded_file", "OneDrive", "Notion", "chrome_extension"]) -> List[Document]:
        self.logger.info(f"Extracting metadata from {len(documents)} documents, using {self.model}")
        chain = self.prompt | self.llm_with_tools
        content_list = [{"content" : doc.page_content} for doc in documents]
        responses =  chain.batch(content_list)

        results = []

        # Merge the responses with the documents
        for doc, response in zip(documents, responses):
            doc_metadata = {
                "filename": doc.metadata.get("filename", ""),
                "filetype": doc.metadata.get("filetype", ""),
                "author": response.get("author") or "",
                "source": source,
                "word_count": self._calculate_word_count(doc.page_content),
                "language": response.get("language") or "",
                "content_type": response.get("content_type") or "",
                "tags": str(response.get("tags") or ""),
                "summary": response.get("summary") or "",
                "read_time": self._calculate_read_time(self._calculate_word_count(doc.page_content)),
                "last_modified": doc.metadata.get("last_modified", ""),
            }
            results.append(Document(
                page_content=doc.page_content,
                metadata=doc_metadata
            ))
        return results