from unstructured.partition.auto import partition
from langchain_core.documents import Document
from typing import List

class ExtractDocuments():
    def __init__(self, logger):
        self.logger = logger

    def extract(self, file_path : str) -> List[str]:
        '''
        This function extracts documents from a file.
        '''
        self.logger.info(f"Extracting documents from {file_path}")
        elements = partition(
            filename=file_path, 
            chunking_strategy="by_title",
            max_characters=1000
            )
        
        documents = [Document(
            page_content=element.text,
            metadata=element.metadata.to_dict()
        ) for element in elements]
        return documents