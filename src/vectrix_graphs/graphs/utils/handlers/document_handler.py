from typing import List

from langchain_core.documents import Document


class DocumentHandler:
    @staticmethod
    async def filter_duplicates(documents: List[Document]) -> List[Document]:
        """Filter duplicate documents based on UUID."""
        seen_uuids = set()
        return [
            doc
            for doc in documents
            if doc.metadata.get("uuid") not in seen_uuids
            and not seen_uuids.add(doc.metadata.get("uuid"))
        ]

    @staticmethod
    def format_sources(documents: List[Document]) -> str:
        """Format documents into source string."""
        sources = ""
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            url = doc.metadata.get("url", "No URL provided")
            sources += f"{i}. {doc.page_content}\n\nURL: {url}\nSOURCE: {source}\n"
        return sources
