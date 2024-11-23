from typing import Annotated, List, Sequence

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from vectrix_graphs.db.weaviate import Weaviate

from ..base_nodes import BaseNodes


class MultiModalRetrievalState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    results: List[Document]
    base_64_images: List[str]


class RAGNodes(BaseNodes):
    def __init__(self, logger, mode="online", document_handler=None):
        super().__init__(logger, mode)
        self.weaviate = Weaviate()
        self.mode = mode

    async def multi_modal_retrieval(self, state: MultiModalRetrievalState, config):
        collection_name = config.get("configurable", {}).get("collection_name")
        self.weaviate.set_collection(collection_name)

        print("Running multi-modal retrieval")
        print(f"Searching for {state['messages'][-1].content}")

        results = self.weaviate.similarity_search(
            query=state["messages"][-1].content, k=3, type="multimodal"
        )
        return {"results": results}

    async def answer_question(self, state: MultiModalRetrievalState, config):
        print("Answering question")
        llm = self.llm_factory.create_llm(mode=self.mode, model_type="default")
        include_images = config.get("configurable", {}).get("include_images", False)
        chain = self.chain_factory.create_multi_modal_chain(
            llm,
            state["messages"][-1].content,
            state["results"],
            include_images=include_images,
        )
        response = await chain.ainvoke({})

        message = AIMessage(content=response)
        return {"messages": message}
