import json
import time
import uuid

from langsmith import Client

from ...logger import setup_logger


class StreamProcessor:
    """
    A class for processing streaming events from a langgraph.

    This class initializes a LangSmith Client and provides methods to process
    streaming events from a graph, handling various event types and yielding
    progress updates, streamed data, and final outputs.

    Attributes:
        client (Client): An instance of the LangSmith Client.

    Methods:
        process_stream(graph, question): Asynchronously processes the stream of events
        from the given graph for the provided question.
    """

    def __init__(self, graph):
        self.client = Client()
        self.logger = setup_logger("stream_processor")
        self.graph = graph
        self.session_id = str(uuid.uuid4())

    async def process_stream(self, messages):
        """
        Asynchronously processes the stream of events from the given graph for the provided question.

        This method yields progress updates, streamed data, and final outputs as the graph processes
        the question. It handles various event types, including chat model streaming and chain end events.

        Args:
            graph: The langgraph object to process events from.
            messages (list): The messages to be processed by the graph.

        Yields:
            dict: A dictionary containing one of the following keys:
                - "progress": A string indicating the current step being processed.
                - "data": A string containing streamed chunks of the answer.
                - "final_output": The final generated answer.
            str: The URL of the completed run in LangSmith.

        Note:
            This method uses the LangSmith Client to retrieve the final run URL.
        """

        config = {"configurable": {}}

        # Generate a unique ID for this chat completion
        chat_id = f"chatcmpl-{self.session_id}"

        # Yield the initial chunk
        yield json.dumps(
            {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "demo",
                "system_fingerprint": f"fp_{self.session_id[:8]}",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
        )

        async for event in self.graph.astream_events(
            {"messages": messages}, version="v1", config=config
        ):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                if event["metadata"]["langgraph_node"] in ["llm_answer", "rag_answer"]:
                    chunk_content = event["data"]["chunk"].content

                    yield json.dumps(
                        {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": "demo",
                            "system_fingerprint": f"fp_{self.session_id[:8]}",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk_content},
                                    "logprobs": None,
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )

        # Add a final chunk to indicate completion
        yield json.dumps(
            {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "demo",
                "system_fingerprint": f"fp_{self.session_id[:8]}",
                "choices": [
                    {"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}
                ],
            }
        )
