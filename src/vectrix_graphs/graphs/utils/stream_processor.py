from langsmith import Client


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
    def __init__(self, graph, logger):
        self.client = Client()
        self.logger = logger
        self.graph = graph

    async def process_stream(self, messages):
        """
        Asynchronously processes the stream of events from the given graph for the provided question.

        This method yields progress updates, streamed data, and final outputs as the graph processes
        the question. It handles various event types, including chat model streaming and chain end events.

        Args:
            graph: The langgraph object to process events from.
            question (str): The question to be processed by the graph.

        Yields:
            dict: A dictionary containing one of the following keys:
                - "progress": A string indicating the current step being processed.
                - "data": A string containing streamed chunks of the answer.
                - "final_output": The final generated answer.
            str: The URL of the completed run in LangSmith.

        Note:
            This method uses the LangSmith Client to retrieve the final run URL.
        """

        config = {"configurable": {
        }}

        
        async for event in self.graph.astream_events({"messages": messages}, version="v1", config=config):
            kind = event["event"]

            #print(event)

            if kind == "on_chat_model_stream":
                if event["metadata"]["langgraph_node"] in ["llm_answer", "final_answer", 'rag_answer', "rewrite_last_message"]:
                    #chunk_content = event["data"]["chunk"].content
                    #print(chunk_content, end='', flush=True)  # Print on the same line and flush the output

                    # OpenAI API format
                    '''
                    yield {
                        "id": event["run_id"],
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": event["metadata"]["ls_model_name"],
                        "system_fingerprint": "fp_" + event["run_id"][:8],
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk_content},
                            "logprobs": None,
                            "finish_reason": None
                        }]
                    }
                    '''
            if kind == "on_chain_stream":
                if event['name'] == 'final_answer':
                    yield(event['data']['chunk']['messages'].content)
                    pass