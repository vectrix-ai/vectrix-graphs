import json
import time
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from ..schemas.openai import ChatCompletionRequest, ChatCompletionResponse
from ..graphs.default_flow import default_flow
from ..graphs.local_slm_demo import local_slm_demo
from ..graphs.utils.stream_processor import StreamProcessor

router = APIRouter()

def _transform_response(model: str, response: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=f"chatcmpl-{response['messages'][-1].id}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": response["messages"][-1].content},
            "finish_reason": "stop"
        }],
    )


def _transform_messages(request_messages):
    messages = []
    for message in request_messages:
        if message.role == "system":
            messages.append(SystemMessage(content=message.content))
        elif message.role == "assistant":
            messages.append(AIMessage(content=message.content))
        elif message.role == "user":
            messages.append(HumanMessage(content=message.content))
    return messages

@router.post("/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    print(json.dumps(request.model_dump(), indent=4))
    messages = _transform_messages(request.messages)
    print(messages)

    if request.stream:
        if request.model == "navid_ai_demo_local":
            stream = StreamProcessor(local_slm_demo)
            
            async def event_generator():
                async for chunk in stream.process_stream(messages=messages):
                    yield f"data: {chunk}\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")
        
        elif request.model == "navid_ai_demo_online":
            stream = StreamProcessor(default_flow)

            async def event_generator():
                async for chunk in stream.process_stream(messages=messages):
                    yield f"data: {chunk}\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")
        
    else:
        if request.model == "navid_ai_demo_local":
            response = await local_slm_demo.ainvoke({"messages": messages})
            return _transform_response(request.model, response)
        
        elif request.model == "navid_ai_demo_online":
            response = await default_flow.ainvoke(request.model_dump())
            return _transform_response(request.model, response)
        
        else:
            raise ValueError(f"Unsupported model: {request.model}")
  