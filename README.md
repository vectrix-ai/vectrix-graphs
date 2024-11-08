# Vectrix Graphs

A powerful graph-based framework for building AI applications with streaming capabilities, local inference, and API integrations.

## Features

- **Multiple Inference Options**
  - Local inference using open-source LLMs
  - API-based inference with Claude 3.5 and OpenAI models
  - Streaming responses for real-time interaction

- **Graph-Based Architecture**
  - Visual graph representation using Mermaid
  - Modular node system for easy customization
  - Built-in document processing and retrieval

- **Vector Database Integration**
  - ChromaDB integration for efficient document storage
  - Similarity search capabilities
  - Metadata extraction and processing

## Prerequisites

### System Requirements
- Python 3.11+
- libmagic: `brew install libmagic`
- Docker (for ChromaDB)

### Services
- ChromaDB server: `docker run -p 7777:8000 chromadb/chroma`
- API keys for Claude/OpenAI (if using API inference)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vectrix-graphs.git
cd vectrix-graphs
```

2. Install dependencies:
```bash
uv venv
source .venv/bin/activate
uv install -e .
```

3. Create a `.env` file with your API keys:
```env
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

## Usage

### Running the API
For development:
```bash
uv run fastapi dev
```

### Example Notebooks

The project includes several example notebooks demonstrating different use cases:

1. `process_streaming.ipynb`: Demonstrates streaming responses
2. `local_inference.ipynb`: Shows local inference using open-source LLMs
3. `api_inference.ipynb`: Examples using Claude 3.5 and OpenAI APIs
4. `tutorial.ipynb`: Step-by-step guide to using the framework

### Basic Code Example

```python
import sys
from IPython.display import Markdown, display, Image
from langchain_core.messages import HumanMessage
from vectrix_graphs import local_slm_demo

# Display the graph visualization
display(Image(local_slm_demo.get_graph().draw_mermaid_png()))

# Ask a question
input = [HumanMessage(content="Your question here")]
response = await local_slm_demo.ainvoke({"messages": input})
print(response['messages'][-1].content)
```

## Project Structure

```
vectrix-graphs/
├── src/
│   └── vectrix_graphs/
│       ├── graphs/
│       ├── db/
│       └── utils/
├── examples/
│   ├── process_streaming.ipynb
│   ├── local_inference.ipynb
│   ├── api_inference.ipynb
│   └── tutorial.ipynb
└── README.md
```

## Development

For development purposes, you can run the API with hot reloading:
```bash
uv run fastapi dev
```

## Notes

- The local inference example uses publicly available LLMs through TogetherAI's hosting service
- For simple classification tasks, GPT-4-mini is recommended for cost efficiency
- ChromaDB must be running for vector database functionality

## License

Ben Selleslagh - Vectrix BV - 2024