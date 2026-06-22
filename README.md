# MITRE ATT&CK RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot designed to explore and answer questions based on the MITRE ATT&CK framework. The assistant leverages a modern web interface and state-of-the-art vector search to provide fast, accurate, and context-aware responses to cybersecurity queries.

## Features

- **MITRE ATT&CK Knowledge Base**: Specifically tailored to answer queries about threat tactics, techniques, and procedures (TTPs).
- **Unlimited Retrieval**: Efficient document retrieval using Qdrant vector store.
- **Modern UI**: An intuitive, responsive interface built with Gradio.
- **Multiple LLM Support**: Supports both local models via Ollama and fast cloud inference via Groq APIs.
- **High-Performance Embeddings**: Uses `BAAI/bge-small-en-v1.5` via FastEmbed for accurate semantic search.

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) (if using local models)
- A Groq API Key

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rihem-bs02/attack-rag-assistant.git
   cd attack-rag-assistant
   ```

2. Install the required dependencies:
   ```bash
   pip install gradio qdrant-client fastembed pandas requests python-dotenv
   ```

3. Configure your environment variables:
   Create a `.env` file in the root directory and add your API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

1. Start the assistant:
   ```bash
   python app6.py
   ```
2. Open your browser and navigate to the provided local URL (typically `http://127.0.0.1:7860`).

## Architecture Overview

- **Vector Store**: Qdrant is used to store and query document embeddings.
- **Embeddings**: `fastembed` with BAAI models handles text vectorization.
- **Generation**: Flexible integration with Groq API and local Ollama deployments for text generation.
- **UI**: Gradio powers the interactive chat interface.
