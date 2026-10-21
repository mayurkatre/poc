# Documentation

## Overview

This codebase appears to be a software project focused on integrating and managing data interactions, particularly with a vector database, as indicated by the presence of Pinecone-related documentation and architectural decision records (ADRs). It includes functionality for querying, ingesting, and streaming data, suggesting a system designed for efficient data retrieval and processing. The project emphasizes structured decision-making, as evidenced by the detailed ADRs on topics like retrieval and chunking strategies.

The primary purpose of this project is to solve challenges related to data storage, retrieval, and processing, particularly in scenarios where vector databases are utilized. It likely aims to streamline workflows for querying and ingesting data while ensuring scalability and performance. The inclusion of API routes like `/query`, `/query/stream`, and `/ingest` highlights its focus on enabling seamless data interactions.

The tech stack is predominantly Python, with a smaller contribution from TypeScript, indicating a backend-heavy system with some frontend or tooling components. No specific frameworks were detected, suggesting a custom or lightweight implementation tailored to the project’s needs. The use of `.env` files and detailed setup documentation points to a focus on configurability and ease of deployment.

The likely target audience includes developers, data engineers, or organizations working with vector databases or similar data-intensive systems. End-users may include technical teams seeking efficient solutions for data retrieval, ingestion, and streaming, particularly in contexts where performance and scalability are critical.

## Architecture

The project appears to be a sophisticated system designed for information retrieval and processing, likely involving a combination of machine learning and web technologies. Below is a detailed architecture overview based on the provided directory layout:

### 1. Overall System Design

The system is structured into several key components, each responsible for a specific aspect of the overall functionality. The main components include:

- **API Layer**: Handles incoming requests and serves as the interface for client interactions.
- **Configuration Management**: Manages settings and configurations for the system.
- **Data Ingestion and Processing**: Responsible for loading, chunking, and embedding documents.
- **Retrieval and Ranking**: Implements strategies for retrieving and ranking relevant documents.
- **Evaluation**: Provides mechanisms to evaluate the performance of the retrieval and ranking strategies.
- **Frontend**: A web-based interface for users to interact with the system.
- **Deployment and Environment Setup**: Scripts and documentation for setting up and deploying the system.

### 2. Core Components Interaction

- **API Layer (`api/`)**: The `app.py` file likely contains the main application logic for handling HTTP requests. It interacts with other components like retrieval and ranking to process queries and return results.

- **Configuration (`config/`)**: The `settings.yaml` file and related Python modules manage configuration settings, likely loaded at runtime to configure various components like database connections or API keys.

- **Data Ingestion (`ingestion/`)**: This component includes scripts for loading documents (`document_loader.py`), chunking them into manageable pieces (`chunking.py`), and generating embeddings (`embedding_pipeline.py`) for use in retrieval.

- **Retrieval and Ranking (`retrieval/` and `reranking/`)**: The retrieval component implements various strategies, such as hybrid search and maximum marginal relevance (MMR), to fetch relevant documents. The reranking component, with a cross-encoder, refines the order of retrieved documents based on relevance.

- **Evaluation (`evaluation/`)**: Contains scripts and datasets for evaluating the performance of retrieval strategies using different metrics.

- **Frontend (`frontend/`)**: A TypeScript-based web application that provides a user interface for interacting with the system. It likely communicates with the API to submit queries and display results.

### 3. Data Flow or Execution Path

1. **Ingestion**: Documents are loaded and processed through the ingestion pipeline, where they are chunked and embedded.
2. **Storage**: Processed data is stored in a vector database, as suggested by the presence of Pinecone-related documentation.
3. **Query Handling**: The API receives queries from the frontend, which are then processed by the retrieval component to fetch relevant documents.
4. **Reranking**: Retrieved documents are reranked to improve relevance before being returned to the frontend.
5. **Evaluation**: The evaluation component can be used to assess the effectiveness of retrieval and ranking strategies.

### 4. Design Patterns and Libraries

- **Microservices Architecture**: The separation of concerns into distinct components (API, ingestion, retrieval, etc.) suggests a microservices-like architecture, promoting modularity and scalability.
- **Pipeline Pattern**: The ingestion and processing of data follow a pipeline pattern, where data flows through a series of transformations (loading, chunking, embedding).
- **Configuration Management**: Use of YAML and Python scripts for managing configurations indicates a centralized configuration management approach.
- **Frontend-Backend Separation**: The use of a separate frontend directory with TypeScript and a backend with Python suggests a clear separation between client-side and server-side logic.

Overall, the system is designed to efficiently handle large volumes of data, process it for retrieval, and provide a user-friendly interface for interaction, leveraging modern software design principles and technologies.

## Installation

# Installation Guide

Follow these steps to clone, install, configure, and execute the codebase locally. This guide covers both the frontend (React) and backend (FastAPI) components of the project.

## Prerequisites

Before you begin, ensure you have the following installed on your machine:

- **Node.js** (version 14 or higher)
- **Python** (version 3.8 or higher)
- **pip** (Python package manager)
- **npm** or **yarn** (JavaScript package managers)

## Step-by-Step Installation

### 1. Clone the Repository

Open your terminal and run the following command to clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

Replace `<repository-url>` with the actual URL of the repository and `<repository-directory>` with the name of the cloned directory.

### 2. Install Frontend Dependencies

#### Using npm

1. Navigate to the frontend directory (if applicable):

   ```bash
   cd frontend
   ```

2. Install the dependencies using npm:

   ```bash
   npm install
   ```

#### Using yarn

1. Navigate to the frontend directory (if applicable):

   ```bash
   cd frontend
   ```

2. Install the dependencies using yarn:

   ```bash
   yarn install
   ```

### 3. Install Backend Dependencies

1. Navigate to the backend directory (if applicable):

   ```bash
   cd backend
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:

   - **On Windows:**

     ```bash
     venv\Scripts\activate
     ```

   - **On macOS/Linux:**

     ```bash
     source venv/bin/activate
     ```

4. Install the dependencies using pip:

   ```bash
   pip install -r requirements.txt
   ```

### 4. Configure Environment Variables

1. Create a `.env` file in the backend directory (if it doesn't exist) and add the necessary environment variables. An example `.env` file might look like this:

   ```plaintext
   DATABASE_URL=<your-database-url>
   SECRET_KEY=<your-secret-key>
   ```

   Replace `<your-database-url>` and `<your-secret-key>` with your actual values.

### 5. Run Database Migrations (if applicable)

If your project requires database migrations, run the following command:

```bash
# Example command, replace with your actual migration command
alembic upgrade head
```

### 6. Start the Backend Server

Run the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload
```

Replace `main:app` with the appropriate module and app name if different.

### 7. Start the Frontend Development Server

#### Using npm

1. Navigate back to the frontend directory (if you are not already there):

   ```bash
   cd frontend
   ```

2. Start the development server:

   ```bash
   npm start
   ```

#### Using yarn

1. Navigate back to the frontend directory (if you are not already there):

   ```bash
   cd frontend
   ```

2. Start the development server:

   ```bash
   yarn start
   ```

### 8. Access the Application

- Open your web browser and go to `http://localhost:3000` (or the port specified in your frontend configuration) to access the frontend application.
- The backend API should be accessible at `http://localhost:8000` (or the port specified in your Uvicorn command).

### 9. Run Tests (Optional)

To run the tests for the backend, execute the following command:

```bash
pytest
```

### Conclusion

You have successfully cloned, installed, configured, and executed the codebase locally. If you encounter any issues, refer to the project documentation or seek help from the community.

## Project Structure

```text
- /.env.example (1 files)
- /.gitignore (1 files)
- /PINECONE_INTEGRATION_SUMMARY.md (1 files)
- /PINECONE_SETUP.md (1 files)
- /PUSH_STATUS.md (1 files)
- /README.md (1 files)
- /RUN_FULLSTACK.md (1 files)
- /adr (5 files)
- /api (2 files)
- /config (3 files)
- /dataSavingfacility.md (1 files)
- /deployment.md (1 files)
- /documents (2 files)
- /evaluation (4 files)
- /frontend (11 files)
- /generation (2 files)
- /ingest.py (1 files)
- /ingestion (4 files)
- /query.py (1 files)
- /requirements.txt (1 files)
- /reranking (2 files)
- /retrieval (5 files)
- /start-fullstack.ps1 (1 files)
- /start-fullstack.sh (1 files)
```

## Code Reference

### `_run_ingest`

Run the ingestion process for a given source.

**Arguments:**
- `source` (str): The source path or URL to ingest.

**Returns:** `None` - No return value.

**Side Effects:** Processes the source for ingestion.

**Example:**
```javascript
_run_ingest('path/to/source')
```

### `get`

Retrieve a specific config value.

**Arguments:**
- `section` (str): Top-level config section (e.g. 'retrieval').
- `key` (str): Key within the section.
- `default` (Any): Fallback value if not found.

**Returns:** `Any` - Config value or default.

**Side Effects:** Accesses cached settings data.

**Example:**
```javascript
get('database', 'host', 'localhost')
```

### `get_streaming_client`

Get the OpenRouter client for streaming responses.

**Returns:** `openai.OpenAI` - Client configured for OpenRouter.

**Side Effects:** Utilizes get_client() to obtain the client instance.

**Example:**
```javascript
streaming_client = get_streaming_client()
```

### `run`

Run the full RAG evaluation pipeline.

**Arguments:**
- `dataset` (str): Path to the evaluation dataset.
- `output` (str): Path for the results JSON file.
- `report` (str): Path for the Markdown report.
- `limit` (Optional[int]): Limit the number of samples to evaluate.
- `skip_generation_eval` (bool): Flag to skip LLM-based generation evaluation for faster execution.

**Returns:** `None` - This function does not return a value.

**Side Effects:** Prints evaluation harness information to the console., Loads configuration settings and builds the RAG pipeline.

**Example:**
```javascript
run(dataset='evaluation/dataset.json', output='evaluation/results.json', report='evaluation/report.md')
```

### `avg`

Calculates the average of a list of values.

**Arguments:**
- `vals` (list[float]): List of values to average.

**Returns:** `float` - Average of the values.

**Example:**
```javascript
average = avg([1.0, 2.0, 3.0])
```

### `handleClear`

Clears the current question and resets the query mutation.

**Returns:** `void` - No return value.

**Side Effects:** Resets the question state to an empty string., Resets the query mutation state.

**Example:**
```javascript
handleClear();
```

### `__init__`

Initializes the RAGPipeline object with specified components and settings.

**Arguments:**
- `self` (RAGPipeline): The instance of the RAGPipeline class.
- `retriever` (BaseRetriever): The retriever component for retrieving documents.
- `reranker` (Union[CrossEncoderReranker, PassThroughReranker]): The reranker component for reranking retrieved documents.
- `llm_model` (str): The language model to use for generation, default is 'openai/gpt-4o-mini'.
- `temperature` (float): The temperature setting for the language model, default is 0.0.
- `max_tokens` (int): The maximum number of tokens for generation, default is 1024.
- `streaming` (bool): Whether to enable streaming responses, default is False.
- `query_rewriting_enabled` (bool): Whether to enable query rewriting, default is True.
- `cache_enabled` (bool): Whether to enable response caching, default is True.

**Returns:** `None` - This constructor does not return a value.

**Side Effects:** Initializes the RAGPipeline with the provided components and settings.

**Example:**
```javascript
rag_pipeline = RAGPipeline(retriever=my_retriever, reranker=my_reranker)
```

### `ingest`

Ingest documents into the RAG vector store.

**Arguments:**
- `source` (str): File path, directory, or URL to ingest.
- `strategy` (Optional[str]): Chunking strategy: semantic, sentence, fixed.
- `chunk_size` (Optional[int]): Chunk size in characters.
- `overlap` (Optional[int]): Chunk overlap in characters.
- `vector_store_provider` (Optional[str]): Vector store: faiss or chroma.
- `dry_run` (bool): Show what would be ingested without indexing.

**Returns:** `None` - This function does not return a value.

**Side Effects:** Prints various messages to the console., Raises a typer.Exit exception if no documents are found or in dry run mode.

**Example:**
```javascript
ingest('./documents', strategy='fixed', chunk_size=256)
```

### `create`

Factory method to create a chunker based on strategy.

**Arguments:**
- `cls` (type): Class reference for creating an instance.
- `strategy` (str): Chunking strategy to use.
- `chunk_size` (int): Target character length per chunk.
- `overlap` (int): Number of characters to repeat between adjacent chunks.

**Returns:** `BaseChunker` - Instance of a chunker based on the specified strategy.

**Side Effects:** None

**Example:**
```javascript
BaseChunker.create(strategy='semantic', chunk_size=512, overlap=64)
```

### `get_loader`

Retrieve the appropriate loader for a given source.

**Arguments:**
- `cls` (type): The class itself, used for class method.
- `source` (str): File path or URL to determine the loader for.

**Returns:** `BaseLoader` - An instance of a loader class suitable for the source.

**Example:**
```javascript
loader = LoaderRegistry.get_loader('path/to/source')
```

### `create_embedder`

Factory function for creating embedders based on the specified provider.

**Arguments:**
- `provider` (str): Provider type, e.g., 'sentence_transformers'.
- `model_name` (str): Model name for the embedder.
- `cache_enabled` (bool): Flag to enable or disable caching.
- `cache_dir` (str): Directory to store cache files.

**Returns:** `BaseEmbedder` - An instance of a BaseEmbedder subclass.

**Example:**
```javascript
embedder = create_embedder(provider='sentence_transformers', model_name='all-MiniLM-L6-v2')
```

### `query`

Query the RAG system.

**Arguments:**
- `question` (Optional[str]): Question to ask the RAG system.
- `interactive` (bool): Flag to enable interactive mode.
- `hyde` (bool): Flag to use HyDE retrieval.
- `no_rerank` (bool): Flag to disable reranking.
- `top_k` (Optional[int]): Number of sources to return.
- `json_output` (bool): Flag to output raw JSON.

**Returns:** `None` - This function does not return a value.

**Side Effects:** Initializes the RAG pipeline., Handles interactive query mode.

**Example:**
```javascript
query(question='Explain HyDE', interactive=False, hyde=True, no_rerank=False, top_k=3, json_output=False)
```

### `_embed_query`

Embeds the query string into a vector using the embedding pipeline.

**Arguments:**
- `self` (BaseRetriever): Instance of the BaseRetriever class.
- `query` (str): The query string to be embedded.

**Returns:** `list[float]` - A list of floats representing the embedded query vector.

**Example:**
```javascript
embedded_query = retriever._embed_query('example query')
```

### `retrieve`

Retrieves using HyDE: generate hypothesis → embed → search.

**Arguments:**
- `self` (HyDERetriever): The instance of the class.
- `query` (str): User query.

**Returns:** `RetrievalResult` - RetrievalResult with matched chunks.

**Side Effects:** Logs the retrieval process and the generated hypothesis.

**Example:**
```javascript
result = retriever.retrieve('What are the benefits of renewable energy?')
```

## Dependencies

| Name | Version | Description |
|---|---|---|
| react | ^18.2.0 | Project dependency type: prod |
| react-dom | ^18.2.0 | Project dependency type: prod |
| axios | ^1.6.0 | Project dependency type: prod |
| @tanstack/react-query | ^5.0.0 | Project dependency type: prod |
| react-markdown | ^9.0.0 | Project dependency type: prod |
| remark-gfm | ^4.0.0 | Project dependency type: prod |
| lucide-react | ^0.294.0 | Project dependency type: prod |
| @types/react | ^18.2.37 | Project dependency type: dev |
| @types/react-dom | ^18.2.15 | Project dependency type: dev |
| @vitejs/plugin-react | ^4.2.0 | Project dependency type: dev |
| typescript | ^5.2.2 | Project dependency type: dev |
| vite | ^5.0.0 | Project dependency type: dev |
| fastapi | 0.111.0 | Project dependency type: prod |
| uvicorn[standard] | 0.29.0 | Project dependency type: prod |
| pydantic | 2.7.1 | Project dependency type: prod |
| pydantic-settings | 2.2.1 | Project dependency type: prod |
| openai | 1.30.1 | Project dependency type: prod |
| sentence-transformers | 2.7.0 | Project dependency type: prod |
| tiktoken | 0.7.0 | Project dependency type: prod |
| faiss-cpu | 1.8.0 | Project dependency type: prod |

