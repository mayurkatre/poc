# Data Storage & Local Caching

In this project, all the ingested information, processed data, and cached responses are stored completely locally on your machine inside the `.cache` directory located at the root of the project (`rag-poc-updated/.cache/`).

Because this is a decoupled Proof of Concept designed to run without expensive cloud infrastructure, it relies on file-based storage. Here is exactly where and how the different types of data are stored inside that folder:

## 1. The Vector Database
When you run the ingestion script (`python ingest.py`), the system breaks your documents (like PDFs or Markdown files) into chunks and converts them into mathematical embeddings.

- **Where it lives:** `.cache/faiss_index/index.faiss` and `.cache/faiss_index/chunks.pkl`
- **What it is:** This is your primary database. FAISS (Facebook AI Similarity Search) saves the multidimensional vectors here. The `.pkl` file holds the actual raw text mapped to those vectors so the system can retrieve the readable paragraphs later.

## 2. The Embedding Cache
Computing embeddings takes time and CPU power. To make the system faster, the project locally caches every single text string it has ever embedded.

- **Where it lives:** `.cache/embeddings/cache.pkl`
- **What it is:** If you ingest a document, delete it, and ingest it again, the system recognizes the exact text and instantly loads the vector from this cache instead of running the heavy machine-learning model (SentenceTransformers) a second time.

## 3. The LLM Response Cache
To save money on API calls (like OpenRouter) and make the frontend feel lightning-fast for repeated questions, the system caches the final answers.

- **Where it lives:** `.cache/responses/`
- **What it is:** When a user asks a question like "What is RAG?", the system hashes the question. The next time anyone asks that exact question, it bypasses the LLM entirely and serves the saved response directly from this folder.

## Clearing the Database
**What happens if I want to "Wipe the Database"?**

Because everything is stored in these local files, resetting the entire database and starting from scratch is incredibly easy. You don't need to run any complex SQL drop commands—you simply delete the `.cache` folder from your file explorer. The next time you run the application or ingest a document, the system will automatically recreate the folder from a blank slate.
