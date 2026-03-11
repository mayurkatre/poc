# Deployment Guide

Deploying this Full-Stack RAG application involves combining the React frontend and the FastAPI Python backend into a single production-ready package.

Currently, in development, you are running two separate development servers (`uvicorn` and `vite`). For production deployment, you want to compile the frontend into static files and have the Python backend serve everything.

Here is the step-by-step guide to deploying this project:

## Step 1: Build the React Frontend
First, you need to compile the Vite/React application into raw, optimized HTML/CSS/JS files for production.

1. Open your terminal and navigate to the frontend folder: `cd frontend`
2. Run the build command: `npm run build`
3. This will create a `dist/` folder inside your `frontend` directory containing the optimized production files.

## Step 2: Configure FastAPI to Serve the Frontend
Next, you need to update the Python backend to serve those built frontend files so you only need to deploy **one** server.

1. Open `api/app.py`
2. Add the `StaticFiles` import at the top of the file:
   ```python
   from fastapi.staticfiles import StaticFiles
   ```
3. Scroll to the very bottom of `api/app.py` and mount the Vite `dist/` directory to the root of the API:
   ```python
   # Important: Do this AFTER defining all your API routes!
   app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
   ```

## Step 3: Choose a Hosting Platform
Because this system requires running heavy python machine-learning models locally (SentenceTransformers for embeddings) and uses the local filesystem for the Vector Database (`.cache/`), you need a host that supports **Python, Persistent Disk Storage, and decent RAM**.

Here are the best options for this specific architecture:

### Option A: Render.com or Railway.app (Easiest)
These modern platforms are perfect for FastAPI apps.

1. Push your repository to GitHub.
2. Connect the repo to Render or Railway and select "Web Service".
3. **Build Command:** `pip install -r requirements.txt && cd frontend && npm install && npm run build`
4. **Start Command:** `uvicorn api.app:app --host 0.0.0.0 --port $PORT`
5. **Crucial:** Make sure you attach a "Persistent Disk" (volume) mapping to the `.cache/` folder, or your vector database will be wiped every time the server restarts!

### Option B: Standard VPS (DigitalOcean Droplet / AWS EC2)
If you want full control and cheaper RAM for the embeddings model:

1. Spin up an Ubuntu server.
2. Clone your repository.
3. Install Python 3.10+, Node.js, and Nginx.
4. Run the app using a process manager like `Gunicorn` or `Supervisor` to keep the FastAPI server running 24/7.
5. Use Nginx as a reverse proxy to route traffic to port `8000`.

## Step 4: Environment Variables
Wherever you deploy it, do not forget to safely add your `.env` variables to the host's environment settings:

```env
OPENAI_API_KEY=sk-or-v1-...
```
