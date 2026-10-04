# Agentic GraphRAG Benchmarking Platform

## Project Purpose
This platform aims to benchmark Agentic GraphRAG pipelines utilizing TigerGraph, Qdrant, and Ollama models.

## Architecture
The platform is built on a zero-cost/local-first strategy, allowing testing and benchmarking on local hardware before migrating to expensive cloud infrastructure.
- **Backend**: FastAPI (Python)
- **Vector DB**: Qdrant (local configuration for Phase 1)
- **Graph DB**: TigerGraph
- **LLM**: Ollama (qwen3:4b locally)
- **Frontend**: React + Vite (Dashboard)

## Zero-Cost/Local-First Strategy
To minimize initial costs, this repository uses open-source components configured for local execution. Ollama powers local LLMs, and Qdrant runs using local storage. Only TigerGraph needs explicit cloud or local container setup with credentials (omitted from repository for security).

## Setup Commands

### 1. Python Virtual Environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r backend/requirements.txt
cd frontend
npm install
cd ..
```

### 3. Environment Variables
Copy `.env.example` to `.env` inside `backend/` and populate it. Do not commit `.env`.
```bash
cp backend/.env.example backend/.env
```

## How to run the application

### Backend
From the `backend` directory:
```bash
cd backend
uvicorn app.main:app --reload
```
Access the API at `http://127.0.0.1:8000`.

### Frontend
From the `frontend` directory (in a new terminal):
```bash
cd frontend
npm run dev
```
Access the Dashboard at `http://localhost:5173`.

## Testing
To run the minimal test suite:
```bash
cd backend
pytest
```
