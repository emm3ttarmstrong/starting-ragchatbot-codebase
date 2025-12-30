# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Course Materials RAG (Retrieval-Augmented Generation) system - a full-stack web application that answers questions about course materials using semantic search and AI-powered responses.

## Commands

### Run the application
```bash
cd starting-ragchatbot-codebase
./run.sh
# Or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Install dependencies
```bash
cd starting-ragchatbot-codebase
uv sync
```

The app runs at http://localhost:8000 with API docs at http://localhost:8000/docs

### Code quality checks
```bash
# Check formatting and linting (no changes made)
./lint.sh check

# Auto-fix formatting issues
./lint.sh fix
```

Tools used: black (formatting), isort (import sorting), flake8 (linting)

## Architecture

### Query Flow
```
User Question → FastAPI /api/query → RAGSystem → Claude with Tools → VectorStore Search → Claude Synthesizes Answer
```

### Key Components

- **RAGSystem** (`backend/rag_system.py`): Main orchestrator that coordinates the query pipeline
- **AIGenerator** (`backend/ai_generator.py`): Manages Claude API interactions with tool-calling flow (query → tool_use → tool_results → final_answer)
- **VectorStore** (`backend/vector_store.py`): ChromaDB interface with two collections:
  - `course_catalog`: Course metadata for name resolution
  - `course_content`: Text chunks with embeddings for semantic search
- **DocumentProcessor** (`backend/document_processor.py`): Parses course files, extracts lessons, and chunks text
- **SearchTools** (`backend/search_tools.py`): Defines `search_course_content` tool that Claude calls to query the vector database

### Tool-Based Architecture

Claude uses tool calling to execute searches. The `CourseSearchTool` accepts:
- `query` (required): What to search
- `course_name` (optional): Filter by course
- `lesson_number` (optional): Filter by lesson

### Configuration

Key settings in `backend/config.py`:
- Embedding model: `all-MiniLM-L6-v2`
- Claude model: `claude-sonnet-4-20250514`
- Chunk size: 800 chars with 100 char overlap
- Max search results: 5
- Conversation memory: 2 exchanges

### Document Format

Course files in `/docs` follow this format:
```
Course Title: [title]
Link: [url]
Instructor: [name]

Lesson 1: [title]
[content]

Lesson 2: [title]
[content]
```

## Environment

Requires `ANTHROPIC_API_KEY` in `.env` file.
