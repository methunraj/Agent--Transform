# Enhanced Agno API

A high-performance API for converting JSON data to Excel files using Agno's AI capabilities.

## Performance Optimizations

This API has been optimized for high performance using Agno's latest capabilities:

### Agent Optimizations

- **Agent Pooling**: Reuses initialized agents to avoid recreation costs
- **SQLite Storage**: Uses Agno's SQLite storage for persistent sessions
- **Exponential Backoff**: Automatically retries failed model calls with backoff
- **LRU Caching**: Caches agent creation to minimize redundant instantiations

### JSON Processing Improvements

- **Faster JSON Parsing**: Replaced standard JSON parser with orjson for better performance
- **Smarter Error Handling**: Better handling of malformed JSON data
- **Streaming Parser**: Support for parsing large JSON datasets efficiently

### Async Operation

- **Async APIs**: All operations support async/await for better concurrency
- **Non-blocking I/O**: File operations and model calls run in separate threads
- **Background Tasks**: Long-running operations executed in the background

### Memory Management

- **Resource Cleanup**: Proper cleanup of agent resources on shutdown
- **Optimized Memory Usage**: Takes advantage of Agno's lightweight agent design (~3.75 KiB per agent)

## Recent Fixes

We've made the following fixes to address issues with the codebase:

1. **Correct Storage Import**: Fixed the import path for storage modules from `agno.storage.memory` to `agno.storage.sqlite`
2. **Added Missing Dependencies**: Installed required dependencies:
   - `googlesearch-python` for web search capabilities
   - `pycountry` for GoogleSearchTools functionality
   - `orjson` for improved JSON parsing performance
3. **Session Persistence**: Implemented proper SQLite-based storage for agent sessions

## Performance Comparison

Based on Agno documentation and benchmarks:

| Metric | Agno (Optimized) | LangGraph | Factor |
|--------|-----------------|-----------|--------|
| Agent Instantiation | ~2μs | ~20ms | ~10,000x faster |
| Memory Per Agent | ~3.75 KiB | ~137 KiB | ~50x lighter |

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the server:

```bash
uvicorn app.main:app --reload
```

API endpoints:

- `POST /process` - Process JSON data to Excel
- `GET /download/{file_id}` - Download generated Excel file
- `GET /metrics` - Get system metrics

## Configuration

Environment variables:

- `GOOGLE_API_KEY` - Required for Gemini model access
- `LOG_LEVEL` - Logging level (default: INFO)

## Dependencies

- fastapi
- uvicorn
- pandas
- agno (>= 0.22.0)
- orjson (>= 3.9.0)
- aiofiles
- asyncio
- googlesearch-python
- pycountry 