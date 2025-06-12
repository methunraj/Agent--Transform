# IntelliExtract Agno AI Backend

A production-ready API for converting JSON data to Excel files using Agno's AI capabilities with advanced monitoring and session management.

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

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

3. **Start the server:**
```bash
python start_server.py
# OR manually:
uvicorn app.main:app --reload
```

4. **Test the API:**
```bash
python run_tests.py smoke  # Quick test
python run_tests.py full   # Comprehensive test suite
```

API endpoints:

- `POST /process` - Process JSON data to Excel
- `GET /download/{file_id}` - Download generated Excel file
- `GET /metrics` - Get system metrics

## Configuration

Copy `.env.example` to `.env` and configure:

### Required Variables
- `GOOGLE_API_KEY` - Required for Gemini model access

### Optional Monitoring Variables
- `AGNO_API_KEY` - For Agno platform monitoring (get from https://app.agno.com/settings)
- `AGNO_MONITOR` - Set to `true` to enable monitoring at app.agno.com/sessions
- `AGNO_DEBUG` - Set to `true` to enable detailed debugging logs
- `DEVELOPMENT_MODE` - Set to `true` for development features (search caching)
- `LOG_LEVEL` - Logging level (default: INFO)

## Testing

The backend includes a comprehensive test suite to validate all functionality:

### Test Types

1. **Smoke Test** (Quick validation):
```bash
python run_tests.py smoke
```

2. **Comprehensive Test Suite** (Full validation):
```bash
python run_tests.py full
```

3. **Server Health Check**:
```bash
python run_tests.py check
```

### Test Coverage

The test suite includes:

- **Valid JSON Structures**: Simple dictionaries, arrays, complex nested data, multi-currency data, time series, large datasets, Unicode characters
- **Invalid JSON Cases**: Malformed JSON, syntax errors, empty data, mixed valid/invalid content
- **Edge Cases**: Very large numbers, null values, deeply nested structures, mixed data types
- **Processing Modes**: Tests for `auto`, `ai_only`, and `direct_only` modes
- **Error Handling**: Network timeouts, API failures, invalid parameters

### Manual Testing

You can also test individual endpoints manually:

```bash
# Test basic processing
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: application/json" \
  -d '{
    "json_data": "{\"company\": \"Test Corp\", \"revenue\": 1000000, \"currency\": \"USD\"}",
    "file_name": "test_file",
    "description": "Test data",
    "processing_mode": "direct_only"
  }'

# Get system metrics
curl "http://localhost:8000/metrics"
```

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