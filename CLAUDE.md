# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IntelliExtract - An AI-powered document extraction and processing system with two main components:
- **Backend**: FastAPI service using Agno AI framework for JSON to Excel conversion
- **Frontend**: Next.js web interface with Google Genkit integration (note: directory is misspelled as "frontent")

## Common Development Commands

### Backend Commands
```bash
# Navigate to backend
cd backend/

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload

# Run with custom settings
GOOGLE_API_KEY=your_key LOG_LEVEL=DEBUG uvicorn app.main:app --reload
```

### Frontend Commands
```bash
# Navigate to frontend (note the typo in directory name)
cd frontent/

# Install dependencies
npm install

# Development
npm run dev           # Next.js on port 9002
npm run genkit:dev    # Genkit AI development server
npm run genkit:watch  # Genkit with file watching

# Code quality
npm run lint          # ESLint
npm run typecheck     # TypeScript type checking

# Production
npm run build         # Build for production
npm run start         # Start production server
```

## Architecture Overview

### Backend Architecture
- **FastAPI + Agno AI**: High-performance API using agent pooling, SQLite storage, and async operations
- **Key optimizations**: 
  - Agent pooling with LRU caching to reuse Agno agents
  - Async file operations and background tasks
  - orjson for fast JSON parsing
  - Exponential backoff for resilient API calls
- **Main endpoints**:
  - `POST /process` - Convert JSON to Excel with AI processing modes
  - `GET /download/{file_id}` - Download generated Excel files
  - `GET /metrics` - System performance metrics

### Frontend Architecture
- **Next.js App Router**: Modern React framework with server components
- **State Management**: React Context API with providers for:
  - Configuration, Files, Jobs, LLM settings, Prompts, and Schemas
- **AI Integration**: Google Genkit framework with Gemini models
- **UI Components**: Radix UI primitives with Tailwind CSS styling
- **Key Features**:
  - Multi-step extraction workflow
  - LLM configuration management
  - Schema definition UI
  - File upload and management
  - Real-time processing status

### API Communication
- Frontend calls `/api/agno-process` which proxies to backend
- 20-minute timeout for long-running AI operations
- Supports multiple LLM providers with API key management

## Environment Configuration

### Backend (.env)
```
GOOGLE_API_KEY=your_gemini_api_key
LOG_LEVEL=INFO
```

### Frontend (.env.local)
```
GOOGLE_API_KEY=your_gemini_api_key
AGNO_BACKEND_URL=http://localhost:8001
```

## Important Implementation Notes

1. **Google Gen AI Development**: Frontend has comprehensive Claude.md in `/frontent/CLAUDE.md` with Google Gen AI SDK guidelines and `/api-context/` documentation

2. **Processing Modes**: Backend supports three modes:
   - `ai`: Full AI processing with Agno
   - `direct`: Direct JSON to Excel conversion
   - `auto`: Automatic mode selection based on data

3. **Performance Considerations**:
   - Backend uses agent pooling for ~10,000x faster agent instantiation vs LangGraph
   - Frontend implements large file upload support (up to 4.5GB)
   - Both use async patterns for non-blocking operations

4. **Error Handling**:
   - Backend implements exponential backoff for model calls
   - Frontend validates data with Zod schemas
   - Both handle file upload limits and validation

5. **File Management**:
   - Backend uses temporary directory with automatic cleanup
   - Frontend supports batch file uploads with progress tracking

## Agno Agent Development Rules

### MANDATORY AGNO DEVELOPMENT PROCESS

#### RULE 1: Documentation-First Approach
**BEFORE writing any Agno code, you MUST:**
1. Search for relevant Agno documentation in the codebase
2. Look for MDX files, example code, and implementation patterns
3. Verify correct imports, parameters, and usage patterns
4. Reference existing implementations in the codebase (especially `backend/app/services.py`)

#### RULE 2: Import Verification Protocol
**For EVERY Agno import, verify:**
```python
# Core imports - ALWAYS check these exist in Agno docs
from agno.agent import Agent
from agno.models.google import Gemini  # or OpenAI, Claude, etc.
from agno.tools.python import PythonTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.storage.sqlite import SqliteStorage  # NOT memory storage
from agno.workflows import Workflow
from agno.teams import Team
```

#### RULE 3: Agent Creation Standards
**When creating ANY Agno agent:**
```python
# MANDATORY parameters to verify
agent = Agent(
    model=Gemini(id=model, api_key=api_key),  # Verify model provider import
    tools=[...],  # Verify each tool import exists
    instructions=[...],  # Clear, specific instructions
    exponential_backoff=True,  # For production resilience
    retries=5,  # Handle transient failures
    # Optional but important:
    storage=SqliteStorage(db_file="path"),  # For persistence
    reasoning=True,  # For complex tasks
    stream=True,  # For real-time responses
)
```

### AGNO AGENT TYPE CHECKLIST

#### 1. Basic Agents
**Documentation to check:** `/docs/agents/basic.mdx`
- Verify imports: Agent, model provider, tools
- Key parameters: model, tools, instructions, exponential_backoff, retries

#### 2. Research Agents
**Documentation to check:** `/docs/agents/research.mdx`
- Additional imports: ExaTools, DuckDuckGoTools, TavilyTools
- Key features: Web search, source citation, report generation

#### 3. Finance Agents
**Documentation to check:** `/docs/agents/finance.mdx`
- Additional imports: YFinanceTools
- Key features: Real-time market data, financial analysis, currency conversion

#### 4. Multi-Agent Teams
**Documentation to check:** `/docs/teams/`
- Import: `from agno.teams import Team`
- Team modes: "route", "coordinate", "collaborate"

#### 5. Workflows
**Documentation to check:** `/docs/workflows/`
- Import: `from agno.workflows import Workflow`
- Key concepts: Stateful execution, storage integration, session management

#### 6. Storage & Memory
**Documentation to check:** `/docs/memory/`, `/docs/storage/`
- Storage options: SqliteStorage, PostgresStorage, MongoDbStorage
- Memory types: Session memory, user memories, knowledge bases

### TOOL INTEGRATION RULES

Before using ANY tool:
1. **Search for tool documentation**: `/docs/tools/{toolname}.mdx`
2. **Verify tool import path**: `from agno.tools.{category} import {ToolName}`
3. **Check required parameters** in tool initialization
4. **Verify tool capabilities** match your use case

Common Tool Patterns:
```python
# Python execution
from agno.tools.python import PythonTools
tools = [PythonTools(
    run_code=True,
    pip_install=True,
    base_dir=Path("workspace"),
    save_and_run=True,
    read_files=True
)]

# Search tools
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.tavily import TavilyTools

# Database tools
from agno.tools.postgresql import PostgresqlTools
from agno.tools.duckdb import DuckDbTools

# Communication tools
from agno.tools.slack import SlackTools
from agno.tools.email import EmailTools
```

### ERROR HANDLING REQUIREMENTS

Always implement:
```python
try:
    agent = create_agno_agent(...)
    response = agent.run(prompt)
except Exception as e:
    logger.error(f"Agno processing failed: {e}")
    # Implement retry logic
    # Fall back to alternative approach
```

### PERFORMANCE OPTIMIZATION RULES

1. **Agent Pooling**: Reuse agents when possible (they're only ~3.75 KiB)
2. **Async Operations**: Use async versions for I/O operations
3. **Streaming**: Enable for real-time responses
4. **Caching**: Use SqliteStorage for persistence across sessions

### VALIDATION CHECKLIST

Before running ANY Agno code:
- [ ] Searched for relevant documentation?
- [ ] Verified all imports exist in Agno?
- [ ] Checked example implementations?
- [ ] Validated model provider setup?
- [ ] Confirmed tool availability?
- [ ] Added error handling?
- [ ] Implemented retry logic?
- [ ] Set appropriate timeout values?

### COMMON PITFALLS TO AVOID

1. **NEVER use `agno.storage.memory`** - Use SqliteStorage instead
2. **NEVER assume tool availability** - Always verify imports
3. **NEVER skip exponential_backoff** in production
4. **NEVER ignore file path issues** - Use absolute paths or base_dir
5. **NEVER forget to handle streaming responses** properly

### DEBUGGING PROTOCOL

When Agno code fails:
1. Check import paths against documentation
2. Verify API keys and credentials
3. Confirm tool parameters match documentation
4. Check file paths and permissions
5. Review agent instructions for clarity
6. Enable `show_tool_calls=True` for debugging
7. Check logs for specific error messages

### REMEMBER: ALWAYS SEARCH AND VERIFY BEFORE CODING!

This is not optional - it's mandatory for every Agno implementation. The framework is powerful but requires precise usage. When in doubt, search the documentation and check existing implementations.