# Agno AI Framework - Complete Documentation

## Table of Contents

1. [Framework Overview](#framework-overview)
2. [Agent Creation and Configuration](#agent-creation-and-configuration)
3. [Tools and Integrations](#tools-and-integrations)
4. [Workflows and Teams](#workflows-and-teams)
5. [Storage and Memory Systems](#storage-and-memory-systems)
6. [Monitoring, Debugging, and Production](#monitoring-debugging-and-production)
7. [Best Practices and Security](#best-practices-and-security)
8. [Complete Implementation Examples](#complete-implementation-examples)

---

## Framework Overview

### What is Agno?

**Agno** (previously known as Phidata) is an open-source, full-stack framework for building Multi-Agent Systems with memory, knowledge, and reasoning capabilities. It is designed as a lightweight, model-agnostic framework specifically optimized for building high-performance agentic applications with multimodal support.

### Core Purpose
Agno enables developers to build, ship, and monitor AI agents that can handle text, images, audio, and video inputs natively. The framework emphasizes performance, simplicity, and production readiness while providing a clean, composable, and Pythonic approach to agent development.

### Performance Characteristics
- **Ultra-fast instantiation**: Agent creation in ~2μs per agent (~10,000x faster than LangGraph)
- **Memory efficient**: Agents use ~3.75 KiB memory on average (~50x less than LangGraph)
- **Blazing fast execution**: Optimized for minimal execution time and parallelized tool calls
- **Async-first design**: Built for non-blocking operations and concurrent processing

### Key Features and Capabilities

#### Core Capabilities
1. **Model Agnostic**: Unified interface to 23+ model providers (OpenAI, Claude, Gemini, etc.)
2. **Natively Multi-Modal**: Handles text, image, audio, and video inputs/outputs without plugins
3. **Built-in Reasoning**: Three approaches - Reasoning Models, ReasoningTools, or custom chain-of-thought
4. **Advanced Multi-Agent Architecture**: Agent Teams with reasoning, memory, and shared context
5. **Structured Outputs**: Fully-typed responses using Pydantic models or JSON mode

### Architecture Overview

#### Five Levels of Agentic Systems
Agno supports building systems across five progressive levels:

1. **Level 1**: Agents with tools and instructions
2. **Level 2**: Agents with knowledge and storage
3. **Level 3**: Agents with memory and reasoning
4. **Level 4**: Agent Teams that can reason and collaborate
5. **Level 5**: Agentic Workflows with state and determinism

### Installation and Setup

#### System Requirements
- **Python**: 3.8+
- **Platform**: Cross-platform (tested on macOS, Linux, Windows)
- **Dependencies**: Minimal core dependencies with optional extras

#### Installation Methods

**Basic Installation:**
```bash
pip install agno
```

**With Common Dependencies:**
```bash
pip install openai duckduckgo-search yfinance lancedb tantivy pypdf requests exa-py newspaper4k lxml_html_clean sqlalchemy agno
```

**For streaming JSON processing:**
```bash
pip install ijson>=3.2.0
```

#### Environment Setup
```bash
export OPENAI_API_KEY=sk-xxxx
export GOOGLE_API_KEY=your_key
export AGNO_API_KEY=your_agno_api_key  # For monitoring
export AGNO_MONITOR=true
export AGNO_DEBUG=true
```

---

## Agent Creation and Configuration

### Basic Agent Structure

```python
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.python import PythonTools
from agno.storage.sqlite import SqliteStorage

agent = Agent(
    model=Gemini(id="gemini-2.0-flash-001", api_key="your_api_key"),
    tools=[PythonTools()],
    instructions=["Your agent instructions here"],
    storage=SqliteStorage(db_file="agent.db"),
    # Additional configuration parameters...
)
```

### Model Configuration

#### Google Gemini Models

```python
from agno.models.google import Gemini

# Basic configuration
model = Gemini(
    id="gemini-2.0-flash-001",  # Recommended for general use
    api_key="your_google_api_key",
    temperature=0.7,
    max_output_tokens=8192,
)

# Vertex AI configuration
model = Gemini(
    id="gemini-2.0-flash",
    vertexai=True,
    project_id="your-project-id",
    location="us-central1",
    grounding=True,  # Enable grounding capabilities
    search=True      # Enable search capabilities
)
```

**Recommended Gemini Models:**
- `gemini-2.0-flash`: Good for most use cases
- `gemini-2.0-flash-lite`: Most cost-effective
- `gemini-2.5-pro-exp-03-25`: Strongest multi-modal model

#### OpenAI Models

```python
from agno.models.openai import OpenAIChat

model = OpenAIChat(
    id="gpt-4o",  # Best for general use
    api_key="your_openai_api_key",
    temperature=0.7,
    max_tokens=4096,
    frequency_penalty=0.0,
    presence_penalty=0.0
)
```

**Recommended OpenAI Models:**
- `gpt-4o`: Best for general use cases
- `gpt-4o-mini`: Good for smaller tasks
- `o1`: Excellent for complex reasoning
- `o3-mini`: Strong reasoning with tool-calling

#### Claude Models

```python
from agno.models.anthropic import Claude

model = Claude(
    id="claude-sonnet-4-20250514",
    api_key="your_anthropic_api_key",
    temperature=0.7,
    max_tokens=4096
)
```

### Agent Configuration Parameters

#### Core Parameters

```python
agent = Agent(
    # Model configuration
    model=Gemini(id="gemini-2.0-flash-001"),
    
    # Tools configuration
    tools=[PythonTools(), GoogleSearchTools()],
    
    # Instructions and behavior
    instructions=["Be helpful and accurate"],
    expected_output="Structured response format",
    
    # Memory and storage
    storage=SqliteStorage(db_file="agent.db"),
    memory=Memory(db=SqliteMemoryDb()),
    
    # Session management
    session_id="unique_session_id",
    user_id="user_identifier",
    
    # Execution control
    reasoning=True,          # Enable reasoning capabilities
    show_tool_calls=True,    # Show tool execution details
    markdown=True,           # Format output as markdown
    stream=False,            # Enable streaming responses
    
    # History management
    add_history_to_messages=True,
    num_history_runs=3,
    
    # Tool execution limits
    tool_call_limit=20,
    
    # Retry and reliability
    exponential_backoff=True,
    retries=5,
    
    # Debug and monitoring
    debug_mode=False,
    monitoring=False,
    
    # DateTime context
    add_datetime_to_instructions=True
)
```

### Production Agent Configuration

```python
def create_production_agent(
    model_id: str = "gemini-2.0-flash-001",
    api_key: str = None,
    temp_dir: str = None,
    enable_memory: bool = True,
    enable_monitoring: bool = False
) -> Agent:
    """Create a production-ready Agno agent with full configuration."""
    
    # Setup directories
    temp_dir = temp_dir or str(Path.cwd() / "temp")
    storage_dir = Path("storage")
    storage_dir.mkdir(exist_ok=True)
    
    # Unique identifiers
    session_id = str(uuid.uuid4())
    db_path = storage_dir / f"agent_{session_id}.db"
    
    # Configure model
    model = Gemini(
        id=model_id,
        api_key=api_key or os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,
        max_output_tokens=8192,
    )
    
    # Configure tools
    tools = [
        PythonTools(
            run_code=True,
            pip_install=True,
            save_and_run=True,
            read_files=True,
            base_dir=Path(temp_dir).absolute(),
        ),
        GoogleSearchTools(
            fixed_max_results=5,
            timeout=10,
            fixed_language="en",
        )
    ]
    
    # Configure storage
    storage = SqliteStorage(
        table_name="agent_sessions",
        db_file=str(db_path),
        auto_upgrade_schema=True
    )
    
    # Configure memory (optional)
    memory = None
    if enable_memory:
        memory = Memory(
            model=model,
            db=SqliteMemoryDb(
                table_name="user_memories",
                db_file=str(db_path)
            ),
            delete_memories=False,
            clear_memories=False,
        )
    
    # Set monitoring environment
    if enable_monitoring and os.getenv("AGNO_API_KEY"):
        os.environ["AGNO_MONITOR"] = "true"
    
    # Create agent
    agent = Agent(
        model=model,
        tools=tools,
        storage=storage,
        memory=memory,
        
        instructions=[
            "You are a helpful AI assistant.",
            "Always provide accurate and well-reasoned responses.",
            "Use tools when necessary to gather information or perform tasks.",
            "Format your responses clearly and professionally."
        ],
        
        # Execution configuration
        session_id=session_id,
        reasoning=True,
        show_tool_calls=True,
        markdown=True,
        stream=False,
        
        # History and context
        add_history_to_messages=True,
        num_history_runs=5,
        add_datetime_to_instructions=True,
        
        # Reliability
        exponential_backoff=True,
        retries=3,
        tool_call_limit=15,
        
        # Memory configuration
        enable_agentic_memory=enable_memory,
        
        # Debug settings
        debug_mode=os.getenv("AGNO_DEBUG", "false").lower() == "true",
        monitoring=enable_monitoring,
    )
    
    logging.info(f"Created Agno agent with model {model_id}, session {session_id}")
    return agent
```

---

## Tools and Integrations

### Core Tool Categories

#### 1. Programming & System Tools

**PythonTools**
```python
from agno.tools.python import PythonTools

PythonTools(
    run_code=True,          # Execute Python code directly
    pip_install=True,       # Install packages using pip
    save_and_run=True,      # Save code to file and execute
    read_files=True,        # Read file contents
    list_files=True,        # List directory contents
    run_files=True,         # Execute Python files
    base_dir=Path("workspace"),  # Working directory
    safe_globals=None,      # Restricted global scope
    safe_locals=None        # Restricted local scope
)
```

#### 2. Search & Web Tools

**GoogleSearchTools**
```python
from agno.tools.googlesearch import GoogleSearchTools

GoogleSearchTools(
    fixed_max_results=5,                    # Maximum search results
    timeout=10,                             # Request timeout
    fixed_language="en",                    # Search language
    headers={"User-Agent": "Agent/1.0"},    # Custom headers
)
```

**DuckDuckGoTools**
```python
from agno.tools.duckduckgo import DuckDuckGoTools

agent = Agent(
    tools=[DuckDuckGoTools()],
    show_tool_calls=True
)
```

**TavilyTools & ExaTools**
```python
from agno.tools.tavily import TavilyTools
from agno.tools.exa import ExaTools

# Privacy-focused search
TavilyTools()

# Domain-specific search
ExaTools(
    include_domains=["cnbc.com", "reuters.com", "bloomberg.com"]
)
```

#### 3. Database Integration Tools

**SqliteStorage**
```python
from agno.storage.sqlite import SqliteStorage

storage = SqliteStorage(
    table_name="agent_sessions",
    db_file="storage/agents.db",
    auto_upgrade_schema=True
)
```

**PostgreSQL Tools**
```python
from agno.tools.postgresql import PostgresqlTools

PostgresqlTools(
    connection_string="postgresql://user:pass@host:port/db"
)
```

**DuckDbTools**
```python
from agno.tools.duckdb import DuckDbTools

agent = Agent(
    tools=[DuckDbTools()],
    show_tool_calls=True,
    system_prompt="Use this file for data: https://example.com/data.csv"
)
```

#### 4. Financial Data Tools

**YFinanceTools**
```python
from agno.tools.yfinance import YFinanceTools

YFinanceTools(
    stock_price=True,              # Current stock prices
    analyst_recommendations=True,   # Analyst ratings
    stock_fundamentals=True,       # Financial fundamentals
    company_info=True,             # Company information
    company_news=True              # Latest company news
)
```

#### 5. Communication Tools

**SlackTools**
```python
from agno.tools.slack import SlackTools

SlackTools(
    bot_token="xoxb-...",          # Slack Bot User OAuth Token
    signing_secret="..."           # App Signing Secret
)
```

**EmailTools**
```python
from agno.tools.email import EmailTools

EmailTools(
    smtp_server="smtp.gmail.com",
    smtp_port=587,
    username="user@gmail.com",
    password="app_password"
)
```

#### 6. Reasoning & Thinking Tools

**ReasoningTools**
```python
from agno.tools.reasoning import ReasoningTools

agent = Agent(
    tools=[ReasoningTools()],
    reasoning=True
)
```

### Custom Tool Creation

#### Basic Custom Tool Pattern
```python
from agno.tools import tool

@tool
def custom_calculator(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

agent = Agent(
    tools=[custom_calculator],
    show_tool_calls=True
)
```

#### Advanced Custom Tool Class
```python
from typing import List
from agno.tools import Toolkit

class CustomAPITool(Toolkit):
    def __init__(self, api_key: str, base_url: str):
        super().__init__(name="custom_api")
        self.api_key = api_key
        self.base_url = base_url
        
    @tool
    def api_request(self, endpoint: str, params: dict) -> dict:
        """Make API request to custom service."""
        # Implementation here
        pass
```

---

## Workflows and Teams

### Agno Workflows

Workflows in Agno are **deterministic, stateful, multi-agent programs** designed for production applications.

#### Core Workflow Architecture

```python
from agno.workflows import Workflow
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.postgres import PostgresStorage

class MyWorkflow(Workflow):
    # Define agents as attributes
    task_agent = Agent(
        model=OpenAIChat(),
        instructions=["Extract tasks from meeting notes"],
        structured_outputs=True
    )
    
    notification_agent = Agent(
        model=OpenAIChat(),
        tools=[SlackTools()],
        instructions=["Send notifications to team"]
    )
    
    def run(self, message: str) -> Iterator[RunResponse]:
        # Implement workflow logic here
        # Cache intermediate results
        if cached_result := self.session_state.get("cached_tasks"):
            return cached_result
            
        # Process with agents
        tasks = self.task_agent.run(message)
        
        # Store results
        self.session_state["cached_tasks"] = tasks
        
        # Send notifications
        self.notification_agent.run(f"New tasks: {tasks}")
        
        yield RunResponse(content=tasks)
```

### Agno Teams

Teams enable **collaborative multi-agent systems** where agents work together toward common goals.

#### Team Configuration

```python
from agno.teams import Team

# Create specialized agents
web_search_agent = Agent(
    name="Web Search Agent",
    model=OpenAIChat(),
    tools=[DuckDuckGoTools()],
    instructions=[
        "You are a web search specialist",
        "Find relevant information using web search",
        "Provide accurate, up-to-date information"
    ]
)

finance_agent = Agent(
    name="Finance Agent", 
    model=OpenAIChat(),
    tools=[YFinanceTools()],
    instructions=[
        "You are a financial data analyst",
        "Analyze market trends and financial metrics",
        "Provide data-driven insights"
    ]
)

# Create team
analysis_team = Team(
    name="Market Analysis Team",
    members=[web_search_agent, finance_agent],
    mode="collaborate",
    instructions=[
        "Collaborate to provide comprehensive market analysis",
        "Share findings and coordinate research",
        "Reach consensus on final recommendations"
    ],
    show_tool_calls=True,
    debug_mode=True
)
```

#### Team Collaboration Modes

1. **Route Mode**: Routes tasks to specific team members based on expertise
2. **Coordinate Mode** (Default): Coordinates between members with structured handoffs
3. **Collaborate Mode**: Enables direct collaboration and discussion

### Team Workflows

```python
class TeamWorkflow(Workflow):
    # Define team as workflow attribute
    research_team = Team(
        name="Research Team",
        members=[
            Agent(
                name="Reddit Researcher",
                model=OpenAIChat("gpt-4o"),
                tools=[ExaTools()],
                instructions=[
                    "You are a Reddit researcher",
                    "Find most relevant posts on Reddit",
                    "Provide detailed analysis of discussions"
                ]
            ),
            Agent(
                name="HackerNews Researcher", 
                model=OpenAIChat("gpt-4o"),
                tools=[HackerNewsTools()],
                instructions=[
                    "You are a HackerNews researcher",
                    "Find trending tech discussions",
                    "Analyze developer sentiment and trends"
                ]
            )
        ],
        mode="collaborate",
        instructions=[
            "Research the given topic thoroughly",
            "Share findings between researchers",
            "Collaborate on comprehensive analysis"
        ],
        show_tool_calls=True,
        debug_mode=True
    )
    
    def run(self, topic: str) -> Iterator[RunResponse]:
        # Step 1: Team research
        yield RunResponse(content=f"Starting research on: {topic}")
        
        research_results = self.research_team.run(
            f"Research the latest trends and discussions about: {topic}"
        )
        
        # Step 2: Cache research results
        self.session_state["research_data"] = research_results
        
        yield RunResponse(content=research_results)
```

---

## Storage and Memory Systems

### Storage Backend Options

#### 1. SQLite Storage (Development)

```python
from agno.storage.sqlite import SqliteStorage

storage = SqliteStorage(
    table_name="agent_sessions",
    db_file="tmp/agent_data.db",
    auto_upgrade_schema=True
)
```

#### 2. PostgreSQL Storage (Production)

```python
from agno.storage.postgres import PostgresStorage

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
storage = PostgresStorage(
    table_name="agent_sessions",
    db_url=db_url,
    schema="ai",
    auto_upgrade_schema=False
)
```

#### 3. MongoDB Storage

```python
from agno.storage.mongodb import MongoDbStorage

storage = MongoDbStorage(
    collection_name="agent_sessions",
    db_url="mongodb://localhost:27017",
    db_name="agno_ai"
)
```

### Memory Management

Agno provides three distinct types of memory:

#### 1. Session Storage (Short-term Memory)

```python
agent = Agent(
    storage=SqliteStorage(
        table_name="agent_sessions",
        db_file="sessions.db"
    ),
    add_history_to_messages=True,
    num_history_runs=5,  # Include last 5 interactions
    read_chat_history=True
)
```

#### 2. User Memories (Long-term Memory)

```python
from agno.memory.db.sqlite import SqliteMemoryDb

memory = SqliteMemoryDb(
    table_name="user_memories",
    db_file="memories.db"
)

agent = Agent(
    memory=memory,
    enable_agentic_memory=True,  # Allow agent to manage memories
    personalize_output=True
)
```

#### 3. Session Summaries

```python
agent = Agent(
    storage=storage,
    create_session_summary=True,
    summary_model=Gemini(id="gemini-1.5-flash-latest"),
    max_session_length=50  # Summarize after 50 messages
)
```

### Knowledge Base and Vector Storage

#### PgVector Configuration

```python
from agno.knowledge.pdf import PDFUrlKnowledgeBase
from agno.vectordb.pgvector import PgVector

vector_db = PgVector(
    collection="financial_docs",
    db_url="postgresql+psycopg://ai:ai@localhost:5532/ai"
)

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://example.com/financial_report.pdf"],
    vector_db=vector_db
)

agent = Agent(
    knowledge=knowledge_base,
    search_knowledge=True,  # Enable agentic RAG
    show_tool_calls=True
)
```

---

## Monitoring, Debugging, and Production

### Built-in Monitoring

**Enable Monitoring:**
```python
import os

# Set monitoring environment variables
os.environ["AGNO_MONITOR"] = "true"
os.environ["AGNO_DEBUG"] = "true"
os.environ["AGNO_API_KEY"] = "your_agno_api_key"

agent = Agent(
    model=model,
    monitoring=True,
    debug_mode=True,
    show_tool_calls=True
)
```

**Real-time monitoring at:**
- [app.agno.com](https://app.agno.com)
- [app.agno.com/sessions](https://app.agno.com/sessions)

### Debugging Capabilities

**Debug Mode Configuration:**
```python
agent = Agent(
    model=Claude(id="claude-sonnet-4-20250514"),
    tools=[YFinanceTools(stock_price=True)],
    debug_mode=True,           # Enable debug logging
    show_tool_calls=True,      # Show tool execution details
    reasoning=True,            # Enable reasoning traces
    stream=True               # Enable streaming for real-time debugging
)
```

### Production Logging

```python
import logging
import traceback

def setup_production_logging():
    """Configure comprehensive logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("/var/log/agno/agent.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_production_logging()
```

### Error Handling Patterns

```python
def convert_with_agno_with_error_handling(
    json_data: str,
    file_name: str,
    model: str,
    temp_dir: str,
    max_retries: int = 3
) -> tuple[str, str]:
    """Enhanced error handling for Agno operations."""
    
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            agent = create_agno_agent(model, temp_dir)
            
            # Validate inputs
            if not json_data or not json_data.strip():
                raise ValueError("Empty JSON data provided")
            
            response = agent.run(json_data)
            
            if not hasattr(response, 'content') or not response.content:
                raise ValueError("Agent returned empty response")
            
            return response.content, "success"
            
        except Exception as e:
            retry_count += 1
            last_error = str(e)
            
            logger.error(
                f"Agno processing failed (attempt {retry_count}/{max_retries}): "
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            )
            
            # Exponential backoff for retries
            if retry_count < max_retries:
                delay = min(retry_count * 2, 10)
                time.sleep(delay)
    
    raise Exception(f"All retry attempts failed. Last error: {last_error}")
```

### Performance Optimization

#### Agent Pool Management

```python
from cachetools import LRUCache

# Agent pool configuration
AGENT_POOL: LRUCache[str, Agent] = LRUCache(maxsize=10)

def create_agno_agent_with_pooling(model: str, temp_dir: str) -> Agent:
    """Create agent with LRU caching for performance."""
    agent_key = f"agent_{model}"
    
    # Check cache first
    agent = AGENT_POOL.get(agent_key)
    
    if agent:
        # Reconfigure agent for reuse
        for tool in agent.tools:
            if hasattr(tool, 'base_dir'):
                tool.base_dir = Path(temp_dir).absolute()
        
        # Fresh storage for isolation
        unique_db_id = uuid.uuid4().hex[:8]
        agent.storage = SqliteStorage(
            table_name="agent_sessions",
            db_file=f"storage/agents_{unique_db_id}.db",
            auto_upgrade_schema=True
        )
        
        return agent
    
    # Create new agent if not cached
    agent = Agent(
        model=Gemini(id=model, api_key=api_key),
        # ... full configuration
    )
    
    AGENT_POOL[agent_key] = agent
    return agent
```

### Production Deployment

#### Docker Configuration

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/storage /app/logs /app/temp

# Set environment variables
ENV PYTHONPATH=/app
ENV AGNO_STORAGE_DIR=/app/storage
ENV AGNO_LOG_FILE=/app/logs/agno.log

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Environment Variables

```bash
# Core Configuration
AGNO_MONITOR=true
AGNO_DEBUG=false
AGNO_API_KEY=your_agno_api_key

# Model Configuration
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/agno
AGNO_STORAGE_DIR=/app/storage

# Performance Configuration
MAX_POOL_SIZE=10
STORAGE_CLEANUP_HOURS=24
MIN_DISK_SPACE_MB=100

# Logging Configuration
LOG_LEVEL=INFO
AGNO_LOG_FILE=/app/logs/agno.log
```

---

## Best Practices and Security

### Security Best Practices

#### API Key Management

```python
import os
from typing import Optional

class SecureConfig:
    def __init__(self):
        self.google_api_key = self._get_required_env("GOOGLE_API_KEY")
        self.agno_api_key = self._get_optional_env("AGNO_API_KEY")
        self.database_url = self._get_required_env("DATABASE_URL")
    
    def _get_required_env(self, key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} not set")
        return value
    
    def _get_optional_env(self, key: str) -> Optional[str]:
        return os.getenv(key)
    
    def validate_api_keys(self):
        """Validate API keys format."""
        if not self.google_api_key.startswith('AIza'):
            raise ValueError("Invalid Google API key format")
        
        if self.agno_api_key and not self.agno_api_key.startswith('agno-'):
            raise ValueError("Invalid Agno API key format")
```

#### Input Validation and Sanitization

```python
import re
from pathlib import Path

def validate_and_sanitize_inputs(json_data: str, file_name: str, temp_dir: str) -> tuple[str, str, str]:
    """Validate and sanitize inputs to prevent security issues."""
    
    # Validate JSON data
    if not json_data or not json_data.strip():
        raise ValueError("Empty JSON data provided")
    
    # Limit JSON size to prevent DoS
    max_json_size = 10 * 1024 * 1024  # 10MB
    if len(json_data.encode('utf-8')) > max_json_size:
        raise ValueError(f"JSON data too large: {len(json_data)} bytes")
    
    # Sanitize file name
    safe_filename = re.sub(r'[^\w\s-]', '', file_name.strip())
    safe_filename = re.sub(r'[-\s]+', '-', safe_filename)
    
    if not safe_filename:
        safe_filename = "default_file"
    
    # Validate temp directory
    temp_path = Path(temp_dir)
    if not temp_path.exists():
        raise ValueError(f"Temporary directory does not exist: {temp_dir}")
    
    return json_data, safe_filename, str(temp_path)
```

### Resource Management

#### Agent Pool Cleanup

```python
def cleanup_agent_pool():
    """Remove agents from the pool to free up memory with proper resource cleanup."""
    global AGENT_POOL
    
    logger.info(f"Cleaning up agent pool with {len(AGENT_POOL)} agents")
    
    # Clean up each agent's resources before clearing the pool
    for agent_key, agent in list(AGENT_POOL.items()):
        try:
            if hasattr(agent, 'storage') and agent.storage:
                agent.storage.close()
        except Exception as cleanup_error:
            logger.warning(f"Error cleaning up agent {agent_key} storage: {cleanup_error}")
    
    AGENT_POOL.clear()
    logger.info("Agent pool cleanup completed")
```

#### Storage Cleanup

```python
async def cleanup_storage_files_async(max_age_hours: int = 24):
    """Async version: Clean up old storage files."""
    import time
    import glob
    
    storage_dir = Path("storage")
    cutoff_time = time.time() - (max_age_hours * 3600)
    
    # glob.glob is blocking, so run in thread
    matching_files = await asyncio.to_thread(glob.glob, str(storage_dir / "agents_*.db"))
    
    for db_file_path_str in matching_files:
        db_file_path = Path(db_file_path_str)
        try:
            file_stat = await asyncio.to_thread(db_file_path.stat)
            if file_stat.st_mtime < cutoff_time:
                await asyncio.to_thread(db_file_path.unlink)
                logger.debug(f"Cleaned up old storage file: {db_file_path}")
        except OSError as e:
            logger.warning(f"Failed to remove old storage file {db_file_path}: {e}")
```

---

## Complete Implementation Examples

### 1. Production Financial Analysis Agent

```python
import os
import uuid
import logging
import asyncio
from pathlib import Path
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.python import PythonTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.yfinance import YFinanceTools
from agno.storage.sqlite import SqliteStorage
from agno.memory.db.sqlite import SqliteMemoryDb

class FinancialAnalysisAgent:
    def __init__(self, config):
        self.config = config
        self.setup_logging()
        self.agent = self.create_agent()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def create_agent(self) -> Agent:
        """Create financial analysis agent with full configuration."""
        
        # Setup storage
        storage_dir = Path("storage")
        storage_dir.mkdir(exist_ok=True)
        
        session_id = str(uuid.uuid4())
        db_path = storage_dir / f"financial_agent_{session_id}.db"
        
        # Configure storage
        storage = SqliteStorage(
            table_name="financial_sessions",
            db_file=str(db_path),
            auto_upgrade_schema=True
        )
        
        # Configure memory
        memory = SqliteMemoryDb(
            table_name="financial_memories",
            db_file=str(db_path)
        )
        
        # Create agent
        agent = Agent(
            model=Gemini(
                id="gemini-2.0-flash-001",
                api_key=os.getenv("GOOGLE_API_KEY")
            ),
            tools=[
                YFinanceTools(
                    stock_price=True,
                    analyst_recommendations=True,
                    stock_fundamentals=True,
                    company_news=True
                ),
                GoogleSearchTools(
                    fixed_max_results=5,
                    timeout=10,
                    fixed_language="en"
                ),
                PythonTools(
                    run_code=True,
                    pip_install=True,
                    read_files=True,
                    base_dir=Path("temp").absolute()
                )
            ],
            storage=storage,
            memory=memory,
            
            instructions=[
                "You are a professional financial analyst.",
                "Use YFinance tools to gather current market data.",
                "Search for recent news and analysis using Google Search.",
                "Perform calculations and create visualizations using Python.",
                "Provide comprehensive, data-driven financial analysis.",
                "Always cite your sources and show your calculations."
            ],
            
            session_id=session_id,
            add_history_to_messages=True,
            num_history_runs=5,
            
            enable_agentic_memory=True,
            personalize_output=True,
            
            exponential_backoff=True,
            retries=3,
            show_tool_calls=True,
            reasoning=True,
            markdown=True,
            
            monitoring=os.getenv("AGNO_MONITOR", "false").lower() == "true",
            debug_mode=os.getenv("AGNO_DEBUG", "false").lower() == "true"
        )
        
        self.logger.info(f"Created financial analysis agent with session {session_id}")
        return agent
    
    async def analyze_stock(self, symbol: str, analysis_type: str = "comprehensive") -> str:
        """Perform stock analysis with error handling and monitoring."""
        
        start_time = time.time()
        
        try:
            prompt = f"""
            Perform a {analysis_type} analysis of {symbol} stock. Include:
            
            1. Current stock price and recent performance
            2. Key financial metrics and ratios
            3. Recent news and analyst recommendations
            4. Technical analysis and charts (if applicable)
            5. Risk assessment and investment recommendation
            
            Use all available tools to gather comprehensive data.
            Present findings in a professional report format.
            """
            
            self.logger.info(f"Starting {analysis_type} analysis for {symbol}")
            
            response = self.agent.run(prompt)
            
            execution_time = time.time() - start_time
            self.logger.info(f"Analysis completed in {execution_time:.2f}s")
            
            return response.content
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Analysis failed after {execution_time:.2f}s: {e}")
            raise

# Usage
config = {
    "model": "gemini-2.0-flash-001",
    "enable_monitoring": True
}

financial_agent = FinancialAnalysisAgent(config)
analysis = await financial_agent.analyze_stock("AAPL", "comprehensive")
print(analysis)
```

### 2. Multi-Agent Team Workflow

```python
from agno.workflows import Workflow
from agno.teams import Team
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.storage.postgres import PostgresStorage

class MarketResearchWorkflow(Workflow):
    """Multi-agent workflow for comprehensive market research."""
    
    # Research team
    research_team = Team(
        name="Market Research Team",
        members=[
            Agent(
                name="Web Researcher",
                model=OpenAIChat("gpt-4o"),
                tools=[DuckDuckGoTools()],
                instructions=[
                    "Research market trends and industry news",
                    "Find relevant articles and reports",
                    "Provide comprehensive market intelligence"
                ]
            ),
            Agent(
                name="Financial Analyst",
                model=OpenAIChat("gpt-4o"),
                tools=[YFinanceTools(stock_price=True, company_news=True)],
                instructions=[
                    "Analyze financial data and market metrics",
                    "Evaluate company performance and trends",
                    "Provide quantitative analysis"
                ]
            )
        ],
        mode="collaborate",
        instructions=[
            "Work together to provide comprehensive market analysis",
            "Share findings and coordinate research efforts",
            "Ensure all aspects of the market are covered"
        ]
    )
    
    # Report writer
    report_writer = Agent(
        name="Report Writer",
        model=OpenAIChat("gpt-4o"),
        instructions=[
            "Synthesize research findings into comprehensive report",
            "Create executive summary and detailed analysis",
            "Format report professionally with clear structure"
        ]
    )
    
    def run(self, research_topic: str, companies: list = None) -> Iterator[RunResponse]:
        """Execute market research workflow."""
        
        # Step 1: Initial research
        yield RunResponse(content=f"Starting market research on: {research_topic}")
        
        research_query = f"Research market trends, opportunities, and challenges for: {research_topic}"
        if companies:
            research_query += f" Focus on these companies: {', '.join(companies)}"
        
        # Team research phase
        research_results = self.research_team.run(research_query)
        
        # Store intermediate results
        self.session_state["research_data"] = research_results
        self.session_state["research_topic"] = research_topic
        self.session_state["companies"] = companies
        
        yield RunResponse(content="Research phase completed. Generating comprehensive report...")
        
        # Step 2: Report generation
        report_prompt = f"""
        Create a comprehensive market research report based on the following research:
        
        Topic: {research_topic}
        Companies: {companies if companies else 'Industry-wide analysis'}
        
        Research Findings:
        {research_results}
        
        Structure the report with:
        1. Executive Summary
        2. Market Overview
        3. Key Trends and Opportunities
        4. Company Analysis (if applicable)
        5. Risk Assessment
        6. Recommendations
        7. Conclusion
        
        Make it professional and actionable.
        """
        
        final_report = self.report_writer.run(report_prompt)
        
        # Store final results
        self.session_state["final_report"] = final_report
        
        yield RunResponse(content=final_report)

# Usage
workflow = MarketResearchWorkflow(
    storage=PostgresStorage(
        table_name="market_research_workflows",
        db_url="postgresql://localhost/agno"
    )
)

# Run workflow
for response in workflow.run(
    research_topic="Electric Vehicle Market",
    companies=["TSLA", "RIVN", "LCID"]
):
    print(response.content)
```

### 3. Streaming JSON Processing Example

```python
import asyncio
import json
from pathlib import Path
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.python import PythonTools

async def process_large_json_streaming(json_data: str, chunk_size: int) -> list:
    """Process large JSON data using streaming to avoid memory issues."""
    
    # Estimate if we need streaming based on data size (>10MB)
    data_size_mb = len(json_data.encode('utf-8')) / (1024 * 1024)
    if data_size_mb < 10:
        # For smaller data, use regular JSON parsing
        return await asyncio.to_thread(json.loads, json_data)
    
    logger.info(f"Large JSON detected ({data_size_mb:.1f}MB), using streaming parser")
    
    try:
        # Try to use ijson for streaming if available
        import ijson
        from io import StringIO
        
        json_stream = StringIO(json_data)
        
        # Try to parse as array first
        try:
            parser = await asyncio.to_thread(lambda: list(ijson.items(json_stream, 'item')))
            return parser
        except (ijson.JSONError, ValueError):
            # If array parsing fails, try parsing as single object
            json_stream.seek(0)
            obj = await asyncio.to_thread(json.loads, json_stream.read())
            return [obj] if obj else []
            
    except ImportError:
        logger.warning("ijson not available, using chunked processing fallback")
        # Fallback to regular parsing
        return await asyncio.to_thread(json.loads, json_data)

class StreamingJSONProcessor:
    def __init__(self):
        self.agent = Agent(
            model=Gemini(id="gemini-2.0-flash-001", api_key=os.getenv("GOOGLE_API_KEY")),
            tools=[
                PythonTools(
                    run_code=True,
                    pip_install=True,
                    save_and_run=True,
                    read_files=True,
                    base_dir=Path("temp").absolute()
                )
            ],
            instructions=[
                "You are an expert data processor.",
                "Process JSON data efficiently and create Excel reports.",
                "Handle large datasets with memory optimization.",
                "Use Python tools for data manipulation and file creation."
            ],
            exponential_backoff=True,
            retries=3,
            show_tool_calls=True
        )
    
    async def process_json_to_excel(self, json_data: str, output_filename: str) -> str:
        """Process JSON data and create Excel file with streaming optimization."""
        
        try:
            # Use streaming processing for large JSON
            processed_data = await process_large_json_streaming(json_data, chunk_size=1000)
            
            prompt = f"""
            Create an Excel report from this JSON data:
            
            Data (first 500 chars): {str(processed_data)[:500]}...
            
            Requirements:
            1. Create an Excel file named '{output_filename}'
            2. If data is a list, create separate sheets for different data types
            3. Include summary statistics where applicable
            4. Format headers and apply basic styling
            5. Handle currency data appropriately
            6. Save the file and return the absolute path
            
            Total records to process: {len(processed_data) if isinstance(processed_data, list) else 1}
            """
            
            response = self.agent.run(prompt)
            return response.content
            
        except Exception as e:
            logger.error(f"JSON processing failed: {e}")
            raise

# Usage
processor = StreamingJSONProcessor()
result = await processor.process_json_to_excel(large_json_data, "financial_report.xlsx")
print(result)
```

This comprehensive documentation covers all aspects of the Agno AI framework, from basic setup to advanced production deployments. The framework's performance advantages, model-agnostic design, and comprehensive tooling make it an excellent choice for building sophisticated AI agent systems.

---

## Official Resources

- **Documentation**: https://docs.agno.com
- **GitHub Repository**: https://github.com/agno-agi/agno
- **PyPI Package**: https://pypi.org/project/agno/
- **Monitoring Platform**: https://agno.com
- **Examples**: https://docs.agno.com/examples/introduction
- **Community**: GitHub Discussions and Issues