# Claude Code Rules for Agno Agent Development

## MANDATORY AGNO DEVELOPMENT PROCESS

### RULE 1: Documentation-First Approach
**BEFORE writing any Agno code, you MUST:**
1. Search for relevant Agno documentation in the codebase
2. Look for MDX files, example code, and implementation patterns
3. Verify correct imports, parameters, and usage patterns
4. Reference existing implementations in the codebase (especially `backend/app/services.py`)

### RULE 2: Import Verification Protocol
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

### RULE 3: Agent Creation Standards
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

## AGNO AGENT TYPE CHECKLIST

### 1. Basic Agents
**Documentation to check:** `/docs/agents/basic.mdx`
```python
# Verify imports
from agno.agent import Agent
from agno.models.{provider} import {Model}
from agno.tools.{toolname} import {Tool}

# Key parameters
- model: LLM instance (required)
- tools: List of tool instances
- instructions: String or list of strings
- exponential_backoff: Boolean (recommended True)
- retries: Integer (recommended 3-5)
```

### 2. Research Agents
**Documentation to check:** `/docs/agents/research.mdx`
```python
# Additional imports
from agno.tools.exa import ExaTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.tavily import TavilyTools

# Key features
- Web search capabilities
- Source citation
- Report generation
```

### 3. Finance Agents
**Documentation to check:** `/docs/agents/finance.mdx`
```python
# Additional imports
from agno.tools.yfinance import YFinanceTools

# Key features
- Real-time market data
- Financial analysis
- Currency conversion (via GoogleSearchTools)
```

### 4. Multi-Agent Teams
**Documentation to check:** `/docs/teams/`
```python
from agno.teams import Team

# Team modes to verify
- "route": Leader routes to appropriate member
- "coordinate": Leader delegates and synthesizes
- "collaborate": All members work together
```

### 5. Workflows
**Documentation to check:** `/docs/workflows/`
```python
from agno.workflows import Workflow

# Key concepts
- Stateful execution
- Storage integration
- Session management
```

### 6. Storage & Memory
**Documentation to check:** `/docs/memory/`, `/docs/storage/`
```python
# Storage options - verify availability
from agno.storage.sqlite import SqliteStorage
from agno.storage.postgres import PostgresStorage
from agno.storage.mongodb import MongoDbStorage

# Memory types
- Session memory
- User memories
- Knowledge bases
```

## TOOL INTEGRATION RULES

### Before using ANY tool:
1. **Search for tool documentation**: `/docs/tools/{toolname}.mdx`
2. **Verify tool import path**: `from agno.tools.{category} import {ToolName}`
3. **Check required parameters** in tool initialization
4. **Verify tool capabilities** match your use case

### Common Tool Patterns:
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

## ERROR HANDLING REQUIREMENTS

### Always implement:
```python
try:
    agent = create_agno_agent(...)
    response = agent.run(prompt)
except Exception as e:
    logger.error(f"Agno processing failed: {e}")
    # Implement retry logic
    # Fall back to alternative approach
```

## PERFORMANCE OPTIMIZATION RULES

1. **Agent Pooling**: Reuse agents when possible (they're only ~3.75 KiB)
2. **Async Operations**: Use async versions for I/O operations
3. **Streaming**: Enable for real-time responses
4. **Caching**: Use SqliteStorage for persistence across sessions

## VALIDATION CHECKLIST

Before running ANY Agno code:
- [ ] Searched for relevant documentation?
- [ ] Verified all imports exist in Agno?
- [ ] Checked example implementations?
- [ ] Validated model provider setup?
- [ ] Confirmed tool availability?
- [ ] Added error handling?
- [ ] Implemented retry logic?
- [ ] Set appropriate timeout values?

## COMMON PITFALLS TO AVOID

1. **NEVER use `agno.storage.memory`** - Use SqliteStorage instead
2. **NEVER assume tool availability** - Always verify imports
3. **NEVER skip exponential_backoff** in production
4. **NEVER ignore file path issues** - Use absolute paths or base_dir
5. **NEVER forget to handle streaming responses** properly

## DEBUGGING PROTOCOL

When Agno code fails:
1. Check import paths against documentation
2. Verify API keys and credentials
3. Confirm tool parameters match documentation
4. Check file paths and permissions
5. Review agent instructions for clarity
6. Enable `show_tool_calls=True` for debugging
7. Check logs for specific error messages

## EXAMPLE VERIFICATION WORKFLOW

```python
# Step 1: Search documentation
# Look for: /docs/agents/data-analysis.mdx

# Step 2: Verify imports
# Check: Can I import Agent, Gemini, PythonTools?

# Step 3: Check existing implementations
# Review: backend/app/services.py for patterns

# Step 4: Validate parameters
# Ensure: All required params are provided

# Step 5: Test with minimal example
# Start simple, add complexity gradually
```

## REMEMBER: ALWAYS SEARCH AND VERIFY BEFORE CODING!

This is not optional - it's mandatory for every Agno implementation. The framework is powerful but requires precise usage. When in doubt, search the documentation and check existing implementations.