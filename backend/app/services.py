# app/services.py
import os
import json
import uuid
import time
import glob
import logging
import pandas as pd
from pathlib import Path
import asyncio
import traceback
from functools import lru_cache
from typing import Dict, List, Tuple, Any, Optional

# Set Matplotlib backend to a non-GUI backend to avoid threading issues
import matplotlib
matplotlib.use('Agg')  # Must be before any other matplotlib imports

from agno.agent import Agent
from agno.models.google import Gemini
from agno.storage.sqlite import SqliteStorage
from agno.tools.python import PythonTools
from agno.tools.googlesearch import GoogleSearchTools
from .core.config import settings

logger = logging.getLogger(__name__)

# Set up Agno monitoring environment variables if configured
if settings.AGNO_API_KEY:
    os.environ["AGNO_API_KEY"] = settings.AGNO_API_KEY
if settings.AGNO_MONITOR:
    os.environ["AGNO_MONITOR"] = "true"
if settings.AGNO_DEBUG:
    os.environ["AGNO_DEBUG"] = "true"

# Agent pool for reusing initialized agents (lightweight in Agno ~3.75 KiB per agent)
AGENT_POOL: Dict[str, Agent] = {}
MAX_POOL_SIZE = 10  # Reasonable limit for different model types


def create_agno_agent(model: str, temp_dir: str) -> Agent:
    """Creates and configures the Agno agent with efficient pooling strategy.
    
    Agno agents are extremely lightweight (~3.75 KiB) and fast to instantiate (~2μs).
    Using model-based pooling for maximum reuse while maintaining thread safety.
    """
    # Clean up old storage files to prevent accumulation
    cleanup_storage_files()
    
    # Create agent key based on model only for cross-request reuse
    agent_key = f"agent_{model}"
    
    # Check if agent exists in pool for reuse across requests
    if agent_key in AGENT_POOL:
        agent = AGENT_POOL[agent_key]
        logger.info(f"Reusing cached agent for model: {model} (pool size: {len(AGENT_POOL)})")
        
        # Update agent's working directory for this request
        for tool in agent.tools:
            if hasattr(tool, 'base_dir'):
                tool.base_dir = Path(temp_dir).absolute()
                logger.debug(f"Updated tool base_dir to: {temp_dir}")
        
        # CRITICAL: Assign fresh storage to prevent message history contamination
        # This prevents the 400 "contents.parts must not be empty" error
        import uuid
        storage_dir = Path("storage")
        storage_dir.mkdir(exist_ok=True)
        unique_db_id = uuid.uuid4().hex[:8]
        agent.storage = SqliteStorage(
            table_name="financial_agent_sessions",
            db_file=str(storage_dir / f"agents_{unique_db_id}.db"),
            auto_upgrade_schema=True
        )
        logger.info(f"Assigned fresh storage to cached agent: agents_{unique_db_id}.db")
        
        return agent
    
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set in the environment.")

    # --- OPTIMIZED INSTRUCTIONS (Following Agno Best Practices) ---
    instructions = [
        "You are a financial analyst. Create a multi-sheet Excel report from JSON data.",
        "Save and run a Python script named 'excel_report_generator.py'",
        "For currency conversion, use the google_search function to find current exchange rates (e.g., 'USD to EUR exchange rate')", 
        "The google_search function is available and returns results as JSON formatted string",
        "Create sheets: Summary, Income Statement, Balance Sheet, Cash Flow, Regional Performance",
        "Use relative paths only - all files in current working directory",
        "Keep code simple and reliable",
        "CRITICAL: If any code execution fails, you MUST:",
        "1. Read the error message carefully",
        "2. Identify the root cause (e.g. data structure issues, missing imports)",
        "3. Fix the code immediately", 
        "4. Save and run the corrected code",
        "5. Repeat until Excel file is successfully created",
        "REMEMBER: You have run_files=True so you can execute Python files directly",
        "The balance sheet equity section is a flat dictionary, not nested like assets/liabilities",
        "DO NOT GIVE UP - keep trying until the Excel file exists and can be accessed"
    ]
    
    # Expected output format for consistent markdown structure
    expected_output = """
# Financial Analysis Report

## Process Overview
- Data processing steps
- Currency conversion details
- File generation status

## Technical Details
- Script execution results
- File locations
- Any issues encountered

## Summary
- Final output file path
- Key metrics processed
    """.strip()
    # --- END OF OPTIMIZED INSTRUCTIONS ---
    
    # Environment and monitoring configuration
    development_mode = settings.DEVELOPMENT_MODE
    monitoring_enabled = settings.AGNO_MONITOR
    debug_enabled = settings.AGNO_DEBUG
    
    # Create unique storage per request to prevent message history contamination
    # This prevents the 400 "contents.parts must not be empty" error from agent pooling
    import uuid
    storage_dir = Path("storage")
    storage_dir.mkdir(exist_ok=True)
    
    # Use unique database per request to isolate message history
    unique_db_id = uuid.uuid4().hex[:8]
    db_path = storage_dir / f"agents_{unique_db_id}.db"
    
    # Ensure storage directory has write permissions
    storage_dir.chmod(0o755)
    
    agent_storage = SqliteStorage(
        table_name="financial_agent_sessions",
        db_file=str(db_path),
        auto_upgrade_schema=True
    )
    
    # Ensure database file has write permissions
    if db_path.exists():
        db_path.chmod(0o644)
    
    # Create the agent with optimized tools and storage
    # According to docs, Agno agents are very lightweight (~3.75 KiB) and fast to instantiate (~2μs)
    # 
    # Tool optimization strategy:
    # - PythonTools: Directory isolation, security settings, selective feature enabling
    # - GoogleSearchTools: Result limiting, timeout management, conditional caching
    # - Performance: Tool call limits, custom headers, language optimization
    #
    # Storage strategy:
    # - Session persistence for conversation continuity across requests
    # - Simple chat history for context (no complex user preference learning needed)
    
    agent = Agent(
        model=Gemini(id=model, api_key=settings.GOOGLE_API_KEY),
        tools=[
            PythonTools(
                # Core execution settings (following official docs)
                run_code=True, 
                pip_install=True, 
                save_and_run=True, 
                read_files=True,
                list_files=True,
                run_files=True,                       # CRITICAL: Enable file running (was disabled!)
                
                # Directory and security configuration
                base_dir=Path(temp_dir).absolute(),   # Enforce absolute path for directory isolation
                
                # Performance optimizations
                safe_globals=None,                    # Can be configured for additional security
                safe_locals=None,                     # Can be configured for additional security
            ),
            GoogleSearchTools(
                # Performance optimization settings (according to Agno docs)
                fixed_max_results=5,                  # Limit results to reduce processing time
                timeout=10,                           # Set reasonable timeout for network calls
                fixed_language="en",                  # Pre-set language to avoid detection overhead
                headers={"User-Agent": "IntelliExtract-Agent/1.0"},  # Custom user agent
            )
        ],
        storage=agent_storage,           # Re-enabled for session continuity and error recovery
        add_history_to_messages=True,    # Enable for learning from previous attempts
        num_history_runs=3,              # Allow some history for context
        reasoning=True,                  # Enable reasoning mode for complex financial analysis
        show_tool_calls=True,
        markdown=True,                   # Enable markdown formatting for structured output
        add_datetime_to_instructions=True,  # Include timestamp in markdown output
        
        # Tool performance optimization (following Agno documentation)
        tool_call_limit=20,              # Sufficient for search + code generation + error fixing
        # tool_choice="auto",            # Let model choose appropriate tools (default)
        instructions=instructions,
        expected_output=expected_output,  # Provide markdown structure template
        exponential_backoff=True,        # Auto-retry with backoff on model errors
        retries=5,                       # Number of retries for model calls
        debug_mode=debug_enabled,        # Enable based on AGNO_DEBUG environment variable
        monitoring=monitoring_enabled,   # Enable Agno.com monitoring based on AGNO_MONITOR environment variable
    )
    # Log monitoring configuration for transparency
    if monitoring_enabled:
        if settings.AGNO_API_KEY:
            logger.info(f"Agno monitoring ENABLED - sessions will be tracked at app.agno.com/sessions")
        else:
            logger.warning("Agno monitoring enabled but AGNO_API_KEY not set - monitoring may not work properly")
    else:
        logger.info("Agno monitoring DISABLED - set AGNO_MONITOR=true to enable")
    
    if debug_enabled:
        logger.info("Agno debug mode ENABLED - detailed tool calls and reasoning will be logged")
    
    logger.info(f"Created new Agno agent with model: {model}, monitoring={monitoring_enabled}, debug={debug_enabled}")
    
    # Store agent in pool for cross-request reuse with size management
    if len(AGENT_POOL) >= MAX_POOL_SIZE:
        # Remove oldest agent (simple FIFO strategy)
        oldest_key = next(iter(AGENT_POOL))
        del AGENT_POOL[oldest_key]
        logger.info(f"Removed oldest agent from pool: {oldest_key}")
    
    AGENT_POOL[agent_key] = agent
    logger.info(f"Agent pool updated (size: {len(AGENT_POOL)})")
    return agent

async def direct_json_to_excel_async(json_data: str, file_name: str, chunk_size: int, temp_dir: str) -> Tuple[str, str, str]:
    """Async version of direct_json_to_excel for better performance."""
    # Run the synchronous function in a thread pool to make it non-blocking
    return await asyncio.to_thread(direct_json_to_excel, json_data, file_name, chunk_size, temp_dir)

def direct_json_to_excel(json_data: str, file_name: str, chunk_size: int, temp_dir: str) -> Tuple[str, str, str]:
    """
    Convert JSON data directly to Excel with automatic retry mechanism.
    Will retry up to 3 times with different approaches on each retry.
    """
    max_retries = 3
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            decoder = json.JSONDecoder()
            pos = 0
            data_objects = []
            clean_json_data = json_data.strip()
            
            # Try different parsing approaches based on retry count
            if retry_count == 0:
                # First attempt: Standard parsing
                while pos < len(clean_json_data):
                    obj, end_pos = decoder.raw_decode(clean_json_data[pos:])
                    data_objects.append(obj)
                    pos = end_pos
                    while pos < len(clean_json_data) and clean_json_data[pos].isspace():
                        pos += 1
            elif retry_count == 1:
                # Second attempt: Line-by-line parsing
                for line in clean_json_data.split('\n'):
                    if line.strip():
                        try:
                            obj = json.loads(line.strip())
                            data_objects.append(obj)
                        except json.JSONDecodeError:
                            continue
            else:
                # Third attempt: Try with more lenient approach (wrap in array if needed)
                try:
                    data_objects = [json.loads(clean_json_data)]
                except json.JSONDecodeError:
                    try:
                        data_objects = [json.loads(f"[{clean_json_data}]")]
                    except:
                        # Last resort: Try to extract any valid JSON objects
                        import re
                        json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
                        matches = re.findall(json_pattern, clean_json_data)
                        for match in matches:
                            try:
                                obj = json.loads(match)
                                data_objects.append(obj)
                            except:
                                continue
            
            if not data_objects:
                raise ValueError("No valid JSON objects found in the input data")
            
            data = data_objects[0] if len(data_objects) == 1 else data_objects

            file_id = str(uuid.uuid4())
            safe_filename = "".join(c for c in file_name if c.isalnum() or c in (' ', '-', '_')).strip()
            xlsx_filename = f"{safe_filename}_direct.xlsx"
            file_path = os.path.join(temp_dir, f"{file_id}_{xlsx_filename}")

            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                sheets_created = False
                
                if isinstance(data, list):
                    if len(data) > 0:  # Only process if list is not empty
                        try:
                            if len(data) > chunk_size:
                                for i in range(0, len(data), chunk_size):
                                    chunk_data = data[i:i + chunk_size]
                                    if chunk_data:  # Ensure chunk is not empty
                                        try:
                                            df = pd.json_normalize(chunk_data)
                                            if not df.empty:  # Only create sheet if DataFrame has data
                                                df.to_excel(writer, sheet_name=f'Data_Chunk_{i//chunk_size + 1}', index=False)
                                                sheets_created = True
                                        except Exception as e:
                                            logger.warning(f"Failed to normalize chunk {i}: {e}")
                                            # Create a simple representation of the chunk
                                            df = pd.DataFrame([{"chunk_data": str(chunk_data)}])
                                            df.to_excel(writer, sheet_name=f'Chunk_{i//chunk_size + 1}_Raw', index=False)
                                            sheets_created = True
                            else:
                                try:
                                    df = pd.json_normalize(data)
                                    if not df.empty:  # Only create sheet if DataFrame has data
                                        df.to_excel(writer, sheet_name='Data', index=False)
                                        sheets_created = True
                                except Exception as e:
                                    logger.warning(f"Failed to normalize list data: {e}")
                                    # Create a simple representation of the data
                                    df = pd.DataFrame([{"raw_data": str(item)} for item in data])
                                    df.to_excel(writer, sheet_name='Raw_Data', index=False)
                                    sheets_created = True
                        except Exception as e:
                            logger.warning(f"Failed to process list data: {e}")
                            # Last resort: create basic representation
                            df = pd.DataFrame([{"list_item": str(item), "index": i} for i, item in enumerate(data)])
                            df.to_excel(writer, sheet_name='List_Items', index=False)
                            sheets_created = True
                
                elif isinstance(data, dict):
                    # Handle dictionary data
                    if data:  # Only process if dict is not empty
                        for key, value in data.items():
                            if isinstance(value, list) and value:  # Only process non-empty lists
                                df = pd.json_normalize(value)
                                if not df.empty:
                                    safe_sheet_name = str(key)[:31].replace('/', '_').replace('\\', '_')
                                    df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
                                    sheets_created = True
                        
                        # If no list values were found, create a sheet with the dict itself
                        if not sheets_created:
                            df = pd.json_normalize([data])
                            if not df.empty:
                                df.to_excel(writer, sheet_name='Data', index=False)
                                sheets_created = True
                
                else:
                    # Handle primitive data types
                    df = pd.DataFrame([{'value': data}])
                    df.to_excel(writer, sheet_name='Data', index=False)
                    sheets_created = True
                
                # Ensure at least one sheet exists
                if not sheets_created:
                    # Create a minimal sheet with error information
                    df = pd.DataFrame([{'error': 'No valid data found', 'original_data_type': str(type(data))}])
                    df.to_excel(writer, sheet_name='Error', index=False)

            return file_id, xlsx_filename, file_path
            
        except Exception as e:
            retry_count += 1
            last_error = str(e)
            error_details = traceback.format_exc()
            logger.error(f"Direct conversion failed (attempt {retry_count}/{max_retries}): {e}\n{error_details}")
            
            if retry_count >= max_retries:
                logger.error(f"All {max_retries} direct conversion attempts failed. Giving up.")
                raise
            
            # Wait briefly before retrying (with increasing delay)
            time.sleep(retry_count)
            logger.info(f"Retrying direct conversion (attempt {retry_count+1}/{max_retries})...")

async def convert_with_agno_async(json_data: str, file_name: str, description: str, model: str, temp_dir: str, user_id: str = None, session_id: str = None) -> tuple[str, str]:
    """Async version of convert_with_agno for better performance."""
    # Run the synchronous function in a thread pool to make it non-blocking
    return await asyncio.to_thread(convert_with_agno, json_data, file_name, description, model, temp_dir, user_id, session_id)

def convert_with_agno(json_data: str, file_name: str, description: str, model: str, temp_dir: str, user_id: str = None, session_id: str = None) -> tuple[str, str]:
    """
    Convert JSON data to Excel using Agno AI with streamlined processing.
    Will retry up to 2 times for faster processing.
    Returns tuple of (response_content, actual_session_id) for session continuity.
    """
    import uuid
    
    # Get debug setting from configuration
    debug_enabled = settings.AGNO_DEBUG
    
    # Generate session IDs if not provided (allow continuity for error recovery)
    if not user_id:
        user_id = f"user_{uuid.uuid4().hex[:8]}"
    if not session_id:
        session_id = f"session_{uuid.uuid4().hex[:8]}"
    
    # Validate input prompt to prevent empty content
    if not json_data or not json_data.strip():
        raise ValueError("Empty JSON data provided - cannot process")
    
    max_retries = 2  # Reduced for faster processing
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            agent = create_agno_agent(model, temp_dir)
            
            # Create a focused prompt with currency conversion requirements
            # Note: JSON data is truncated in logs for privacy/readability
            json_preview = json_data[:200] + "..." if len(json_data) > 200 else json_data
            
            # Validate prompt content before creating
            if not file_name or not file_name.strip():
                raise ValueError("Empty file_name provided")
            if not temp_dir or not temp_dir.strip():
                raise ValueError("Empty temp_dir provided")
            
            prompt = f"""
            Create an Excel report from this financial data. Follow your instructions exactly.
            
            Base filename: {file_name}
            Data source: {description}
            
            REMEMBER: 
            1. Change directory to {temp_dir} FIRST in your script
            2. Detect the currency in the financial data
            3. Use the google_search function to find current USD exchange rates
            4. Create columns for both original currency AND USD values
            5. Print the absolute file path when done
            6. Keep the analysis simple but include currency conversion
            7. Use openpyxl library for Excel creation
            
            **WORKING DIRECTORY:** {temp_dir}
            **IMPORTANT:** All files must be created in the working directory above. Use relative paths only.
            
            **GOOGLE SEARCH USAGE:** You have access to google_search(query, max_results=5, language="en") function
            Example: google_search("USD to EUR exchange rate today", max_results=3)
            
            JSON Data:
            {json_data}
            """
            
            # Final validation of the complete prompt
            if not prompt or not prompt.strip():
                raise ValueError("Generated prompt is empty")
            
            # Log clean debug info (without full JSON data)
            if debug_enabled:
                logger.info(f"🤖 AI PROMPT SUMMARY:")
                logger.info(f"   📁 Working Directory: {temp_dir}")
                logger.info(f"   📄 File Name: {file_name}")
                logger.info(f"   📝 Description: {description}")
                logger.info(f"   📊 JSON Data Size: {len(json_data)} characters")
                logger.info(f"   🔍 JSON Preview: {json_preview}")
                logger.info(f"   🎯 Task: Financial Excel report with currency conversion")
            
            # Using run method with enhanced streaming and real-time monitoring
            logger.info(f"Starting Agno agent processing (attempt {retry_count + 1}/{max_retries})...")
            
            try:
                # Use non-streaming mode to get a proper response object
                response = agent.run(
                    prompt,
                    stream=False,                    # Disable streaming to get response object with .content
                    show_tool_calls=True,            # Enable to see tool execution for debugging
                    markdown=True,                   # Ensure markdown formatting consistency
                    user_id=user_id,                 # Restore for session continuity and error recovery
                    session_id=session_id            # Restore for conversation continuity
                )
                
                # Validate response content to prevent empty responses
                if not hasattr(response, 'content') or not response.content:
                    logger.warning("Agent returned empty response - this could cause 400 errors")
                    if hasattr(response, 'content'):
                        logger.warning(f"Response content: '{response.content}'")
                    raise ValueError("Agent returned empty response content")
                
            except Exception as agent_error:
                error_msg = str(agent_error)
                logger.error(f"Agent.run() failed: {type(agent_error).__name__}: {agent_error}")
                
                # Special handling for the specific 400 error we're troubleshooting
                if "contents.parts must not be empty" in error_msg:
                    logger.error("❌ DETECTED TARGET ERROR: Empty content parts in Gemini request")
                    logger.error("This is likely caused by message history accumulation or empty tool responses")
                    
                    # Clear session history to prevent recurrence
                    try:
                        if hasattr(agent, 'storage') and agent.storage:
                            logger.info("Attempting to clear problematic session history...")
                            # Note: This is a temporary mitigation - real fix is in agent configuration
                    except Exception as clear_error:
                        logger.warning(f"Could not clear session history: {clear_error}")
                
                # Re-raise to be handled by outer retry logic with enhanced error categorization
                raise
            
            logger.info(f"Agno agent completed processing successfully")
            
            # Enhanced debug logging for AI response analysis
            if debug_enabled:
                logger.info(f"🧠 AI RESPONSE ANALYSIS:")
                
                # Log reasoning steps
                if hasattr(response, 'reasoning') and response.reasoning:
                    logger.info(f"   🤔 Reasoning Steps: {len(response.reasoning)} steps captured")
                    for i, step in enumerate(response.reasoning[:3]):  # Show first 3 steps
                        step_preview = str(step)[:150] + "..." if len(str(step)) > 150 else str(step)
                        logger.info(f"   💭 Step {i+1}: {step_preview}")
                
                # Log tool calls
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    logger.info(f"   🔧 Tool Calls: {len(response.tool_calls)} executed")
                    for i, tool_call in enumerate(response.tool_calls):
                        tool_name = getattr(tool_call, 'name', 'Unknown Tool')
                        logger.info(f"   🛠️  Tool {i+1}: {tool_name}")
                
                # Log response content summary (not full content)
                content_preview = response.content[:300] + "..." if len(response.content) > 300 else response.content
                logger.info(f"   📄 Response Length: {len(response.content)} characters")
                logger.info(f"   📋 Response Preview: {content_preview}")
            else:
                # Basic logging when debug is off
                if hasattr(response, 'reasoning') and response.reasoning:
                    logger.info(f"Agent reasoning steps captured: {len(response.reasoning)} steps")
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    logger.info(f"Tool calls executed: {len(response.tool_calls)} calls")
            
            # Return both response content and session_id for continuity
            logger.info(f"Session management: user_id={user_id}, session_id={session_id}")
            return response.content, session_id
            
        except Exception as e:
            retry_count += 1
            last_error = str(e)
            error_details = traceback.format_exc()
            error_type = type(e).__name__
            
            # Enhanced error logging with categorization
            logger.error(f"Agno AI processing failed (attempt {retry_count}/{max_retries})")
            logger.error(f"Error type: {error_type}")
            logger.error(f"Error message: {e}")
            logger.error(f"Full traceback:\n{error_details}")
            
            # Categorize errors for better handling
            is_retryable = True
            if "API" in str(e) or "rate limit" in str(e).lower():
                logger.warning("API-related error detected - likely handled by Agno's exponential backoff")
            elif "memory" in str(e).lower() or "timeout" in str(e).lower():
                logger.warning("Resource-related error detected - may benefit from retry")
            elif "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
                logger.error("Authentication error detected - retries likely won't help")
                is_retryable = False
            
            if retry_count >= max_retries:
                logger.error(f"All {max_retries} attempts failed. Final error type: {error_type}")
                logger.error(f"Final error message: {last_error}")
                raise
            
            if not is_retryable:
                logger.error("Error is not retryable - failing immediately")
                raise
            
            # Exponential backoff at application level (in addition to Agno's built-in backoff)
            delay = min(retry_count * 2, 10)  # Cap delay at 10 seconds
            logger.info(f"Retrying conversion in {delay} seconds (attempt {retry_count+1}/{max_retries})...")
            time.sleep(delay)

def find_newest_file(directory: str, files_before: set) -> Optional[str]:
    # Check both the main directory and subdirectories for Excel files
    patterns = [
        os.path.join(directory, "*.xlsx"),
        os.path.join(directory, "**", "*.xlsx"),  # Recursive search
    ]
    
    # Add a small delay to ensure file system operations are complete
    import time
    time.sleep(0.5)  # 500ms delay to ensure file is fully written
    
    files_after = set()
    for pattern in patterns:
        files_after.update(glob.glob(pattern, recursive=True))
    
    logger.info(f"=== FILE DETECTION DEBUG ===")
    logger.info(f"Directory being searched: {directory}")
    logger.info(f"Directory exists: {os.path.exists(directory)}")
    logger.info(f"Search patterns: {patterns}")
    logger.info(f"Files before: {len(files_before)} - {list(files_before)}")
    logger.info(f"Files after: {len(files_after)} - {list(files_after)}")
    
    # Also check for all files in the directory for debugging
    all_files = []
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                full_path = os.path.join(root, file)
                all_files.append(full_path)
        logger.info(f"ALL files in directory tree: {all_files}")
    except Exception as e:
        logger.warning(f"Could not walk directory {directory}: {e}")
    
    # ALSO check current working directory as fallback
    cwd = os.getcwd()
    logger.info(f"Current working directory: {cwd}")
    if cwd != directory:
        logger.info(f"Checking for Excel files in CWD as fallback...")
        cwd_xlsx_files = glob.glob(os.path.join(cwd, "*.xlsx"))
        logger.info(f"Excel files found in CWD: {cwd_xlsx_files}")
        
        # If no files in temp dir but files in CWD, log this issue
        if not files_after and cwd_xlsx_files:
            logger.error(f"ERROR: Agent created files in wrong location! Files in CWD: {cwd_xlsx_files}")
            logger.error(f"These should have been created in: {directory}")
    
    new_files = files_after - files_before
    logger.info(f"New files: {list(new_files)}")
    logger.info(f"=== END FILE DETECTION DEBUG ===")
    
    if not new_files:
        logger.warning(f"No new Excel files found in {directory}")
        return None
    
    newest_file = max(new_files, key=os.path.getmtime)
    logger.info(f"Newest file selected: {newest_file}")
    return newest_file

# Cleanup function to manage the agent pool and storage
def cleanup_agent_pool():
    """Remove agents from the pool to free up memory."""
    global AGENT_POOL
    logger.info(f"Cleaning up agent pool with {len(AGENT_POOL)} agents")
    AGENT_POOL.clear()

def cleanup_storage_files():
    """Clean up old storage files to prevent disk space issues."""
    import glob
    import time
    storage_dir = Path("storage")
    if storage_dir.exists():
        # Remove storage files older than 1 hour
        cutoff_time = time.time() - 3600  # 1 hour
        for db_file in glob.glob(str(storage_dir / "agents_*.db")):
            if os.path.getmtime(db_file) < cutoff_time:
                try:
                    os.remove(db_file)
                    logger.debug(f"Cleaned up old storage file: {db_file}")
                except OSError as e:
                    logger.warning(f"Failed to remove old storage file {db_file}: {e}")

