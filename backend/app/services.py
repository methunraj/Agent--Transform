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
MAX_POOL_SIZE = settings.MAX_POOL_SIZE  # Reasonable limit for different model types


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

    # --- SIMPLIFIED AND FOCUSED INSTRUCTIONS (approx. 50 lines) ---
    instructions = [
        "You are an AI assistant. Your task is to generate a Python script to convert JSON data into an Excel file.",
        "Goal: Create a functional Excel report quickly. Focus on speed and core data representation over exhaustive detail or complex formatting.",
        "Complete the task within 10 minutes.",
        
        "**Core Requirements:**",
        "1. Script Name: `excel_generator.py`.",
        "2. Data Handling: Process all main fields from the input JSON. If data is nested, flatten it appropriately for tabular representation.",
        "3. Excel Output: Create a multi-sheet Excel file if the JSON structure suggests multiple tables (e.g., list of objects for different sheets, or a top-level dictionary where keys are sheet names). Otherwise, a single sheet is fine.",
        "4. Currency: If currency data is present, represent it as numbers. Basic currency formatting (e.g., '$#,##0.00') is a plus if easily done.",
        "5. Charts (Optional): If time permits and data is suitable, add 1-2 simple charts (bar or line) for key metrics. Do not spend excessive time on charts.",
        "6. Formatting (Basic): Use `openpyxl` for basic table headers (bold). Complex coloring or styling is secondary to speed.",
        
        "**Technical Guidelines:**",
        "- Use `openpyxl` for Excel generation. `pandas` can be used for data manipulation if it simplifies the process.",
        "- Ensure the script saves the Excel file in the current working directory (provided as `temp_dir`).",
        "- Script must be runnable and handle potential errors gracefully (e.g., missing keys in JSON).",
        "- Print the absolute file path of the generated Excel file upon successful completion.",
        
        "**Error Handling:**",
        "- If your script fails, review the error, correct the script, and retry.",
        "- Prioritize generating a usable Excel file quickly. If complex features cause issues, simplify them.",
        
        "**Tool Usage:**",
        "- `PythonTools`: For writing, saving, and running your `excel_generator.py` script.",
        "- `GoogleSearchTools`: Only if absolutely necessary (e.g., to quickly find a specific `openpyxl` usage pattern if stuck). Avoid for general research to save time.",
        
        "**Focus on Speed:**",
        "- Avoid overly complex logic or visual enhancements if they significantly slow down development or processing.",
        "- A simpler, correct Excel file delivered quickly is better than a complex one that is late or error-prone.",
        "- If the JSON is very large or deeply nested, focus on representing the top-level data and key nested structures.",
    ]
    
    # Simplified expected output focusing on the essentials
    expected_output = """
# Excel Generation Script Output

## Script Execution
- Status: (Success/Failure)
- Generated File Path: (Absolute path to .xlsx file, if successful)
- Error Message: (If any errors occurred)

## Excel File Summary
- Sheets Created: (List of sheet names)
- Brief description of content in each sheet.

## Notes
- Any simplifications made for speed.
- Any parts of the JSON data that were particularly complex or omitted for brevity.
    """.strip()
    # --- END OF SIMPLIFIED INSTRUCTIONS ---
    
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
        db_path.chmod(0o644) # Read/write for owner, read for group/others
    
    # Create the agent with optimized tools and storage
    agent_config = {
        "model": Gemini(id=model, api_key=settings.GOOGLE_API_KEY),
        "tools": [
            PythonTools(
                run_code=True, pip_install=True, save_and_run=True,
                read_files=True, list_files=True, run_files=True,
                base_dir=Path(temp_dir).absolute(),
                safe_globals=None, safe_locals=None,
            ),
            GoogleSearchTools(
                fixed_max_results=5, timeout=10, fixed_language="en",
                headers={"User-Agent": "IntelliExtract-Agent/1.0"},
            )
        ],
        "storage": agent_storage,
        "add_history_to_messages": True,
        "num_history_runs": 3,
        "reasoning": False, # Keep False for Gemini API compatibility
        "show_tool_calls": True, # Good for debugging
        "markdown": True,
        "add_datetime_to_instructions": True,
        "tool_call_limit": 20,
        "instructions": instructions,
        "expected_output": expected_output,
        "exponential_backoff": True, # Agno handles retries for model calls
        "retries": 5, # Number of retries for model calls by Agno
        "debug_mode": debug_enabled,
        "monitoring": monitoring_enabled,
    }

    # Add cancellation token if job_manager and job_id are present
    # This requires Agno Agent to support a cancellation_token or similar mechanism.
    # Assuming Agno Agent might have a `cancellation_callback` or `is_cancelled_func`
    # If not, this part needs to be adapted based on how Agno handles cancellations.
    # For now, this is a conceptual addition.
    # if job_manager and job_id:
    #     job = await job_manager.get_job(job_id) # This needs to be async call if job_manager is async
    #     if job and job.cancellation_token:
    #         # This is hypothetical. Agno might need a specific way to integrate this.
    #         # Option 1: Pass a callback
    #         # agent_config["is_cancelled_callback"] = job.cancellation_token.is_set
    #         # Option 2: If Agent supports an asyncio.Event directly (less likely for sync agent)
    #         # agent_config["cancellation_event"] = job.cancellation_token
    #         pass


    agent = Agent(**agent_config)

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
        # This should be thread-safe if AGENT_POOL is accessed by multiple threads from to_thread
        # Consider using a lock for AGENT_POOL modifications if concurrency issues arise.
        # For now, assuming GIL provides sufficient protection for this simple dict operation.
        try:
            oldest_key = next(iter(AGENT_POOL)) # This could be problematic if dict is modified during iteration
            del AGENT_POOL[oldest_key]
            logger.info(f"Removed oldest agent from pool: {oldest_key}")
        except StopIteration: # AGENT_POOL was empty
            pass
        except RuntimeError: # Dictionary changed size during iteration
            logger.warning("Agent pool changed size during cleanup, skipping removal this time.")
            pass

    AGENT_POOL[agent_key] = agent
    logger.info(f"Agent pool updated (size: {len(AGENT_POOL)})")
    return agent

async def direct_json_to_excel_async(json_data: str, file_name: str, chunk_size: int, temp_dir: str) -> Tuple[str, str, str]:
    """Async version of direct_json_to_excel for better performance."""
    # Run the synchronous function in a thread pool to make it non-blocking
    # This is suitable if direct_json_to_excel is CPU-bound or blocking I/O not easily made async.
    # If it's I/O bound and can be made async, an async version would be better.
    return await asyncio.to_thread(direct_json_to_excel, json_data, file_name, chunk_size, temp_dir)


def direct_json_to_excel(json_data: str, file_name: str, chunk_size: int, temp_dir: str, job_id: Optional[str] = None, job_manager: Optional[Any] = None) -> Tuple[str, str, str]:
    """
    Convert JSON data directly to Excel with automatic retry mechanism.
    Will retry up to 3 times with different approaches on each retry.
    Optionally updates job progress if job_id and job_manager are provided.
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
    # Run the synchronous function in a thread pool to make it non-blocking.
    # Similar to direct_json_to_excel_async, if convert_with_agno involves significant CPU-bound work
    # or blocking I/O that's hard to make async directly, this is a reasonable approach.
    # For Agno agent calls that might be I/O bound (network requests to an AI service),
    # an async version of the Agno client/agent interaction would be ideal if available.
    return await asyncio.to_thread(convert_with_agno, json_data, file_name, description, model, temp_dir, user_id, session_id, job_id, job_manager)

def convert_with_agno(
    json_data: str,
    file_name: str,
    description: str,
    model: str,
    temp_dir: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    job_id: Optional[str] = None,      # For job progress updates
    job_manager: Optional[Any] = None  # For job progress updates
) -> tuple[str, str]:
    """
    Convert JSON data to Excel using Agno AI with streamlined processing.
    Will retry up to 2 times for faster processing.
    Returns tuple of (response_content, actual_session_id) for session continuity.
    Optionally updates job progress if job_id and job_manager are provided.
    """
    import uuid
    from .job_manager import JobStatus # Local import for JobStatus if needed by job_manager methods

    # Get debug setting from configuration
    debug_enabled = settings.AGNO_DEBUG
    
    # Generate session IDs if not provided (allow continuity for error recovery)
    current_user_id = user_id or f"user_{uuid.uuid4().hex[:8]}"
    current_session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
    
    # Validate input prompt to prevent empty content
    if not json_data or not json_data.strip():
        # Update job status to FAILED if job_manager is available
        if job_manager and job_id:
            asyncio.run(job_manager.update_job_status(job_id, JobStatus.FAILED, current_step="Input validation failed", result={"error": "Empty JSON data provided"}))
        raise ValueError("Empty JSON data provided - cannot process")
    
    max_retries = 2  # Reduced for faster processing
    retry_count = 0
    last_error = None
    
    # Initial progress update
    if job_manager and job_id:
        asyncio.run(job_manager.update_job_status(job_id, JobStatus.PROCESSING, progress=0.05, current_step="Initializing Agno agent"))

    while retry_count < max_retries:
        try:
            # Check for cancellation before starting/retrying
            if job_manager and job_id:
                job = asyncio.run(job_manager.get_job(job_id)) # This is sync context, need to run async code
                if job and job.cancellation_token and job.cancellation_token.is_set():
                    logger.info(f"Job {job_id} cancelled before Agno processing attempt {retry_count + 1}.")
                    asyncio.run(job_manager.update_job_status(job_id, JobStatus.CANCELLED, current_step="Cancelled by user before processing"))
                    # This function is synchronous, so we can't directly return an async response.
                    # The job status is updated. The caller (worker) should handle this.
                    # We might return a specific value or raise a custom exception.
                    return "Job cancelled", current_session_id


            agent = create_agno_agent(model, temp_dir) # temp_dir should be specific to this job
            
            if job_manager and job_id:
                asyncio.run(job_manager.update_job_status(job_id, JobStatus.PROCESSING, progress=0.1, current_step=f"Agent created (attempt {retry_count + 1})"))

            json_preview = json_data[:200] + "..." if len(json_data) > 200 else json_data
            
            if not file_name or not file_name.strip(): raise ValueError("Empty file_name provided")
            if not temp_dir or not temp_dir.strip(): raise ValueError("Empty temp_dir provided")
            
            prompt = f"""
            Create an Excel report from this financial data. Follow your instructions exactly.
            Base filename: {file_name}, Data source: {description}
            REMEMBER: 
            1. Change directory to {temp_dir} FIRST. 2. Detect currency. 3. Use google_search for USD rates.
            4. Create columns for original currency AND USD. 5. Print absolute file path.
            6. Simple analysis with currency conversion. 7. Use openpyxl.
            WORKING DIRECTORY: {temp_dir}. All files in this directory, relative paths.
            GOOGLE SEARCH: google_search(query, max_results=5, language="en") -> JSON string. Ex: google_search("USD to EUR rate")
            JSON Data: {json_data}"""
            
            if not prompt or not prompt.strip(): raise ValueError("Generated prompt is empty")
            
            if debug_enabled: logger.info(f"AI PROMPT: Dir={temp_dir}, File={file_name}, Desc={description}, JSON Size={len(json_data)}, Preview={json_preview}")
            
            logger.info(f"Starting Agno agent processing (attempt {retry_count + 1}/{max_retries})...")
            if job_manager and job_id:
                asyncio.run(job_manager.update_job_status(job_id, JobStatus.PROCESSING, progress=0.2, current_step=f"Sending prompt to AI (attempt {retry_count + 1})"))

            response = agent.run(
                prompt, stream=False, show_tool_calls=True, markdown=True,
                user_id=current_user_id, session_id=current_session_id
            )
            
            if not hasattr(response, 'content') or not response.content:
                logger.warning(f"Agent returned empty response. Content: '{getattr(response, 'content', None)}'")
                raise ValueError("Agent returned empty response content")

            if job_manager and job_id:
                 progress_after_run = 0.8 # Assuming most work is done if run succeeds
                 asyncio.run(job_manager.update_job_status(job_id, JobStatus.PROCESSING, progress=progress_after_run, current_step="AI processing complete, finalizing"))
            
            logger.info(f"Agno agent completed processing successfully.")
            if debug_enabled:
                reason_steps = len(response.reasoning) if hasattr(response, 'reasoning') else 0
                tool_calls_count = len(response.tool_calls) if hasattr(response, 'tool_calls') else 0
                logger.info(f"AI RESPONSE: Reasoning Steps={reason_steps}, Tool Calls={tool_calls_count}, Response Length={len(response.content)}")
            
            return response.content, current_session_id
            
        except Exception as e:
            retry_count += 1
            last_error = str(e)
            error_details = traceback.format_exc()
            error_type = type(e).__name__
            
            logger.error(f"Agno AI processing failed (attempt {retry_count}/{max_retries}): {error_type}: {e}\n{error_details}")

            if job_manager and job_id:
                # Update progress with error for this attempt
                asyncio.run(job_manager.update_job_status(job_id, JobStatus.PROCESSING,
                                                          current_step=f"Attempt {retry_count} failed: {error_type}",
                                                          result={"error_attempt_{retry_count}": str(e)}))
            
            is_retryable = not ("authentication" in str(e).lower() or "unauthorized" in str(e).lower())
            if "400" in str(e) or "Bad Request" in str(e): # Potentially remove agent from pool
                agent_key = f"agent_{model}"
                if agent_key in AGENT_POOL: del AGENT_POOL[agent_key]; logger.info(f"Removed faulty agent {agent_key} from pool.")
            
            if retry_count >= max_retries or not is_retryable:
                logger.error(f"Final attempt failed or error not retryable. Error: {last_error}")
                if job_manager and job_id:
                    asyncio.run(job_manager.update_job_status(job_id, JobStatus.FAILED, current_step=f"Fatal error after {retry_count} attempts", result={"error": last_error, "details": error_details}))
                agent_key = f"agent_{model}";
                if agent_key in AGENT_POOL: del AGENT_POOL[agent_key]; logger.info(f"Removed agent {agent_key} from pool after final failure.")
                raise # Re-raise the last exception
            
            delay = min(retry_count * 2, 10)
            logger.info(f"Retrying Agno conversion in {delay}s (attempt {retry_count+1}/{max_retries})...")
            if job_manager and job_id: # Update before sleep
                 asyncio.run(job_manager.update_job_status(job_id, JobStatus.PROCESSING, current_step=f"Retrying in {delay}s after error: {error_type}"))
            time.sleep(delay)

def find_newest_file(directory: str, files_before: set, job_id: Optional[str] = None, job_manager: Optional[Any] = None) -> Optional[str]:
    """
    Find the newest Excel file in directory with improved reliability.
    Optionally updates job progress if job_id and job_manager are provided.
    """
    from .job_manager import JobStatus # Local import

    # Use a helper for job updates to avoid repeating asyncio.run - only if job_manager and job_id are present
    async def _update_job_if_present(progress_val: Optional[float] = None, current_step_val: Optional[str] = None, status_val: Optional[JobStatus] = None, result_val: Optional[dict]=None):
        if job_manager and job_id:
            # This service function might be called from a sync or async context.
            # If called from sync (like current to_thread), asyncio.run is okay.
            # If called from async, it should be `await job_manager.update_job_status(...)`
            # For now, assuming it's run in a thread, so asyncio.run is used.
            # This is a common challenge when mixing sync/async code.
    # A better long-term solution is to make the service functions fully async.
            try:
        # Simplified approach for now: always use asyncio.run.
        # This assumes this helper is always called from a synchronous context (e.g., inside a function run by to_thread).
        # If the surrounding function `find_newest_file` itself becomes async, this needs to change to `await`.
        current_job = asyncio.run(job_manager.get_job(job_id))
        if current_job:
             asyncio.run(job_manager.update_job_status(job_id,
                                status=status_val or current_job.status,
                                progress=progress_val if progress_val is not None else current_job.progress,
                                current_step=current_step_val if current_step_val is not None else current_job.current_step,
                                result=result_val))
            except Exception as e_update:
                logger.error(f"Error updating job {job_id} status in find_newest_file: {e_update}")

    asyncio.run(_update_job_if_present(current_step_val="Searching for output file", progress_val=0.85))

    patterns = [os.path.join(directory, "*.xlsx"), os.path.join(directory, "**", "*.xlsx")]
    max_attempts, attempt_delay = 10, 0.3
    
    files_after = set()
    new_files = set()

    for attempt in range(max_attempts):
        if attempt > 0: time.sleep(attempt_delay); attempt_delay *= 1.2
        
        current_files_after = set()
        for pattern in patterns: current_files_after.update(glob.glob(pattern, recursive=True))
        files_after.update(current_files_after)
        new_files = files_after - files_before
        
        if new_files: logger.info(f"File detection ok attempt {attempt+1}. New: {new_files}"); break
        logger.info(f"File detection attempt {attempt+1}/{max_attempts} - no new files in {directory}")
        asyncio.run(_update_job_if_present(progress_val=0.85 + (0.1 * (attempt + 1) / max_attempts)))

    if settings.AGNO_DEBUG:
        logger.info(f"FILE DEBUG: Dir={directory}, Exists={os.path.exists(directory)}, Patterns={patterns}")
        logger.info(f"Files Before ({len(files_before)}): {list(files_before)}")
        logger.info(f"Files After ({len(files_after)}): {list(files_after)}")
        all_files_in_dir = [os.path.join(r,f) for r,ds,fs in os.walk(directory) for f in fs] if os.path.exists(directory) else []
        logger.info(f"ALL files in dir tree ({len(all_files_in_dir)}): {all_files_in_dir}")
        logger.info(f"New Files ({len(new_files)}): {list(new_files)}")

    if not new_files:
        logger.warning(f"No new Excel files in {directory} after {max_attempts} attempts.")
        asyncio.run(_update_job_if_present(status_val=JobStatus.FAILED, current_step_val="Output file not found", result_val={"error": "No output file generated."}))
        return None
    
    time.sleep(0.2)
    
    try:
        def get_file_sort_key(f):
            try: return (os.path.getmtime(f), os.path.getctime(f))
            except OSError: return (0,0)

        sorted_new_files = sorted(list(new_files), key=get_file_sort_key, reverse=True)

        for potential_file in sorted_new_files:
            try:
                if os.path.exists(potential_file) and os.path.getsize(potential_file) > 0:
                    logger.info(f"Newest file: {potential_file} (size: {os.path.getsize(potential_file)}B)")
                    asyncio.run(_update_job_if_present(current_step_val="Output file found", progress_val=0.98))
                    return potential_file
                else: logger.warning(f"Candidate {potential_file} zero size or gone. Checking next.")
            except OSError as e_size: logger.warning(f"Error accessing candidate {potential_file} for size: {e_size}")

        logger.warning(f"All {len(new_files)} new file(s) were empty or inaccessible.")
        asyncio.run(_update_job_if_present(status_val=JobStatus.FAILED, current_step_val="Output file empty/inaccessible", result_val={"error": "Generated file empty."}))
        return None

    except Exception as e:
        logger.error(f"Error selecting/accessing newest file: {e}")
        asyncio.run(_update_job_if_present(status_val=JobStatus.FAILED, current_step_val="Error accessing output file", result_val={"error": str(e)}))
        return None

# Cleanup function to manage the agent pool and storage
def cleanup_agent_pool():
    """Remove agents from the pool to free up memory."""
    global AGENT_POOL
    # Add lock if AGENT_POOL operations become complex or concurrent access is an issue
    logger.info(f"Cleaning up agent pool with {len(AGENT_POOL)} agents")
    AGENT_POOL.clear()

def cleanup_storage_files(max_age_hours: int = 1):
    """Clean up old storage files (e.g., .db files for Agno) to prevent disk space issues."""
    import glob
    import time
    storage_dir = Path(getattr(settings, 'AGNO_STORAGE_DIR', 'storage')) # Use configurable path
    storage_dir.mkdir(exist_ok=True) # Ensure it exists

    cutoff_time = time.time() - (max_age_hours * 3600)
    # Example pattern, adjust if Agno uses a different naming scheme or location
    file_pattern = str(storage_dir / "agents_*.db")

    cleaned_count = 0
    failed_count = 0

    for db_file_path_str in glob.glob(file_pattern):
        db_file_path = Path(db_file_path_str)
        try:
            if db_file_path.stat().st_mtime < cutoff_time:
                db_file_path.unlink() # Remove file
                logger.debug(f"Cleaned up old storage file: {db_file_path}")
                cleaned_count +=1
        except OSError as e:
            logger.warning(f"Failed to remove old storage file {db_file_path}: {e}")
            failed_count +=1

    if cleaned_count > 0 or failed_count > 0:
        logger.info(f"Storage cleanup: {cleaned_count} files removed, {failed_count} failures for pattern '{file_pattern}'.")

def json_serialize_job_result(result: Any) -> Any:
    """Helper to ensure job results are JSON serializable for SSE/WebSocket."""
    if result is None:
        return None
    if isinstance(result, (str, int, float, bool, list, dict)):
        # Attempt to serialize complex dicts/lists, but handle failures
        try:
            json.dumps(result) # Test serialization
            return result
        except (TypeError, OverflowError):
            return f"Non-serializable data of type {type(result).__name__}"
    return str(result) # Fallback for other types

