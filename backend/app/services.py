# app/services.py
import os
import shutil # Added for disk usage check
import json
import uuid
import time
import glob
import logging
import tempfile # Added for fallback in disk space check
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
# --- Disk Space Check Configuration & Helper ---
MIN_FREE_SPACE_BYTES = settings.MIN_DISK_SPACE_MB * 1024 * 1024 if hasattr(settings, 'MIN_DISK_SPACE_MB') and settings.MIN_DISK_SPACE_MB > 0 else 50 * 1024 * 1024  # Default 50MB

def _has_sufficient_disk_space(path: str, required_bytes: int) -> bool:
    """Checks if the specified path has at least required_bytes of free disk space."""
    try:
        # Ensure the path (or its parent for a file path) exists for disk_usage
        check_path = path
        if not os.path.exists(check_path):
            check_path = os.path.dirname(check_path)
            if not os.path.exists(check_path): # If parent also doesn't exist, tempfile.gettempdir() as last resort
                check_path = Path(tempfile.gettempdir())
                logger.warning(f"Path {path} and its parent do not exist. Checking disk space for system temp {check_path}.")

        total, used, free = shutil.disk_usage(check_path)
        logger.debug(f"Disk usage for partition of {check_path}: Total={total // (1024*1024)}MB, Used={used // (1024*1024)}MB, Free={free // (1024*1024)}MB. Required free: {required_bytes // (1024*1024)}MB")
        if free < required_bytes:
            logger.warning(f"Insufficient disk space. Free: {free // (1024*1024)}MB, Required: {required_bytes // (1024*1024)}MB for path based on {check_path}")
            return False
        return True
    except FileNotFoundError:
        logger.error(f"Path not found for disk space check: {path}. Cannot determine disk space.")
        return False # Fail safe if path doesn't exist for check
    except Exception as e:
        logger.error(f"Could not check disk space for {path} (checking on {check_path if 'check_path' in locals() else path}): {e}", exc_info=True)
        return False # Fail safe: if we can't check, assume not enough space


# --- Agent Pool & Creation ---
if settings.AGNO_MONITOR:
    os.environ["AGNO_MONITOR"] = "true"
if settings.AGNO_DEBUG:
    os.environ["AGNO_DEBUG"] = "true"

from cachetools import LRUCache

# Agent pool using LRUCache
# Initialize AGENT_POOL as an LRUCache instance.
# The settings.MAX_POOL_SIZE will be used from the config.
AGENT_POOL: LRUCache[str, Agent] = LRUCache(maxsize=settings.MAX_POOL_SIZE if settings.MAX_POOL_SIZE > 0 else 10)

def create_agno_agent(model: str, temp_dir: str) -> Agent:
    """
    Creates and configures the Agno agent using an LRU cache for pooling.
    Ensures critical state (base_dir, storage) is reset for every use.
    """
    cleanup_storage_files() # Clean up old SQLite files
    
    agent_key = f"agent_{model}"
    
    # Attempt to retrieve agent from LRU cache
    # A lock might be considered if AGENT_POOL access becomes a bottleneck,
    # but cachetools itself is generally thread-safe for basic operations.
    # However, the agent object modification after retrieval should be handled carefully if accessed concurrently.
    # Given FastAPI's threading model for sync endpoints, this should be okay.
    agent: Optional[Agent] = AGENT_POOL.get(agent_key)

    if agent:
        logger.info(f"Reusing agent for model '{model}' from LRU cache. Cache size: {len(AGENT_POOL)}/{AGENT_POOL.maxsize}")
        # CRITICAL: Re-configure state for the retrieved agent
        # 1. Update tool base_dir
        for tool in agent.tools:
            if hasattr(tool, 'base_dir'):
                tool.base_dir = Path(temp_dir).absolute()
        logger.debug(f"Updated tool base_dir for cached agent '{model}' to: {temp_dir}")

        # 2. Assign fresh SqliteStorage
        storage_dir = Path(getattr(settings, 'AGNO_STORAGE_DIR', 'storage'))
        storage_dir.mkdir(exist_ok=True)
        unique_db_id = uuid.uuid4().hex[:8]
        db_path = storage_dir / f"agents_{model.replace('-', '_')}_{unique_db_id}.db"
        agent.storage = SqliteStorage(
            table_name="financial_agent_sessions", # Consider making table_name dynamic if needed
            db_file=str(db_path),
            auto_upgrade_schema=True
        )
        logger.info(f"Assigned fresh storage to cached agent '{model}': {db_path.name}")
        
        # Explicitly put it back if you want to update its "recently used" status,
        # though get() itself might do this depending on cachetools version/config.
        # For LRUCache, get() typically marks item as recently used.
        # AGENT_POOL[agent_key] = agent
        return agent
    
    # Agent not in cache or evicted, create a new one
    logger.info(f"Creating new agent for model '{model}'. Cache size: {len(AGENT_POOL)}/{AGENT_POOL.maxsize}")
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set in the environment for new agent creation.")

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
    
    # Store the newly created agent in the LRU cache.
    # The LRUCache will automatically handle eviction if maxsize is reached.
    AGENT_POOL[agent_key] = agent
    logger.info(f"Stored new agent for model '{model}' in LRU cache. Cache size: {len(AGENT_POOL)}/{AGENT_POOL.maxsize}")
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
    from .job_manager import JobStatus # Local import for type hinting if needed

    # Helper for job updates, similar to the one in find_newest_file
    async def _update_job_if_present(progress_val: Optional[float] = None, current_step_val: Optional[str] = None, status_val: Optional[JobStatus] = None, result_val: Optional[dict]=None):
        if job_manager and job_id:
            try:
                current_job = asyncio.run(job_manager.get_job(job_id)) # Assuming sync context
                if current_job:
                    await job_manager.update_job_status(job_id,
                                                        status=status_val or current_job.status,
                                                        progress=progress_val if progress_val is not None else current_job.progress,
                                                        current_step=current_step_val if current_step_val is not None else current_job.current_step,
                                                        result=result_val)
            except Exception as e_update:
                logger.error(f"Error updating job {job_id} status in direct_json_to_excel: {e_update}")

    max_retries = 3
    retry_count = 0
    last_error = None
    
    # Initial progress update
    asyncio.run(_update_job_if_present(progress_val=0.05, current_step_val="Starting direct JSON to Excel conversion"))

    while retry_count < max_retries:
        try:
            decoder = json.JSONDecoder()
            pos = 0
            data_objects = []
            clean_json_data = json_data.strip()
            
            # Parsing attempts (simplified for brevity, original logic retained)
            if retry_count == 0: # Standard parsing
                while pos < len(clean_json_data): obj, end_pos = decoder.raw_decode(clean_json_data[pos:]); data_objects.append(obj); pos = end_pos; pos += len(clean_json_data[pos:]) - len(clean_json_data[pos:].lstrip())
            elif retry_count == 1: # Line-by-line
                for line in clean_json_data.split('\n'):
                    if line.strip():
                        try: data_objects.append(json.loads(line.strip()))
                        except json.JSONDecodeError: continue
            else: # Lenient array wrapping / regex extraction
                try: data_objects = [json.loads(clean_json_data)]
                except json.JSONDecodeError:
                    try: data_objects = [json.loads(f"[{clean_json_data}]")]
                    except:
                        import re; matches = re.findall(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}', clean_json_data)
                        for match in matches:
                            try: data_objects.append(json.loads(match))
                            except: continue

            if not data_objects: raise ValueError("No valid JSON objects found")

            asyncio.run(_update_job_if_present(progress_val=0.15, current_step_val="JSON parsing complete"))
            
            data = data_objects[0] if len(data_objects) == 1 else data_objects

            file_id = str(uuid.uuid4())
            # Sanitize file_name to prevent path traversal
            base_name = os.path.basename(file_name)
            safe_filename = "".join(c for c in base_name if c.isalnum() or c in (' ', '-', '_')).strip()
            # Ensure safe_filename is not empty after sanitization, provide a default if it is.
            if not safe_filename:
                safe_filename = "default_excel_file"

            xlsx_filename = f"{safe_filename}_direct_chunked.xlsx" if isinstance(data, list) and len(data) > chunk_size else f"{safe_filename}_direct.xlsx"
            file_path = os.path.join(temp_dir, f"{file_id}_{xlsx_filename}")

            # Further security: Ensure the temp_dir is a legitimate, controlled directory
            # and that the resolved file_path is within this directory.
            # This is implicitly handled by os.path.join if temp_dir is absolute and controlled,
            # but an explicit check can be added if needed:
            # if not Path(file_path).resolve().is_relative_to(Path(temp_dir).resolve()):
            #     raise SecurityException("Resolved file path is outside of the designated temporary directory.")

            # Check for sufficient disk space before attempting to write the file
            if not _has_sufficient_disk_space(temp_dir, MIN_FREE_SPACE_BYTES):
                error_msg = f"Insufficient disk space in {temp_dir}. Requires at least {MIN_FREE_SPACE_BYTES // (1024*1024)}MB free."
                logger.error(error_msg)
                # Update job status if applicable
                asyncio.run(_update_job_if_present(status_val=JobStatus.FAILED, current_step_val="Disk space check failed", result_val={"error": error_msg}))
                raise IOError(error_msg) # Raise an IOError or a custom exception

            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                sheets_created = False
                
                if isinstance(data, list) and data: # Data is a non-empty list
                    num_records = len(data)
                    # Determine if chunking is needed based on record count and chunk_size parameter
                    # The problem states "Detect files > 50MB" - this function gets json_data string.
                    # We can estimate based on string length or record count.
                    # Let's use record count > chunk_size as the primary trigger for list processing.
                    # A more sophisticated size check could be `len(json_data.encode('utf-8')) > SOME_BYTE_THRESHOLD`

                    # Effective chunk_size from request, default to a large number if not chunking by count
                    actual_chunk_size = chunk_size if chunk_size > 0 else num_records + 1 # ensure no chunking if chunk_size is 0 or invalid

                    if num_records > actual_chunk_size:
                        logger.info(f"Large JSON list detected ({num_records} records > chunk_size {actual_chunk_size}). Processing in chunks.")
                        asyncio.run(_update_job_if_present(current_step_val=f"Processing {num_records} records in chunks of {actual_chunk_size}"))

                        num_chunks = (num_records + actual_chunk_size - 1) // actual_chunk_size
                        for i in range(num_chunks):
                            chunk_data = data[i * actual_chunk_size : (i + 1) * actual_chunk_size]
                            chunk_name = f'Data_Chunk_{i+1}'
                            if chunk_data:
                                try:
                                    df = pd.json_normalize(chunk_data)
                                    if not df.empty:
                                        df.to_excel(writer, sheet_name=chunk_name, index=False)
                                        sheets_created = True
                                        logger.debug(f"Processed chunk {i+1}/{num_chunks} to sheet {chunk_name}")
                                except Exception as e_chunk:
                                    logger.warning(f"Failed to normalize chunk {i+1}: {e_chunk}. Writing raw.")
                                    df_raw = pd.DataFrame([{"raw_chunk_data": str(item)} for item in chunk_data])
                                    df_raw.to_excel(writer, sheet_name=f'{chunk_name}_Raw', index=False)
                                    sheets_created = True # Still a sheet created

                            current_progress = 0.15 + (0.80 * ((i + 1) / num_chunks)) # 0.15 base, 0.80 for chunk processing
                            asyncio.run(_update_job_if_present(progress_val=current_progress, current_step_val=f"Processed chunk {i+1}/{num_chunks}"))
                    else: # Not chunking by record count, process as a single list
                        try:
                            df = pd.json_normalize(data)
                            if not df.empty: df.to_excel(writer, sheet_name='Data', index=False); sheets_created = True
                        except Exception as e_list:
                            logger.warning(f"Failed to normalize full list data: {e_list}. Writing raw.")
                            df_raw = pd.DataFrame([{"raw_list_data": str(item)} for item in data])
                            df_raw.to_excel(writer, sheet_name='Raw_Data', index=False); sheets_created = True

                elif isinstance(data, dict) and data: # Data is a non-empty dictionary
                    asyncio.run(_update_job_if_present(current_step_val="Processing dictionary data"))
                    for key, value in data.items():
                        if isinstance(value, list) and value:
                            try:
                                df = pd.json_normalize(value)
                                if not df.empty:
                                    safe_sheet_name = str(key)[:30].replace('/', '_').replace('\\', '_') # Max sheet name length is 31
                                    df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
                                    sheets_created = True
                            except Exception as e_dict_list:
                                logger.warning(f"Failed to normalize list for dict key '{key}': {e_dict_list}. Writing raw.")
                                df_raw = pd.DataFrame([{"raw_item_for_{key}": str(item)} for item in value])
                                df_raw.to_excel(writer, sheet_name=f'{str(key)[:25]}_Raw', index=False); sheets_created = True
                    if not sheets_created: # If dict had no lists or they were empty, process dict itself
                        try:
                            df = pd.json_normalize([data]) # Normalize requires list of dicts
                            if not df.empty: df.to_excel(writer, sheet_name='Data', index=False); sheets_created = True
                        except Exception as e_dict_flat:
                             logger.warning(f"Failed to normalize flat dictionary: {e_dict_flat}. Writing raw.")
                             df_raw = pd.DataFrame([{"key":k, "value":str(v)} for k,v in data.items()])
                             df_raw.to_excel(writer, sheet_name='Raw_Dict_Data', index=False); sheets_created = True
                elif data: # Primitive data type or other non-list/non-dict but non-empty
                     df = pd.DataFrame([{'value': str(data)}]); df.to_excel(writer, sheet_name='Data', index=False); sheets_created = True

                if not sheets_created: # Fallback if no data was written
                    df_error = pd.DataFrame([{'error': 'No processable data found or data was empty', 'original_data_type': str(type(data))}])
                    df_error.to_excel(writer, sheet_name='Error', index=False)

            asyncio.run(_update_job_if_present(progress_val=0.95, current_step_val="Excel file generated"))
            return file_id, xlsx_filename, file_path
            
        except Exception as e:
            current_progress_on_error = 0.10 + (retry_count * 0.05) # Show some progress even on retry
            asyncio.run(_update_job_if_present(progress_val=current_progress_on_error, current_step_val=f"Attempt {retry_count+1} failed: {str(e)[:50]}..."))
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

    # Check for sufficient disk space before starting any processing that might create files
    if not _has_sufficient_disk_space(temp_dir, MIN_FREE_SPACE_BYTES):
        error_msg = f"Insufficient disk space in {temp_dir} for Agno agent. Requires at least {MIN_FREE_SPACE_BYTES // (1024*1024)}MB free."
        logger.error(error_msg)
        if job_manager and job_id:
            asyncio.run(job_manager.update_job_status(job_id, JobStatus.FAILED, current_step="Disk space check failed for Agno", result={"error": error_msg}))
        raise IOError(error_msg)

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

    # Adjusted retry parameters for ~30 second total wait time with 10 attempts
    max_attempts = 10
    current_delay = 0.25  # Initial delay in seconds
    backoff_factor = 1.5
    total_slept_time = 0
    
    files_after = set()
    new_files = set()

    for attempt in range(max_attempts):
        # Perform file search
        current_files_after_scan = set()
        for pattern in patterns: current_files_after_scan.update(glob.glob(pattern, recursive=True))
        files_after.update(current_files_after_scan) # Accumulate files found over attempts, in case of very slow writes
        
        new_files_on_this_attempt = files_after - files_before
        
        if new_files_on_this_attempt:
            logger.info(f"File detection: Found {len(new_files_on_this_attempt)} new candidate(s) on attempt {attempt + 1}.")
            # Check readability and size immediately for candidates found on this attempt
            # This allows quicker exit if a valid file is found early.
            # Sort these candidates to check the newest one first.
            def get_file_sort_key_attempt(f):
                try: return (os.path.getmtime(f), os.path.getctime(f))
                except OSError: return (0,0)

            sorted_candidates = sorted(list(new_files_on_this_attempt), key=get_file_sort_key_attempt, reverse=True)

            for candidate_file in sorted_candidates:
                try:
                    if os.path.exists(candidate_file) and \
                       os.access(candidate_file, os.R_OK) and \
                       os.path.getsize(candidate_file) > 0:
                        logger.info(f"Valid file found: {candidate_file} (size: {os.path.getsize(candidate_file)}B) on attempt {attempt + 1}.")
                        asyncio.run(_update_job_if_present(current_step_val="Output file found", progress_val=0.98))
                        return candidate_file # Exit as soon as a valid file is found
                except OSError as e_check:
                    logger.warning(f"Error checking candidate file {candidate_file} on attempt {attempt+1}: {e_check}")
            # If no valid file found among this attempt's candidates, new_files remains the cumulative set for next logging/final check
            new_files = new_files_on_this_attempt # Update new_files to what was found in this pass for logging below

        if attempt < max_attempts - 1: # Don't sleep after the last attempt
            if not new_files_on_this_attempt: # Only log "no new files" if none were found in this specific attempt
                 logger.info(f"File detection attempt {attempt + 1}/{max_attempts}: No new files yet in {directory}. Sleeping for {current_delay:.2f}s.")
            else: # Log that candidates were found but none were valid yet
                 logger.info(f"File detection attempt {attempt + 1}/{max_attempts}: Found candidates, but none valid yet. Sleeping for {current_delay:.2f}s.")

            time.sleep(current_delay)
            total_slept_time += current_delay
            current_delay *= backoff_factor
            asyncio.run(_update_job_if_present(progress_val=0.85 + (0.1 * (attempt + 1) / max_attempts)))
        elif new_files_on_this_attempt: # Last attempt, and some new files were seen in this last pass
             logger.info(f"File detection: Last attempt ({max_attempts}/{max_attempts}). Found candidates, but none were valid in the final check pass.")
        else: # Last attempt, no new files found in this pass
             logger.info(f"File detection: Last attempt ({max_attempts}/{max_attempts}). No new files found.")


    if settings.AGNO_DEBUG:
        logger.info(f"FILE DEBUG (after all attempts): Total slept time: {total_slept_time:.2f}s")
        logger.info(f"Dir={directory}, Exists={os.path.exists(directory)}, Patterns={patterns}")
        logger.info(f"Files Before ({len(files_before)}): {list(files_before)}")
        logger.info(f"Files After (cumulative list of files found across attempts) ({len(files_after)}): {list(files_after)}")
        all_files_in_dir = [os.path.join(r,f) for r,ds,fs in os.walk(directory) for f in fs] if os.path.exists(directory) else []
        logger.info(f"ALL files in dir tree ({len(all_files_in_dir)}): {all_files_in_dir}")
        # new_files here would be the candidates from the *last* attempt if we didn't find a valid one earlier
        # It's better to log the set of all new files ever considered:
        all_new_candidates_considered = files_after - files_before
        logger.info(f"All new candidates considered ({len(all_new_candidates_considered)}): {list(all_new_candidates_considered)}")

    # After the loop, if we haven't returned a file, it means no valid file was found
    # The `new_files` variable at this stage would reflect candidates from the last attempt,
    # or an earlier attempt if that's where the loop stopped due to finding candidates (none of which were valid).
    # It's more accurate to check `all_new_candidates_considered`.
    if not (files_after - files_before): # No new files ever appeared across all attempts
        logger.warning(f"No new Excel files appeared in {directory} after {max_attempts} attempts and ~{total_slept_time:.2f}s total sleep time.")
        asyncio.run(_update_job_if_present(status_val=JobStatus.FAILED, current_step_val="Output file not found", result_val={"error": "No output file generated by AI."}))
        return None
    
    # If we reach here, it means new files might have appeared at some point, but none passed all validation checks
    # (exists, readable, size > 0) during the attempts where they were first seen.
    # We can do one final check on all accumulated new files.
    final_new_candidates = files_after - files_before
    logger.info(f"Performing final validation on {len(final_new_candidates)} accumulated new candidates...")
    
    if not final_new_candidates: # Should have been caught by the above, but as a safeguard.
        logger.warning(f"No new files identified for final validation despite earlier indications.")
        asyncio.run(_update_job_if_present(status_val=JobStatus.FAILED, current_step_val="Output file disappeared or error in logic", result_val={"error": "Output file candidates lost."}))
        return None

    def get_file_sort_key_final(f):
        try: return (os.path.getmtime(f), os.path.getctime(f))
        except OSError: return (0,0)

    sorted_final_candidates = sorted(list(final_new_candidates), key=get_file_sort_key_final, reverse=True)

    for potential_file in sorted_final_candidates:
        try:
            if os.path.exists(potential_file) and \
               os.access(potential_file, os.R_OK) and \
               os.path.getsize(potential_file) > 0:
                logger.info(f"Final validation successful: {potential_file} (size: {os.path.getsize(potential_file)}B)")
                asyncio.run(_update_job_if_present(current_step_val="Output file found (final check)", progress_val=0.98))
                return potential_file
            else:
                logger.warning(f"Final validation: Candidate {potential_file} not valid (exists: {os.path.exists(potential_file)}, readable: {os.access(potential_file, os.R_OK)}, size: {os.path.getsize(potential_file) if os.path.exists(potential_file) else 'N/A'}).")
        except OSError as e_final_check:
            logger.warning(f"Error during final validation of {potential_file}: {e_final_check}")

    logger.warning(f"All {len(final_new_candidates)} accumulated new file(s) failed final validation or were inaccessible.")
    asyncio.run(_update_job_if_present(status_val=JobStatus.FAILED, current_step_val="Output file(s) found but invalid/inaccessible", result_val={"error": "Generated file(s) were invalid or inaccessible."}))
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


# --- Model Listing Logic ---
# In-memory cache for models: (timestamp, model_list)
_model_cache: Optional[Tuple[float, List[Dict[str, Any]]]] = None
_MODEL_CACHE_TTL_SECONDS = 5 * 60  # 5 minutes

# Fallback model list
_FALLBACK_MODELS = [
    {
        "id": "gemini-1.5-flash-latest",
        "name": "Gemini 1.5 Flash (Fallback)",
        "description": "Fast and versatile multimodal model for a variety of tasks. (Fallback data)",
        "version": "1.5-flash",
        "provider": "Google",
        "input_token_limit": 1048576, # Based on public info for 1.5 Flash
        "output_token_limit": 8192,    # Based on public info for 1.5 Flash
        "pricing_details_url": "https://cloud.google.com/vertex-ai/generative-ai/pricing",
        "notes": "This is a fallback entry. Live data may vary."
    },
    {
        "id": "gemini-1.0-pro-latest",
        "name": "Gemini 1.0 Pro (Fallback)",
        "description": "Mid-size multimodal model for a wide range of tasks. (Fallback data)",
        "version": "1.0-pro",
        "provider": "Google",
        "input_token_limit": 30720, # Based on public info for 1.0 Pro (text)
        "output_token_limit": 2048,   # Based on public info for 1.0 Pro (text)
        "pricing_details_url": "https://cloud.google.com/vertex-ai/generative-ai/pricing",
        "notes": "This is a fallback entry. Live data may vary."
    },
]

# Known token limits for some models (approximations, as API doesn't provide this directly)
# See: https://ai.google.dev/models/gemini (Rates and quotas section often has context window info)
# And: https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models
MODEL_TOKEN_LIMITS = {
    "gemini-1.0-pro": {"input": 30720, "output": 2048}, # For gemini-1.0-pro-001, text only
    "gemini-1.0-pro-latest": {"input": 30720, "output": 2048},
    "gemini-1.0-pro-001": {"input": 30720, "output": 2048},
    "gemini-pro": {"input": 30720, "output": 2048}, # Alias, often refers to 1.0 Pro

    "gemini-1.5-flash-latest": {"input": 1048576, "output": 8192}, # Context window 1M, output 8K
    "gemini-1.5-pro-latest": {"input": 1048576, "output": 8192},   # Context window 1M, output 8K

    # Vision models have different limits, typically lower for text part
    "gemini-1.0-pro-vision-latest": {"input": 12288, "output": 4096}, # Text part for vision model
    "gemini-pro-vision": {"input": 12288, "output": 4096}, # Alias
}


async def list_available_models() -> List[Dict[str, Any]]:
    """
    Lists available generative models from Google Generative AI.
    Uses caching and falls back to a hardcoded list on failure.
    """
    global _model_cache
    current_time = time.time()

    if _model_cache and (current_time - _model_cache[0]) < _MODEL_CACHE_TTL_SECONDS:
        logger.info("Returning cached model list.")
        return _model_cache[1]

    try:
        logger.info("Fetching model list from Google Generative AI API...")
        import google.generativeai as genai
        # Ensure API key is configured for the genai module if not done globally
        if not genai.conf.api_key and settings.GOOGLE_API_KEY:
             genai.configure(api_key=settings.GOOGLE_API_KEY)
        elif not genai.conf.api_key and not settings.GOOGLE_API_KEY:
            logger.error("Google API Key not configured for genai. Falling back to hardcoded models.")
            _model_cache = (current_time, _FALLBACK_MODELS)
            return _FALLBACK_MODELS

        processed_models = []
        # The genai.list_models() is synchronous, so run in thread pool
        raw_models_iterator = await asyncio.to_thread(genai.list_models)

        for model_obj in raw_models_iterator:
            # We are interested in models that support 'generateContent' for chat/text generation
            # and are not embedding models.
            if 'generateContent' in model_obj.supported_generation_methods and \
               not model_obj.name.startswith('models/embedding') and \
               not model_obj.name.startswith('models/aqa'): # Attributed Question Answering models

                model_id = model_obj.name.replace("models/", "") # Strip "models/" prefix for user-friendliness

                token_limits = MODEL_TOKEN_LIMITS.get(model_id, {}) # Get known limits
                # Try to find limits for base model if specific version not found
                if not token_limits:
                    base_model_id_parts = model_id.split('-')
                    if len(base_model_id_parts) > 2 and not base_model_id_parts[-1].startswith(('0','1','latest')): # e.g. gemini-1.5-pro-001 -> gemini-1.5-pro
                         base_model_id_prefix = '-'.join(base_model_id_parts[:-1])
                         token_limits = MODEL_TOKEN_LIMITS.get(base_model_id_prefix + "-latest",
                                         MODEL_TOKEN_LIMITS.get(base_model_id_prefix, {}))


                processed_models.append({
                    "id": model_id,
                    "name": model_obj.display_name,
                    "description": model_obj.description,
                    "version": model_obj.version,
                    "provider": "Google",
                    "input_token_limit": token_limits.get("input"),
                    "output_token_limit": token_limits.get("output"),
                    "pricing_details_url": "https://cloud.google.com/vertex-ai/generative-ai/pricing",
                    "notes": None
                })

        if not processed_models:
            logger.warning("No suitable generative models found via API. Falling back.")
            _model_cache = (current_time, _FALLBACK_MODELS)
            return _FALLBACK_MODELS

        logger.info(f"Successfully fetched and processed {len(processed_models)} models.")
        _model_cache = (current_time, processed_models)
        return processed_models

    except Exception as e:
        logger.error(f"Failed to list models from Google API: {e}", exc_info=True)
        logger.warning("Falling back to hardcoded model list.")
        _model_cache = (current_time, _FALLBACK_MODELS) # Cache fallback to prevent repeated API errors
        return _FALLBACK_MODELS
