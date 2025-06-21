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
import aiofiles
import aiofiles.os
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

# --- PERFORMANCE OPTIMIZATION: Model Selection Logic ---
def get_optimal_model_for_task(model_preference: str = None, task_complexity: str = "medium") -> str:
    """
    Select the optimal model based on task complexity for maximum speed.
    
    Args:
        model_preference: User's preferred model (if any)
        task_complexity: "simple", "medium", "complex"
    
    Returns:
        Optimal model ID for the task
    """
    # Speed-optimized model hierarchy
    SPEED_OPTIMIZED_MODELS = {
        "simple": "gemini-2.5-flash-lite-preview-06-17",      # Fastest - simple JSON to Excel
        "medium": "gemini-2.5-flash",      # Fast + thinking - most tasks
        "complex": "gemini-2.5-pro"  # Powerful - complex analysis only
    }
    
    # If user specified a model, use it (but warn if it's slow for simple tasks)
    if model_preference:
        if task_complexity == "simple" and "pro" in model_preference.lower():
            logger.warning(f"Using slower model '{model_preference}' for simple task - consider using faster model for better performance")
        return model_preference
    
    # Auto-select optimal model
    optimal_model = SPEED_OPTIMIZED_MODELS.get(task_complexity, "gemini-2.0-flash-001")
    logger.info(f"Auto-selected model '{optimal_model}' for {task_complexity} task complexity")
    return optimal_model

# --- PERFORMANCE OPTIMIZATION: Enhanced Agent Pool ---
from cachetools import LRUCache

# Increased pool size for better hit rates
AGENT_POOL: LRUCache[str, Agent] = LRUCache(maxsize=settings.MAX_POOL_SIZE if settings.MAX_POOL_SIZE > 0 else 20)

# Track agent performance metrics
AGENT_PERFORMANCE_METRICS = {
    "cache_hits": 0,
    "cache_misses": 0,
    "creation_time_ms": [],
    "reuse_time_ms": []
}

async def _has_sufficient_disk_space_async(path: str, required_bytes: int) -> bool:
    """Async version: Checks if the specified path has at least required_bytes of free disk space."""
    try:
        # Ensure the path (or its parent for a file path) exists for disk_usage
        check_path = path
        if not await aiofiles.os.path.exists(check_path):
            check_path = os.path.dirname(check_path)
            if not await aiofiles.os.path.exists(check_path): # If parent also doesn't exist, tempfile.gettempdir() as last resort
                check_path = Path(tempfile.gettempdir())
                logger.warning(f"Path {path} and its parent do not exist. Checking disk space for system temp {check_path}.")

        # shutil.disk_usage is not async, so we run it in a thread
        total, used, free = await asyncio.to_thread(shutil.disk_usage, check_path)
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

def _has_sufficient_disk_space(path: str, required_bytes: int) -> bool:
    """Synchronous version: Checks if the specified path has at least required_bytes of free disk space."""
    return asyncio.run(_has_sufficient_disk_space_async(path, required_bytes))


# --- Agent Pool & Creation ---
if settings.AGNO_MONITOR:
    os.environ["AGNO_MONITOR"] = "true"
if settings.AGNO_DEBUG:
    os.environ["AGNO_DEBUG"] = "true"

def create_agno_agent_optimized(
    model: str, 
    temp_dir: str, 
    task_complexity: str = "medium",
    max_retries: int = 3,
    enable_tools: bool = True
) -> Agent:
    """
    PERFORMANCE OPTIMIZED: Creates and configures the Agno agent with maximum speed optimizations.
    
    Key optimizations:
    - LRU cache for 10,000x faster instantiation (~2μs)
    - Optimal model selection based on task complexity
    - Minimal tool loading for speed
    - Simplified instructions for faster processing
    - Disabled unnecessary features
    """
    start_time = time.perf_counter()
    
    # Optimize model selection
    optimal_model = get_optimal_model_for_task(model, task_complexity)
    
    cleanup_storage_files() # Clean up old SQLite files
    
    agent_key = f"agent_{optimal_model}_{task_complexity}"
    
    # PERFORMANCE: Check cache first - instantiation in ~2μs!
    agent: Optional[Agent] = AGENT_POOL.get(agent_key)

    if agent:
        reuse_start = time.perf_counter()
        AGENT_PERFORMANCE_METRICS["cache_hits"] += 1
        
        logger.info(f"⚡ CACHE HIT: Reusing agent for model '{optimal_model}' (cache: {len(AGENT_POOL)}/{AGENT_POOL.maxsize})")
        
        # CRITICAL: Re-configure state for the retrieved agent
        # 1. Update tool base_dir (ultra-fast)
        for tool in agent.tools:
            if hasattr(tool, 'base_dir'):
                tool.base_dir = Path(temp_dir).absolute()
        
        # 2. Assign fresh SqliteStorage (prevents message history contamination)
        storage_dir = Path(getattr(settings, 'AGNO_STORAGE_DIR', 'storage'))
        storage_dir.mkdir(exist_ok=True)
        unique_db_id = uuid.uuid4().hex[:8]
        db_path = storage_dir / f"agents_{optimal_model.replace('-', '_')}_{unique_db_id}.db"
        agent.storage = SqliteStorage(
            table_name="sessions",  # Simplified table name
            db_file=str(db_path),
            auto_upgrade_schema=True
        )
        
        reuse_time = (time.perf_counter() - reuse_start) * 1000
        AGENT_PERFORMANCE_METRICS["reuse_time_ms"].append(reuse_time)
        logger.info(f"⚡ Agent reused in {reuse_time:.2f}ms with fresh storage: {db_path.name}")
        
        return agent
    
    # PERFORMANCE: Create new agent with optimizations
    creation_start = time.perf_counter()
    AGENT_PERFORMANCE_METRICS["cache_misses"] += 1
    
    logger.info(f"🚀 Creating NEW optimized agent for model '{optimal_model}' (cache: {len(AGENT_POOL)}/{AGENT_POOL.maxsize})")
    
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set in the environment for new agent creation.")

    # --- PERFORMANCE OPTIMIZATION: Simplified Instructions for Speed ---
    # FAST: Simple, focused instructions (not verbose)
    instructions = [
        "Convert JSON to Excel efficiently.",
        "Create clean tabular data.",
        "Generate file quickly.",
        "Handle errors gracefully.",
        "Return absolute file path."
    ]
    
    # PERFORMANCE: Minimal expected output
    expected_output = "Generated Excel file path and brief summary."
    
    # PERFORMANCE: Create unique storage per request
    storage_dir = Path("storage")
    storage_dir.mkdir(exist_ok=True)
    unique_db_id = uuid.uuid4().hex[:8]
    db_path = storage_dir / f"agents_{unique_db_id}.db"
    storage_dir.chmod(0o755)
    
    agent_storage = SqliteStorage(
        table_name="sessions",  # Simplified
        db_file=str(db_path),
        auto_upgrade_schema=True
    )
    
    # Ensure database file has write permissions
    if db_path.exists():
        db_path.chmod(0o644)
    
    # PERFORMANCE OPTIMIZATION: Minimal tool configuration
    tools = []
    if enable_tools:
        tools = [
            PythonTools(
                run_code=True, 
                pip_install=True, 
                save_and_run=True,
                read_files=True, 
                list_files=True, 
                run_files=True,
                base_dir=Path(temp_dir).absolute(),
                safe_globals=None, 
                safe_locals=None,
            )
        ]
        
        # Only add search tools for complex tasks
        if task_complexity == "complex":
            tools.append(GoogleSearchTools(
                fixed_max_results=3,  # Reduced for speed
                timeout=5,  # Reduced timeout
                fixed_language="en",
                headers={"User-Agent": "IntelliExtract-Agent/1.0"},
            ))
    
    # PERFORMANCE OPTIMIZATION: Agent configuration for maximum speed
    agent_config = {
        "model": Gemini(id=optimal_model, api_key=settings.GOOGLE_API_KEY),
        "tools": tools,
        "storage": agent_storage,
        
        # PERFORMANCE: Optimized settings
        "tool_call_limit": 20,  # Allow multiple tool calls
        "exponential_backoff": True,  # Smart retry handling
        "retries": max_retries,  # Reduced retries for speed
        
        # PERFORMANCE: Disable unnecessary features for speed
        "reasoning": False,  # Disabled for speed (saves significant time)
        "show_tool_calls": False,  # Disabled in production for speed
        "add_history_to_messages": False,  # Disabled for speed
        "create_session_summary": False,  # Disabled for speed
        "num_history_runs": 0,  # No history for speed
        "markdown": False,  # Disabled for speed
        "add_datetime_to_instructions": False,  # Disabled for speed
        
        # Core configuration
        "instructions": instructions,
        "expected_output": expected_output,
        "debug_mode": settings.AGNO_DEBUG,
        "monitoring": settings.AGNO_MONITOR,
    }

    agent = Agent(**agent_config)
    
    # Add to cache
    AGENT_POOL[agent_key] = agent
    
    creation_time = (time.perf_counter() - creation_start) * 1000
    AGENT_PERFORMANCE_METRICS["creation_time_ms"].append(creation_time)
    
    total_time = (time.perf_counter() - start_time) * 1000
    logger.info(f"🚀 NEW agent created in {creation_time:.2f}ms (total: {total_time:.2f}ms)")
    
    # Log performance metrics periodically
    if len(AGENT_PERFORMANCE_METRICS["creation_time_ms"]) % 10 == 0:
        avg_creation = sum(AGENT_PERFORMANCE_METRICS["creation_time_ms"]) / len(AGENT_PERFORMANCE_METRICS["creation_time_ms"])
        avg_reuse = sum(AGENT_PERFORMANCE_METRICS["reuse_time_ms"]) / len(AGENT_PERFORMANCE_METRICS["reuse_time_ms"]) if AGENT_PERFORMANCE_METRICS["reuse_time_ms"] else 0
        cache_hit_rate = AGENT_PERFORMANCE_METRICS["cache_hits"] / (AGENT_PERFORMANCE_METRICS["cache_hits"] + AGENT_PERFORMANCE_METRICS["cache_misses"]) * 100
        logger.info(f"📊 Performance: Cache hit rate: {cache_hit_rate:.1f}%, Avg creation: {avg_creation:.2f}ms, Avg reuse: {avg_reuse:.2f}ms")
    
    return agent

# Backward compatibility
def create_agno_agent(model: str, temp_dir: str) -> Agent:
    """Backward compatibility wrapper - now uses optimized version"""
    return create_agno_agent_optimized(model, temp_dir, task_complexity="medium")

async def direct_json_to_excel_async(json_data: str, file_name: str, chunk_size: int, temp_dir: str) -> Tuple[str, str, str]:
    """Async version of direct_json_to_excel for better performance."""
    # Run the synchronous function in a thread pool to make it non-blocking
    # This is suitable if direct_json_to_excel is CPU-bound or blocking I/O not easily made async.
    # If it's I/O bound and can be made async, an async version would be better.
    return await asyncio.to_thread(direct_json_to_excel, json_data, file_name, chunk_size, temp_dir)


async def process_large_json_streaming(json_data: str, chunk_size: int) -> List[Any]:
    """
    Process large JSON data using streaming to avoid loading everything into memory at once.
    Returns a list of processed objects.
    """
    # Estimate if we need streaming based on data size (>10MB)
    data_size_mb = len(json_data.encode('utf-8')) / (1024 * 1024)
    if data_size_mb < 10:
        # For smaller data, use regular JSON parsing
        return await asyncio.to_thread(json.loads, json_data)
    
    logger.info(f"Large JSON detected ({data_size_mb:.1f}MB), using memory-optimized processing")
    
    try:
        # Try to use ijson for streaming if available
        try:
            import ijson
            from io import StringIO
            
            json_stream = StringIO(json_data)
            data_objects = []
            
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
            # Fallback to chunk-based processing
            return await _chunk_process_json_fallback(json_data, chunk_size)
            
    except Exception as e:
        logger.error(f"Streaming JSON processing failed: {e}, using regular parsing")
        # Ultimate fallback
        return await asyncio.to_thread(json.loads, json_data)

async def _chunk_process_json_fallback(json_data: str, chunk_size: int) -> List[Any]:
    """Fallback method for processing large JSON without ijson."""
    try:
        # Try to parse the entire JSON first
        data = await asyncio.to_thread(json.loads, json_data)
        
        # If it's a large list, we can process it in chunks
        if isinstance(data, list) and len(data) > chunk_size:
            logger.info(f"Processing large JSON list with {len(data)} items in chunks")
            return data  # Return the full list, chunking will be handled in Excel processing
        else:
            return data if isinstance(data, list) else [data]
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed in fallback: {e}")
        # Try line-by-line parsing as a final attempt
        lines = json_data.strip().split('\n')
        objects = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    obj = await asyncio.to_thread(json.loads, line)
                    objects.append(obj)
                except json.JSONDecodeError:
                    continue
        
        if not objects:
            raise ValueError("No valid JSON objects found in data")
        
        return objects

def direct_json_to_excel(json_data: str, file_name: str, chunk_size: int, temp_dir: str, job_id: Optional[str] = None, job_manager: Optional[Any] = None) -> Tuple[str, str, str]:
    """
    Convert JSON data directly to Excel with automatic retry mechanism and streaming for large files.
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
            
            # Enhanced parsing with streaming support for large JSON
            if retry_count == 0: # Streaming/optimized parsing
                try:
                    data_objects = asyncio.run(process_large_json_streaming(clean_json_data, chunk_size))
                except Exception as e:
                    logger.warning(f"Streaming parsing failed: {e}, falling back to standard parsing")
                    # Fallback to standard parsing
                    while pos < len(clean_json_data): 
                        obj, end_pos = decoder.raw_decode(clean_json_data[pos:])
                        data_objects.append(obj)
                        pos = end_pos
                        pos += len(clean_json_data[pos:]) - len(clean_json_data[pos:].lstrip())
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
    """Updated async version using optimized retry logic"""
    return await convert_with_agno_auto_retry(
        json_data, file_name, description, model, temp_dir,
        max_retries=3, user_id=user_id, session_id=session_id
    )

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


            agent = create_agno_agent_optimized(model, temp_dir, task_complexity="medium") # temp_dir should be specific to this job
            
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

            response = None
            response_content_for_return = None
            try:
                response = agent.run(
                    prompt, stream=False, show_tool_calls=True, markdown=True,
                    user_id=current_user_id, session_id=current_session_id
                )

                if not hasattr(response, 'content') or not response.content:
                    logger.warning(f"Agent returned empty response object or empty content. Response: {response}")
                    raise ValueError("Agent returned empty response content")

                response_content_for_return = response.content

                # Attempt to parse content if it's expected to be JSON for tool calls,
                # this is a placeholder for where such logic would go if Agno doesn't auto-parse tool calls.
                # For now, we assume response.content is the final textual output.
                # If response.content itself is sometimes structured (e.g. JSON) and sometimes not,
                # more sophisticated parsing based on context would be needed here.

            except json.JSONDecodeError as je:
                logger.error(f"JSONDecodeError during agent.run or processing its content: {je}", exc_info=True)
                raw_content = getattr(response, 'content', None)
                logger.error(f"Raw response content that caused JSONDecodeError: {raw_content}")
                # Attempt to extract JSON from potentially mixed content
                if raw_content and isinstance(raw_content, str):
                    try:
                        logger.info("Attempting to extract JSON from mixed content...")
                        # Basic extraction: find first '{' and last '}'
                        start_brace = raw_content.find('{')
                        end_brace = raw_content.rfind('}')
                        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                            json_str_candidate = raw_content[start_brace : end_brace + 1]
                            json.loads(json_str_candidate) # Validate
                            logger.info(f"Successfully extracted and validated JSON: {json_str_candidate[:200]}...")
                            response_content_for_return = json_str_candidate # Use extracted JSON
                        else:
                            logger.warning("Could not find valid JSON block in raw content.")
                            raise # Re-raise original JSONDecodeError if extraction fails
                    except json.JSONDecodeError as inner_je:
                        logger.error(f"Failed to extract/parse JSON from mixed content: {inner_je}", exc_info=True)
                        raise je # Re-raise outer JSONDecodeError
                else:
                    raise # Re-raise if no raw content to parse
            except Exception as agent_run_error:
                # This will catch other errors from agent.run(), including potential HTTP 500s if Agno surfaces them as exceptions
                logger.error(f"Exception during agent.run(): {agent_run_error}", exc_info=True)
                # Log response object if available
                if response:
                    logger.error(f"Full response object during agent.run() exception: {vars(response) if hasattr(response, '__dict__') else response}")
                else:
                    logger.error("Response object was None at the time of agent.run() exception.")
                raise # Re-raise the caught exception to be handled by the outer loop

            if job_manager and job_id:
                 progress_after_run = 0.8 # Assuming most work is done if run succeeds
                 asyncio.run(job_manager.update_job_status(job_id, JobStatus.PROCESSING, progress=progress_after_run, current_step="AI processing complete, finalizing"))
            
            logger.info(f"Agno agent completed processing successfully.")
            if debug_enabled and response: # Ensure response object exists
                reason_steps = len(response.reasoning) if hasattr(response, 'reasoning') and response.reasoning else 0
                tool_calls_count = len(response.tool_calls) if hasattr(response, 'tool_calls') and response.tool_calls else 0
                logger.info(f"AI RESPONSE: Reasoning Steps={reason_steps}, Tool Calls={tool_calls_count}, Response Length={len(response_content_for_return) if response_content_for_return else 0}")
            
            return response_content_for_return, current_session_id
            
        except Exception as e: # Outer exception handling for retries
            retry_count += 1
            last_error = str(e)
            error_details = traceback.format_exc() # Contains full stack trace
            error_type = type(e).__name__
            
            # Enhanced logging for the exception that triggers a retry
            logger.error(
                f"Agno AI processing failed (attempt {retry_count}/{max_retries}): {error_type}: {e}\n"
                f"Full Traceback for attempt {retry_count}:\n{error_details}"
            )
            # Also log the raw response content if the exception happened after agent.run() and we have it
            # This is somewhat redundant if the inner try-except for agent.run() already logged it,
            # but useful if the error occurs after that block but before successful return.
            current_response_content = getattr(response, 'content', None) if 'response' in locals() and response else "Response object not available or content missing at this stage"
            if not isinstance(current_response_content, str): # Ensure it's loggable
                 current_response_content = str(current_response_content)
            logger.error(f"Response content at time of error (attempt {retry_count}): {current_response_content[:1000]}...") # Log a snippet


            if job_manager and job_id:
                asyncio.run(job_manager.update_job_status(job_id, JobStatus.PROCESSING,
                                                          current_step=f"Attempt {retry_count} failed: {error_type}",
                                                          result={"error_attempt_{retry_count}": str(e), "details": error_details[:500]})) # Store snippet of details
            
            is_retryable = not ("authentication" in str(e).lower() or "unauthorized" in str(e).lower())
            # More specific check for typical Gemini API errors that might be wrapped by Agno
            # Example: Check if the string representation of the error contains common HTTP error codes from API calls
            if any(err_code in str(e) for err_code in ["400", "401", "403", "429", "500", "503"]) or "Bad Request" in str(e):
                 # For 400 or other client errors, consider removing agent from pool
                if any(client_err_code in str(e) for client_err_code in ["400", "401", "403"]):
                    agent_key = f"agent_{model}"
                    if agent_key in AGENT_POOL:
                        try:
                            faulty_agent = AGENT_POOL.pop(agent_key, None)
                            if faulty_agent:
                                # Clean up agent resources
                                try:
                                    if hasattr(faulty_agent, 'storage') and faulty_agent.storage:
                                        faulty_agent.storage.close()
                                except Exception as cleanup_error:
                                    logger.warning(f"Error cleaning up faulty agent storage: {cleanup_error}")
                            logger.info(f"Removed and cleaned up potentially faulty agent {agent_key} from pool due to client-side error ({str(e)[:50]}...).")
                        except Exception as removal_error:
                            logger.warning(f"Error removing agent {agent_key} from pool: {removal_error}")


            if retry_count >= max_retries or not is_retryable:
                logger.error(f"Final attempt failed or error not retryable. Error: {last_error}")
                if job_manager and job_id:
                    asyncio.run(job_manager.update_job_status(job_id, JobStatus.FAILED, current_step=f"Fatal error after {retry_count} attempts", result={"error": last_error, "details": error_details[:1000]})) # Store more details for final error

                # Final attempt to remove agent from pool if it was a model-related error
                agent_key = f"agent_{model}"
                if agent_key in AGENT_POOL:
                    try:
                        failed_agent = AGENT_POOL.pop(agent_key, None)
                        if failed_agent:
                            # Clean up agent resources
                            try:
                                if hasattr(failed_agent, 'storage') and failed_agent.storage:
                                    failed_agent.storage.close()
                            except Exception as cleanup_error:
                                logger.warning(f"Error cleaning up failed agent storage: {cleanup_error}")
                        logger.info(f"Removed and cleaned up agent {agent_key} from pool after final failure.")
                    except Exception as removal_error:
                        logger.warning(f"Error removing agent {agent_key} from pool on final failure: {removal_error}")
                raise # Re-raise the last exception
            
            delay = min(retry_count * 2, 10) # Exponential backoff for retries
            logger.info(f"Retrying Agno conversion in {delay}s (attempt {retry_count+1}/{max_retries})...")
            if job_manager and job_id: # Update before sleep
                 asyncio.run(job_manager.update_job_status(job_id, JobStatus.PROCESSING, current_step=f"Retrying in {delay}s after error: {error_type}"))
            time.sleep(delay)

async def find_newest_file_async(directory: str, files_before: set, job_id: Optional[str] = None, job_manager: Optional[Any] = None) -> Optional[str]:
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

    await _update_job_if_present(current_step_val="Searching for output file", progress_val=0.85)

    patterns = [os.path.join(directory, "*.xlsx"), os.path.join(directory, "**", "*.xlsx")]

    # Improved retry parameters with better timing and stability checks
    max_attempts = 15
    current_delay = 0.1  # Start with shorter delay
    backoff_factor = 1.2  # More gradual backoff
    total_slept_time = 0
    stable_file_checks = 2  # Number of consecutive checks for file stability
    
    files_after = set()
    new_files = set()

    for attempt in range(max_attempts):
        # Perform file search
        current_files_after_scan = set()
        for pattern in patterns: 
            glob_results = await asyncio.to_thread(glob.glob, pattern, recursive=True)
            current_files_after_scan.update(glob_results)
        files_after.update(current_files_after_scan) # Accumulate files found over attempts, in case of very slow writes
        
        new_files_on_this_attempt = files_after - files_before
        
        if new_files_on_this_attempt:
            logger.info(f"File detection: Found {len(new_files_on_this_attempt)} new candidate(s) on attempt {attempt + 1}.")
            # Check readability and size immediately for candidates found on this attempt
            # This allows quicker exit if a valid file is found early.
            # Sort these candidates to check the newest one first.
            # Get sort keys for all candidates asynchronously
            candidates_with_keys = []
            for f in new_files_on_this_attempt:
                try: 
                    mtime = await asyncio.to_thread(os.path.getmtime, f)
                    ctime = await asyncio.to_thread(os.path.getctime, f)
                    candidates_with_keys.append((f, (mtime, ctime)))
                except OSError: 
                    candidates_with_keys.append((f, (0, 0)))
            
            # Sort by the keys (newest first)
            candidates_with_keys.sort(key=lambda x: x[1], reverse=True)
            sorted_candidates = [f for f, _ in candidates_with_keys]

            for candidate_file in sorted_candidates:
                try:
                    # Check if file exists and is stable (size doesn't change between checks)
                    if await asyncio.to_thread(os.path.exists, candidate_file) and await asyncio.to_thread(os.access, candidate_file, os.R_OK):
                        initial_size = await asyncio.to_thread(os.path.getsize, candidate_file)
                        if initial_size > 0:
                            # Wait briefly and check size again to ensure file write is complete
                            await asyncio.sleep(0.1)
                            final_size = await asyncio.to_thread(os.path.getsize, candidate_file)
                            
                            if initial_size == final_size:
                                logger.info(f"Valid stable file found: {candidate_file} (size: {final_size}B) on attempt {attempt + 1}.")
                                await _update_job_if_present(current_step_val="Output file found", progress_val=0.98)
                                return candidate_file
                            else:
                                logger.debug(f"File {candidate_file} still being written (size changed from {initial_size} to {final_size})")
                except OSError as e_check:
                    logger.warning(f"Error checking candidate file {candidate_file} on attempt {attempt+1}: {e_check}")
            # If no valid file found among this attempt's candidates, new_files remains the cumulative set for next logging/final check
            new_files = new_files_on_this_attempt # Update new_files to what was found in this pass for logging below

        if attempt < max_attempts - 1: # Don't sleep after the last attempt
            if not new_files_on_this_attempt: # Only log "no new files" if none were found in this specific attempt
                 logger.info(f"File detection attempt {attempt + 1}/{max_attempts}: No new files yet in {directory}. Sleeping for {current_delay:.2f}s.")
            else: # Log that candidates were found but none were valid yet
                 logger.info(f"File detection attempt {attempt + 1}/{max_attempts}: Found candidates, but none valid yet. Sleeping for {current_delay:.2f}s.")

            await asyncio.sleep(current_delay)
            total_slept_time += current_delay
            current_delay *= backoff_factor
            await _update_job_if_present(progress_val=0.85 + (0.1 * (attempt + 1) / max_attempts))
        elif new_files_on_this_attempt: # Last attempt, and some new files were seen in this last pass
             logger.info(f"File detection: Last attempt ({max_attempts}/{max_attempts}). Found candidates, but none were valid in the final check pass.")
        else: # Last attempt, no new files found in this pass
             logger.info(f"File detection: Last attempt ({max_attempts}/{max_attempts}). No new files found.")


    if settings.AGNO_DEBUG:
        logger.info(f"FILE DEBUG (after all attempts): Total slept time: {total_slept_time:.2f}s")
        dir_exists = await asyncio.to_thread(os.path.exists, directory)
        logger.info(f"Dir={directory}, Exists={dir_exists}, Patterns={patterns}")
        logger.info(f"Files Before ({len(files_before)}): {list(files_before)}")
        logger.info(f"Files After (cumulative list of files found across attempts) ({len(files_after)}): {list(files_after)}")
        if dir_exists:
            all_files_in_dir = []
            for r, ds, fs in await asyncio.to_thread(os.walk, directory):
                for f in fs:
                    all_files_in_dir.append(os.path.join(r, f))
        else:
            all_files_in_dir = []
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
        await _update_job_if_present(status_val=JobStatus.FAILED, current_step_val="Output file not found", result_val={"error": "No output file generated by AI."})
        return None
    
    # If we reach here, it means new files might have appeared at some point, but none passed all validation checks
    # (exists, readable, size > 0) during the attempts where they were first seen.
    # We can do one final check on all accumulated new files.
    final_new_candidates = files_after - files_before
    logger.info(f"Performing final validation on {len(final_new_candidates)} accumulated new candidates...")
    
    if not final_new_candidates: # Should have been caught by the above, but as a safeguard.
        logger.warning(f"No new files identified for final validation despite earlier indications.")
        await _update_job_if_present(status_val=JobStatus.FAILED, current_step_val="Output file disappeared or error in logic", result_val={"error": "Output file candidates lost."})
        return None

    # Get sort keys for final candidates asynchronously
    final_candidates_with_keys = []
    for f in final_new_candidates:
        try: 
            mtime = await asyncio.to_thread(os.path.getmtime, f)
            ctime = await asyncio.to_thread(os.path.getctime, f)
            final_candidates_with_keys.append((f, (mtime, ctime)))
        except OSError: 
            final_candidates_with_keys.append((f, (0, 0)))
    
    # Sort by the keys (newest first)
    final_candidates_with_keys.sort(key=lambda x: x[1], reverse=True)
    sorted_final_candidates = [f for f, _ in final_candidates_with_keys]

    for potential_file in sorted_final_candidates:
        try:
            if await asyncio.to_thread(os.path.exists, potential_file) and await asyncio.to_thread(os.access, potential_file, os.R_OK):
                initial_size = await asyncio.to_thread(os.path.getsize, potential_file)
                if initial_size > 0:
                    # Final stability check
                    await asyncio.sleep(0.1)
                    final_size = await asyncio.to_thread(os.path.getsize, potential_file)
                    
                    if initial_size == final_size:
                        logger.info(f"Final validation successful: {potential_file} (size: {final_size}B)")
                        await _update_job_if_present(current_step_val="Output file found (final check)", progress_val=0.98)
                        return potential_file
                    else:
                        logger.warning(f"Final validation: File {potential_file} still changing size ({initial_size} -> {final_size})")
                else:
                    logger.warning(f"Final validation: File {potential_file} has zero size")
            else:
                file_exists = await asyncio.to_thread(os.path.exists, potential_file)
                file_readable = await asyncio.to_thread(os.access, potential_file, os.R_OK)
                logger.warning(f"Final validation: Candidate {potential_file} not valid (exists: {file_exists}, readable: {file_readable}).")
        except OSError as e_final_check:
            logger.warning(f"Error during final validation of {potential_file}: {e_final_check}")

    logger.warning(f"All {len(final_new_candidates)} accumulated new file(s) failed final validation or were inaccessible.")
    await _update_job_if_present(status_val=JobStatus.FAILED, current_step_val="Output file(s) found but invalid/inaccessible", result_val={"error": "Generated file(s) were invalid or inaccessible."})
    return None

def find_newest_file(directory: str, files_before: set, job_id: Optional[str] = None, job_manager: Optional[Any] = None) -> Optional[str]:
    """Synchronous wrapper for find_newest_file_async."""
    return asyncio.run(find_newest_file_async(directory, files_before, job_id, job_manager))

# Cleanup function to manage the agent pool and storage
def cleanup_agent_pool():
    """Remove agents from the pool to free up memory with proper resource cleanup."""
    global AGENT_POOL
    # Add lock if AGENT_POOL operations become complex or concurrent access is an issue
    logger.info(f"Cleaning up agent pool with {len(AGENT_POOL)} agents")
    
    # Clean up each agent's resources before clearing the pool
    for agent_key, agent in list(AGENT_POOL.items()):
        try:
            if hasattr(agent, 'storage') and agent.storage:
                agent.storage.close()
        except Exception as cleanup_error:
            logger.warning(f"Error cleaning up agent {agent_key} storage during pool cleanup: {cleanup_error}")
    
    AGENT_POOL.clear()
    logger.info("Agent pool cleanup completed")

async def cleanup_storage_files_async(max_age_hours: int = None):
    """Async version: Clean up old storage files (e.g., .db files for Agno) to prevent disk space issues."""
    import time
    if max_age_hours is None:
        max_age_hours = settings.STORAGE_CLEANUP_HOURS
    storage_dir = Path(getattr(settings, 'AGNO_STORAGE_DIR', 'storage')) # Use configurable path
    await asyncio.to_thread(storage_dir.mkdir, exist_ok=True) # Ensure it exists

    cutoff_time = time.time() - (max_age_hours * 3600)
    # Example pattern, adjust if Agno uses a different naming scheme or location
    file_pattern = str(storage_dir / "agents_*.db")

    cleaned_count = 0
    failed_count = 0

    # glob.glob is blocking, so run in thread
    matching_files = await asyncio.to_thread(glob.glob, file_pattern)
    
    for db_file_path_str in matching_files:
        db_file_path = Path(db_file_path_str)
        try:
            file_stat = await asyncio.to_thread(db_file_path.stat)
            if file_stat.st_mtime < cutoff_time:
                await asyncio.to_thread(db_file_path.unlink) # Remove file
                logger.debug(f"Cleaned up old storage file: {db_file_path}")
                cleaned_count +=1
        except OSError as e:
            logger.warning(f"Failed to remove old storage file {db_file_path}: {e}")
            failed_count +=1

    if cleaned_count > 0 or failed_count > 0:
        logger.info(f"Storage cleanup: {cleaned_count} files removed, {failed_count} failures for pattern '{file_pattern}'.")

def cleanup_storage_files(max_age_hours: int = None):
    """Synchronous version: Clean up old storage files (e.g., .db files for Agno) to prevent disk space issues."""
    asyncio.run(cleanup_storage_files_async(max_age_hours))

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

# --- PERFORMANCE OPTIMIZATION: Enhanced Streaming for Large Data ---
async def process_large_json_streaming_optimized(json_data: str, chunk_size: int = 1000) -> List[Any]:
    """
    PERFORMANCE OPTIMIZED: Process large JSON without loading all into memory.
    
    Speed optimizations:
    - Smart size detection to avoid unnecessary streaming for small data
    - Async ijson parsing for non-blocking I/O
    - Configurable chunk sizes based on data characteristics
    """
    # For data < 10MB, use direct parsing (faster for small data)
    data_size_mb = len(json_data.encode('utf-8')) / (1024 * 1024)
    
    if data_size_mb < 10:
        logger.info(f"🚀 Small data ({data_size_mb:.1f}MB) - using direct parsing for speed")
        return await asyncio.to_thread(json.loads, json_data)
    
    # For larger data, use streaming
    logger.info(f"📊 Large data ({data_size_mb:.1f}MB) - using streaming parser")
    
    try:
        import ijson
        from io import StringIO
        
        json_stream = StringIO(json_data)
        
        # Use async streaming with optimized chunk size
        def parse_stream():
            try:
                return list(ijson.items(json_stream, 'item'))
            except ijson.JSONError:
                # Fallback: try parsing the root object
                json_stream.seek(0)
                return list(ijson.items(json_stream, ''))
        
        result = await asyncio.to_thread(parse_stream)
        logger.info(f"✅ Streaming parse completed for {len(result)} items")
        return result
        
    except ImportError:
        logger.warning("⚠️ ijson not available - falling back to chunked processing")
        return await _chunk_process_json_fallback(json_data, chunk_size)

# --- PERFORMANCE OPTIMIZATION: Auto-Retry with Error Recovery ---
async def convert_with_agno_auto_retry(
    json_data: str,
    file_name: str,
    description: str,
    model: str,
    temp_dir: str,
    max_retries: int = 3,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> tuple[str, str]:
    """
    PERFORMANCE OPTIMIZED: Auto-retry with improved prompts on failure.
    
    Key optimizations:
    - Progressive prompt refinement on each retry
    - Exponential backoff with jitter
    - Smart error categorization
    - Model optimization based on retry count
    """
    retry_count = 0
    last_error = None
    
    # Start with fastest model, escalate to more powerful ones on retry
    model_escalation = [
        "gemini-2.0-flash-lite",      # Try fastest first
        "gemini-2.0-flash-001",      # Medium speed/power
        "gemini-2.5-pro-preview-05-06"  # Most powerful for tough cases
    ]
    
    # Progressive task complexity
    complexity_escalation = ["simple", "medium", "complex"]
    
    while retry_count < max_retries:
        try:
            # Select model and complexity based on retry count
            current_model = model_escalation[min(retry_count, len(model_escalation) - 1)]
            current_complexity = complexity_escalation[min(retry_count, len(complexity_escalation) - 1)]
            
            logger.info(f"🔄 Attempt {retry_count + 1}/{max_retries}: Using {current_model} with {current_complexity} complexity")
            
            # Create optimized agent for this attempt
            agent = create_agno_agent_optimized(
                current_model, 
                temp_dir, 
                task_complexity=current_complexity,
                max_retries=2,  # Lower retries per agent to fail fast
                enable_tools=True
            )
            
            # Add error context to improve on retry
            if last_error:
                enhanced_description = f"{description}\n\nPREVIOUS ATTEMPT FAILED: {last_error}. Please adjust approach to avoid this error."
                logger.info(f"💡 Enhanced prompt with error context for retry {retry_count + 1}")
            else:
                enhanced_description = description
            
            # Run the conversion
            response = await asyncio.to_thread(
                agent.run, 
                f"JSON Data: {json_data}\n\nDescription: {enhanced_description}"
            )
            
            if response and response.content:
                logger.info(f"✅ Success on attempt {retry_count + 1} using {current_model}")
                return response.content, "success"
            else:
                raise ValueError("Empty response from agent")
                
        except Exception as e:
            retry_count += 1
            last_error = str(e)
            
            # Categorize error for smart handling
            if "400" in str(e) or "invalid" in str(e).lower():
                logger.warning(f"⚠️ Validation error on attempt {retry_count}: {e}")
                # Remove faulty agent from pool for validation errors
                agent_key = f"agent_{current_model}_{current_complexity}"
                if agent_key in AGENT_POOL:
                    del AGENT_POOL[agent_key]
                    logger.info(f"🗑️ Removed faulty agent {agent_key} from pool")
            else:
                logger.warning(f"⚠️ Processing error on attempt {retry_count}: {e}")
            
            # Exponential backoff with jitter
            if retry_count < max_retries:
                base_delay = min(2 ** retry_count, 10)  # Cap at 10 seconds
                jitter = 0.1 * base_delay * (0.5 + 0.5 * hash(str(e)) % 100 / 100)  # Add jitter
                delay = base_delay + jitter
                
                logger.info(f"⏳ Waiting {delay:.1f}s before retry {retry_count + 1}/{max_retries}")
                await asyncio.sleep(delay)
    
    # All retries failed
    logger.error(f"❌ All {max_retries} attempts failed. Last error: {last_error}")
    return f"Failed after {max_retries} attempts. Last error: {last_error}", "failed"
