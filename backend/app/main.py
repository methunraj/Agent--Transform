# app/main.py
import os
import uuid
import time
import glob
import shutil
import logging
import tempfile
import threading
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.concurrency import run_in_threadpool # Still potentially useful for CPU-bound tasks within async jobs

from . import schemas, services
from .core.config import settings
from .job_manager import JobManager, Job, JobStatus # Import JobManager

# --- Application State and Logging ---
app_state: Dict[str, Any] = {} # Type hint for app_state

logging.basicConfig(level=settings.LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {} # job_id -> list of WebSockets

    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
        logger.info(f"WebSocket connected for job {job_id}. Total connections for job: {len(self.active_connections[job_id])}")

    def disconnect(self, websocket: WebSocket, job_id: str):
        if job_id in self.active_connections:
            self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]: # No more connections for this job_id
                del self.active_connections[job_id]
            logger.info(f"WebSocket disconnected for job {job_id}. Remaining connections for job: {len(self.active_connections.get(job_id, []))}")

    async def broadcast_job_update(self, job: Job):
        if job.id in self.active_connections:
            message = {
                "job_id": job.id,
                "status": job.status.value,
                "progress": job.progress,
                "current_step": job.current_step,
                "result": job.result, # Include result if available
                "updated_at": job.updated_at.isoformat()
            }
            # Create a list of tasks to send messages concurrently
            send_tasks = [
                conn.send_json(message) for conn in self.active_connections[job.id]
            ]
            results = await asyncio.gather(*send_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error sending WebSocket update for job {job.id} to connection {i}: {result}")
                    # Optionally, handle disconnects if send fails
                    # self.disconnect(self.active_connections[job.id][i], job.id) # Be careful with modifying list while iterating


ws_manager = ConnectionManager()

# --- Lifespan Management (Startup and Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    # Startup
    app_state["TEMP_DIR"] = tempfile.mkdtemp(prefix=settings.TEMP_DIR_PREFIX)
    app_state["TEMP_FILES"] = {} # For direct downloads, might be phased out or integrated with job results
    app_state["METRICS"] = {
        'total_requests': 0, 'successful_conversions': 0, 'ai_conversions': 0,
        'direct_conversions': 0, 'failed_conversions': 0, 'jobs_created': 0,
        'average_processing_time': 0.0
    }
    # app_state["LOCK"] = threading.Lock() # Prefer asyncio.Lock for async code
    app_state["LOCK"] = asyncio.Lock()


    # Initialize JobManager
    job_manager = JobManager()
    app_state["JOB_MANAGER"] = job_manager
    job_manager.register_job_update_callback(ws_manager.broadcast_job_update) # Register WebSocket broadcaster
    job_manager.register_job_update_callback(sse_manager.broadcast_job_update) # Register SSE broadcaster

    # Start JobManager workers
    # The number of workers should be configurable, e.g., from settings
    num_workers = getattr(settings, 'JOB_WORKERS', 2) # Default to 2 if not in settings

    # --- API Key Validation ---
    if not settings.SKIP_API_KEY_VALIDATION_ON_STARTUP:
        if not settings.GOOGLE_API_KEY:
            logger.critical("CRITICAL: GOOGLE_API_KEY is not set. AI features will be unavailable.")
            # Consider raising an exception here to halt startup if key is absolutely mandatory
            # raise RuntimeError("GOOGLE_API_KEY is not configured. Application cannot start.")
            app_state["API_KEY_VALID"] = False
        else:
            try:
                logger.info("Performing startup API key validation for Google GenAI...")
                # Use the existing service function which includes genai.configure
                # list_available_models already has error handling and genai.configure
                await services.list_available_models() # This will attempt to list models
                logger.info("Google GenAI API key validation successful (list_models call succeeded).")
                app_state["API_KEY_VALID"] = True
            except Exception as e: # Catch broad exceptions as list_models might raise various things
                                   # including auth errors from google.api_core.exceptions
                logger.critical(f"CRITICAL: Google GenAI API key validation failed on startup: {e}", exc_info=True)
                logger.critical("AI features may be unavailable or malfunctioning. Please check GOOGLE_API_KEY and permissions.")
                app_state["API_KEY_VALID"] = False
                # Depending on strictness, could raise RuntimeError here to stop app startup
                # raise RuntimeError(f"Google GenAI API key validation failed: {e}")
    else:
        logger.warning("Skipping API key validation on startup as per SKIP_API_KEY_VALIDATION_ON_STARTUP setting.")
        # Assume valid if skipped, or handle as unknown. For now, let it proceed.
        app_state["API_KEY_VALID"] = None # Indicates skipped validation

    # --- Job Manager Setup (Redis connection and worker start) ---
    try:
        await job_manager.connect_redis() # Connect JobManager to Redis
        if job_manager.redis_client:
            if app_state.get("API_KEY_VALID", False) or settings.SKIP_API_KEY_VALIDATION_ON_STARTUP:
                # Start workers only if Redis connected AND (API key is valid or validation was skipped)
                job_manager.start_workers(num_workers)
                logger.info(f"JobManager connected to Redis. Workers started: {num_workers}.")
            else:
                logger.warning("JobManager connected to Redis, but workers NOT started due to API key validation failure.")
            logger.info(f"Application startup: Base setup complete. Temp directory: {app_state['TEMP_DIR']}.")
        else: # Redis connection failed
            logger.error("Application startup: JobManager failed to connect to Redis. Workers not started.")
            logger.info(f"Application startup: Base setup complete but JobManager NOT connected to Redis. Temp directory: {app_state['TEMP_DIR']}.")

    except Exception as e:
        logger.error(f"Error during JobManager Redis connection or worker startup: {e}", exc_info=True)
        logger.info(f"Application startup: Base setup complete with errors in JobManager setup. Temp directory: {app_state['TEMP_DIR']}.")
    
    logger.info("FastAPI application startup sequence finished.")
    yield  # Application is now running
    
    # Shutdown
    logger.info("Application shutdown. Cleaning up resources...")
    job_manager = app_state["JOB_MANAGER"]
    await job_manager.stop_workers() # Stops worker tasks
    logger.info("Job workers stopped.")
    await job_manager.close_redis() # Close Redis connection pool

    services.cleanup_agent_pool() # Assuming this is thread-safe or adapted for async

    # Clean up temp directory if it was created by this app instance
    # This might need more robust handling if multiple instances share a temp root.
    if os.path.exists(app_state["TEMP_DIR"]) and app_state["TEMP_DIR"].startswith(tempfile.gettempdir()):
        try:
            shutil.rmtree(app_state["TEMP_DIR"])
            logger.info(f"Temp directory {app_state['TEMP_DIR']} cleaned up.")
        except Exception as e:
            logger.error(f"Error cleaning up temp directory {app_state['TEMP_DIR']}: {e}")

    logger.info("Cleanup complete.")

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# --- SSE Connection Manager ---
class SSEConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[asyncio.Queue]] = {} # job_id -> list of queues for SSE

    async def connect(self, job_id: str) -> asyncio.Queue:
        queue = asyncio.Queue()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(queue)
        logger.info(f"SSE connection queue created for job {job_id}. Total queues for job: {len(self.active_connections[job_id])}")
        return queue

    def disconnect(self, queue: asyncio.Queue, job_id: str):
        if job_id in self.active_connections:
            if queue in self.active_connections[job_id]:
                self.active_connections[job_id].remove(queue)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
            logger.info(f"SSE connection queue removed for job {job_id}. Remaining queues for job: {len(self.active_connections.get(job_id, []))}")

    async def broadcast_job_update(self, job: Job):
        if job.id in self.active_connections:
            message = (
                f"id: {job.updated_at.timestamp()}\n"
                f"event: job_update\n"
                f"data: {{\"job_id\": \"{job.id}\", \"status\": \"{job.status.value}\", "
                f"\"progress\": {job.progress}, \"current_step\": \"{job.current_step or ''}\", "
                f"\"result\": {services.json_serialize_job_result(job.result)}, " # Ensure result is JSON serializable
                f"\"updated_at\": \"{job.updated_at.isoformat()}\"}}\n\n"
            )
            for queue in self.active_connections[job.id]:
                await queue.put(message)

sse_manager = SSEConnectionManager()


# --- Utility Functions ---
async def update_metrics(processing_time: float, method: str, success: bool):
    async with app_state["LOCK"]: # Use asyncio.Lock
        metrics = app_state["METRICS"]
        metrics['total_requests'] += 1 # This might be better named 'tasks_processed' or similar in job system
        if success:
            metrics['successful_conversions'] += 1
            if method == 'ai':
                metrics['ai_conversions'] += 1
            else: # direct or other
                metrics['direct_conversions'] += 1
            
            # Ensure successful_conversions is not zero before division
            if metrics['successful_conversions'] > 0:
                total_time = metrics['average_processing_time'] * (metrics['successful_conversions'] - 1)
                metrics['average_processing_time'] = (total_time + processing_time) / metrics['successful_conversions']
            else: # First successful conversion
                metrics['average_processing_time'] = processing_time
        else:
            metrics['failed_conversions'] += 1

# --- Existing Endpoints (Modified or Kept for Compatibility) ---

@app.post("/process", response_model=schemas.ProcessResponse, deprecated=True)
async def process_json_data_deprecated(request_data: schemas.ProcessRequest, background_tasks: BackgroundTasks):
    """
    Deprecated synchronous processing endpoint.
    Users should migrate to /process-async and related job endpoints.
    This endpoint now creates a job and waits for its completion for compatibility.
    """
    logger.warning("Deprecated /process endpoint used. Consider migrating to /process-async.")

    job_manager: JobManager = app_state["JOB_MANAGER"]
    job_id = await job_manager.create_job(name=f"legacy_process_{request_data.file_name or 'untitled'}")

    # Store request data or pass it to the job processing logic if needed by services.py
    # For now, assuming services.py can be adapted or JobManager handles it.
    # This might involve adding a payload to the Job model or a separate store.

    # Simulate putting the actual processing task into the job's execution path
    # This requires adapting services.convert_with_agno_async and direct_json_to_excel_async
    # to be callable by the JobManager's workers, receive job_id, and update job status.

    # For now, we'll just return a placeholder or error, as the full refactor of services.py
    # is part of a later step. This endpoint is mainly for API compatibility.

    # Wait for job completion (simplified polling for this deprecated endpoint)
    timeout_seconds = settings.REQUEST_TIMEOUT_SECONDS # Use existing timeout
    start_time = time.monotonic()
    job: Optional[Job] = None
    while time.monotonic() - start_time < timeout_seconds:
        job = await job_manager.get_job(job_id)
        if job and job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            break
        await asyncio.sleep(1) # Poll interval

    if not job or job.status not in [JobStatus.COMPLETED, JobStatus.FAILED]:
        await job_manager.cancel_job(job_id) # Cancel if timed out
        raise HTTPException(status_code=504, detail="Processing timed out (via deprecated /process endpoint).")

    if job.status == JobStatus.FAILED:
        error_detail = "Job failed."
        if job.result and isinstance(job.result, dict) and "error" in job.result:
            error_detail = f"Job failed: {job.result['error']}"
        raise HTTPException(status_code=500, detail=error_detail)

    if job.status == JobStatus.CANCELLED:
        raise HTTPException(status_code=499, detail="Job was cancelled (via deprecated /process endpoint).")

    # Assuming job.result contains necessary info like file_id, file_name, etc.
    # This part needs careful mapping from Job.result to schemas.ProcessResponse
    if job.result and isinstance(job.result, dict):
        # This is a placeholder. The actual structure of job.result needs to be defined
        # by the functions in services.py when they are adapted for the JobManager.
        # For example, job.result might be:
        # {"file_id": "some_uuid", "file_name": "output.xlsx", "download_url": "/download/some_uuid", ...}

        # Store file if job result indicates a file was produced (for /download endpoint)
        if "file_path" in job.result and "file_id" in job.result and "filename" in job.result:
            async with app_state["LOCK"]:
                app_state["TEMP_FILES"][job.result["file_id"]] = {
                    'path': job.result["file_path"],
                    'filename': job.result["filename"]
                }

        return schemas.ProcessResponse(
            success=True,
            file_id=job.result.get("file_id"),
            file_name=job.result.get("file_name"),
            download_url=job.result.get("download_url"), # This should come from job result
            ai_analysis=job.result.get("ai_analysis"),
            processing_method=job.result.get("processing_method", "unknown_async"),
            processing_time=job.result.get("processing_time", time.monotonic() - start_time),
            data_size=len(request_data.json_data) if request_data.json_data else 0, # Or get from job
            user_id=request_data.user_id, # Or get from job
            session_id=request_data.session_id # Or get from job
        )
    else:
        raise HTTPException(status_code=500, detail="Job completed but result format is unexpected.")


# --- New Asynchronous Job Endpoints ---

@app.post("/process-async", response_model=schemas.JobCreationResponse, status_code=202)
async def process_data_asynchronously(request_data: schemas.ProcessRequest):
    """
    Submits data for asynchronous processing. Returns a job ID immediately.
    """
    job_manager: JobManager = app_state["JOB_MANAGER"]

    # Here, you would pass request_data (or relevant parts) to the job
    # The job processing logic in services.py would then use this data.
    # For now, the Job model in job_manager.py doesn't store the input payload directly.
    # This might need adjustment (e.g., Job model gets a `payload: Optional[dict]`).
    # Or, the job creation step could trigger a specific type of job if you have multiple.
    # For now, we pass the file_name as the job name.
    job_name = f"process_{request_data.processing_mode}_{request_data.file_name or 'untitled'}"

    # The actual processing (AI or direct) will be determined by the worker
    # based on the request_data. This means the worker needs access to request_data.
    # This is a key part of adapting services.py.
    # For now, the JobManager's _process_job_placeholder is generic.
    # We'll need to pass `request_data` to `create_job` or have the worker fetch it.
    # Let's assume `create_job` can take a payload for now.
    # Modifying Job and JobManager:
    # In Job class: `payload: Optional[schemas.ProcessRequest] = None`
    # In JobManager.create_job: `async def create_job(self, name: Optional[str] = None, payload: Optional[schemas.ProcessRequest] = None) -> str:`
    #   `job = Job(name=name, payload=payload)`
    # Then, in `_worker`, `job.payload` would be available.
    # This change is NOT made here yet, but is noted for the services.py adaptation.

    job_id = await job_manager.create_job(name=job_name) # Potentially add payload=request_data here later

    async with app_state["LOCK"]:
        app_state["METRICS"]['jobs_created'] = app_state["METRICS"].get('jobs_created', 0) + 1

    return schemas.JobCreationResponse(
        job_id=job_id,
        status_url=f"/job/{job_id}/status",
        stream_url=f"/job/{job_id}/stream",
        websocket_url=f"/job/{job_id}/ws"
    )

@app.get("/job/{job_id}/status", response_model=schemas.JobStatusResponse)
async def get_job_status(job_id: str):
    """Checks the status, progress, and current step of a job."""
    job_manager: JobManager = app_state["JOB_MANAGER"]
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    # If job is completed and result contains file info, ensure it's in TEMP_FILES for /download
    if job.status == JobStatus.COMPLETED and job.result and isinstance(job.result, dict):
        if "file_path" in job.result and "file_id" in job.result and "filename" in job.result:
            # Ensure file_id in result matches job_id or is handled consistently
            # For now, assume job.result["file_id"] is the one to use for downloading.
            file_key_for_download = job.result["file_id"]
            async with app_state["LOCK"]:
                if file_key_for_download not in app_state["TEMP_FILES"]:
                     app_state["TEMP_FILES"][file_key_for_download] = {
                        'path': job.result["file_path"],
                        'filename': job.result["filename"]
                    }

    return schemas.JobStatusResponse(
        id=job.id,
        name=job.name,
        status=job.status,
        progress=job.progress,
        current_step=job.current_step,
        created_at=job.created_at,
        updated_at=job.updated_at,
        result=job.result # This will include download_url if set by the job
    )

@app.get("/job/{job_id}/stream")
async def stream_job_updates(job_id: str, request: Request):
    """Streams job updates using Server-Sent Events (SSE)."""
    job_manager: JobManager = app_state["JOB_MANAGER"]
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found for SSE.")

    queue = await sse_manager.connect(job_id)

    async def event_generator():
        try:
            # Send current state immediately
            initial_message = (
                f"id: {job.updated_at.timestamp()}\n"
                f"event: job_update\n"
                f"data: {{\"job_id\": \"{job.id}\", \"status\": \"{job.status.value}\", "
                f"\"progress\": {job.progress}, \"current_step\": \"{job.current_step or ''}\", "
                f"\"result\": {services.json_serialize_job_result(job.result)}, "
                f"\"updated_at\": \"{job.updated_at.isoformat()}\"}}\n\n"
            )
            yield initial_message
            
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    logger.info(f"SSE client for job {job_id} disconnected.")
                    break
                
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=settings.SSE_TIMEOUT_SECONDS)
                    yield message
                    queue.task_done()
                except asyncio.TimeoutError:
                    # Send a keep-alive comment or heartbeat if needed
                    yield ": KEEPALIVE\n\n"

                # Check if job is in a terminal state
                current_job_state = await job_manager.get_job(job_id) # Get fresh state
                if current_job_state and current_job_state.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    logger.info(f"Job {job_id} reached terminal state ({current_job_state.status}). Closing SSE stream.")
                    break
        except asyncio.CancelledError:
            logger.info(f"SSE stream for job {job_id} cancelled by server.")
        finally:
            sse_manager.disconnect(queue, job_id)
            logger.info(f"SSE event_generator for job {job_id} finished.")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.websocket("/job/{job_id}/ws")
async def websocket_job_updates(websocket: WebSocket, job_id: str):
    """Provides real-time job updates over WebSocket."""
    job_manager: JobManager = app_state["JOB_MANAGER"]
    job = await job_manager.get_job(job_id)
    if not job:
        await websocket.close(code=4004, reason="Job not found") # Using custom code for clarity
        return

    await ws_manager.connect(websocket, job_id)
    
    # Send current state immediately
    try:
        await websocket.send_json({
            "job_id": job.id, "status": job.status.value, "progress": job.progress,
            "current_step": job.current_step, "result": job.result,
            "updated_at": job.updated_at.isoformat()
        })
    except WebSocketDisconnect: # Client might disconnect immediately after connect
        ws_manager.disconnect(websocket, job_id)
        logger.warning(f"WebSocket for job {job_id} disconnected before initial state send.")
        return
    except Exception as e: # Catch other potential send errors
        logger.error(f"Error sending initial WebSocket state for job {job_id}: {e}")
        ws_manager.disconnect(websocket, job_id)
        await websocket.close(code=1011) # Internal server error
        return


    try:
        while True:
            # WebSockets are primarily for server-to-client updates here.
            # If client-to-server messages are needed (e.g., a specific cancel message over WS),
            # handle them with `await websocket.receive_text()` or `receive_json()`.
            # For now, we just keep the connection alive and rely on broadcasts.

            # Keep connection alive, or handle incoming messages if any are expected.
            # A simple way to detect disconnect is to try receiving data with a timeout.
            try:
                # This is a way to pause and allow other tasks to run,
                # and also to periodically check the connection state.
                # If client sends data, it would be caught here.
                data = await asyncio.wait_for(websocket.receive_text(), timeout=settings.WEBSOCKET_KEEP ALIVE_TIMEOUT_SECONDS)
                # If you expect client messages, process `data` here.
                # e.g., if data == "CANCEL", then await job_manager.cancel_job(job_id)
                logger.info(f"WebSocket for job {job_id} received message: {data}")

            except asyncio.TimeoutError:
                # No message from client, connection is likely still alive.
                # Server can send a ping if needed: await websocket.send_text("ping")
                pass
            except WebSocketDisconnect:
                logger.info(f"WebSocket for job {job_id} disconnected (detected during receive).")
                break # Exit loop on disconnect

            # Check if job is in a terminal state to close connection gracefully from server side
            current_job_state = await job_manager.get_job(job_id) # Get fresh state
            if current_job_state and current_job_state.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                logger.info(f"Job {job_id} reached terminal state ({current_job_state.status}). Closing WebSocket from server.")
                break


    except WebSocketDisconnect: # Handles cases where client disconnects abruptly
        logger.info(f"WebSocket for job {job_id} disconnected.")
    except Exception as e:
        logger.error(f"Error in WebSocket handler for job {job_id}: {e}")
    finally:
        ws_manager.disconnect(websocket, job_id)
        logger.info(f"WebSocket connection for job {job_id} closed.")

        # Auto-cancel abandoned job logic
        job_manager: JobManager = app_state["JOB_MANAGER"]
        job = await job_manager.get_job(job_id)

        if job and job.status == JobStatus.PROCESSING:
            # Check if there are any other active WebSocket connections for this job
            if not ws_manager.active_connections.get(job_id):
                logger.info(f"No active WebSocket connections for job {job_id}. Scheduling potential cancellation.")
                
                async def schedule_abandoned_job_cancellation(target_job_id: str):
                    await asyncio.sleep(settings.ABANDONED_JOB_TIMEOUT_SECONDS) # Default 60 seconds
                    
                    # Re-fetch job and check status and connections again before cancelling
                    current_job = await job_manager.get_job(target_job_id)
                    if not ws_manager.active_connections.get(target_job_id) and \
                       current_job and current_job.status == JobStatus.PROCESSING:
                        logger.warning(f"Job {target_job_id} considered abandoned. Auto-cancelling.")
                        await job_manager.cancel_job(target_job_id)
                        await job_manager.update_job_status(target_job_id, current_job.status, current_step="Auto-cancelled due to client disconnect")
                    elif current_job:
                        logger.info(f"Abandoned job {target_job_id} cancellation check: Not cancelling. Current status: {current_job.status}, WS connections: {len(ws_manager.active_connections.get(target_job_id, []))}")

                # Run this check in the background so it doesn't block the finally clause
                asyncio.create_task(schedule_abandoned_job_cancellation(job_id))
            else:
                logger.info(f"Job {job_id} still has other active WebSocket connections. Not scheduling auto-cancellation.")


@app.post("/job/{job_id}/cancel", response_model=schemas.JobStatusResponse)
async def cancel_processing_job(job_id: str):
    """Requests cancellation of an ongoing job."""
    job_manager: JobManager = app_state["JOB_MANAGER"]
    success = await job_manager.cancel_job(job_id)

    job = await job_manager.get_job(job_id) # Get updated job status
    if not job:
        # This case should ideally not happen if cancel_job was called on an existing job_id
        raise HTTPException(status_code=404, detail="Job not found after cancellation attempt.")

    if not success and job.status not in [JobStatus.CANCELLED]:
        # If cancel_job returned False, it means job was already terminal or not cancellable
        # However, the job object itself might have a status like COMPLETED.
        # We should reflect the actual current state.
        detail_message = f"Job {job_id} could not be cancelled. Current status: {job.status.value}."
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
             detail_message = f"Job {job_id} has already finished with status: {job.status.value}."

        raise HTTPException(status_code=409, detail=detail_message) # 409 Conflict

    return schemas.JobStatusResponse(
        id=job.id, name=job.name, status=job.status, progress=job.progress,
        current_step=job.current_step, created_at=job.created_at,
        updated_at=job.updated_at, result=job.result
    )


# --- Other Endpoints (Kept or Slightly Modified) ---

@app.get("/download/{file_id}")
async def download_file(file_id: str):
    """Downloads a file produced by a job or direct processing."""
    # This endpoint now needs to primarily serve files whose info is stored
    # when a job completes and produces a file. The TEMP_FILES dict is populated
    # by the /job/{job_id}/status endpoint or the deprecated /process endpoint
    # when a job result includes file information.

    async with app_state["LOCK"]: # Use asyncio.Lock
        file_info = app_state["TEMP_FILES"].get(file_id)
    
    if not file_info or not os.path.exists(file_info['path']):
        # Attempt to get from job result directly if job exists and completed
        job_manager: JobManager = app_state["JOB_MANAGER"]
        # Assuming file_id might also be a job_id if the file is the primary result of that job.
        # This logic might need refinement based on how file_ids are generated and associated with jobs.
        job = await job_manager.get_job(file_id) # Try file_id as job_id
        if job and job.status == JobStatus.COMPLETED and isinstance(job.result, dict) and \
           "file_path" in job.result and "filename" in job.result and \
           os.path.exists(job.result["file_path"]):

            file_info = {'path': job.result["file_path"], 'filename': job.result["filename"]}
        else: # If not found via job result either
            raise HTTPException(status_code=404, detail="File not found, has expired, or job did not produce this file.")
        
    return FileResponse(
        path=file_info['path'],
        filename=file_info['filename'],
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


@app.get("/metrics", response_model=schemas.SystemMetrics)
async def get_system_metrics():
    async with app_state["LOCK"]: # Use asyncio.Lock
        metrics = app_state["METRICS"].copy()
        active_files = len(app_state["TEMP_FILES"]) # This is less relevant if files are tied to jobs
        job_manager: JobManager = app_state["JOB_MANAGER"]
        active_jobs = len(job_manager.jobs)
        pending_jobs = job_manager.job_queue.qsize()

    success_rate = (metrics.get('successful_conversions',0) / max(metrics.get('total_requests',1), 1)) * 100

    # Add job-specific metrics
    metrics['active_jobs'] = active_jobs
    metrics['pending_jobs_in_queue'] = pending_jobs
    metrics['jobs_created_total'] = metrics.get('jobs_created', 0)

    return schemas.SystemMetrics(
        total_requests=metrics.get('total_requests',0), # Or 'tasks_processed'
        successful_conversions=metrics.get('successful_conversions',0),
        ai_conversions=metrics.get('ai_conversions',0),
        direct_conversions=metrics.get('direct_conversions',0),
        failed_conversions=metrics.get('failed_conversions',0),
        success_rate=round(success_rate, 2),
        average_processing_time=round(metrics.get('average_processing_time',0.0), 2),
        active_files=active_files, # May deprecate or change meaning
        temp_directory=app_state["TEMP_DIR"],
        # New metrics from JobManager
        active_jobs=metrics.get('active_jobs', 0),
        pending_jobs_in_queue=metrics.get('pending_jobs_in_queue', 0),
        jobs_created_total=metrics.get('jobs_created_total', 0),
        job_worker_count=len(job_manager.workers) if hasattr(job_manager, 'workers') else 0
    )


@app.get("/")
async def root():
    return {"message": "Welcome to the IntelliExtract Agno AI API. Asynchronous processing available. See /docs for more info."}


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    job_manager: JobManager = app_state["JOB_MANAGER"]
    health_status = {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "temp_directory_exists": os.path.exists(app_state.get("TEMP_DIR", "")),
        "agent_pool_size": len(services.AGENT_POOL), # Assuming services.AGENT_POOL is still relevant
        "active_files_legacy": len(app_state.get("TEMP_FILES", {})), # Legacy file count
        "job_manager_status": {
            "total_jobs": len(job_manager.jobs),
            "pending_in_queue": job_manager.job_queue.qsize(),
            "workers_running": len(job_manager.workers),
            "worker_errors": "N/A" # Placeholder, could add error tracking in JobManager
        }
    }

    health_status["google_api_configured"] = bool(settings.GOOGLE_API_KEY)

    try:
        test_file_path = os.path.join(app_state.get("TEMP_DIR", "/tmp"), f"health_check_{uuid.uuid4()}.tmp")
        with open(test_file_path, "w") as f:
            f.write("test")
        os.remove(test_file_path)
        health_status["temp_directory_writable"] = True
    except Exception as e:
        logger.error(f"Health check: Temp directory not writable: {e}")
        health_status["temp_directory_writable"] = False
        health_status["status"] = "degraded"
        health_status["temp_directory_error"] = str(e)

    # Check worker health (basic check: are they alive?)
    if not job_manager.workers or any(w.done() and w.exception() for w in job_manager.workers):
        health_status["status"] = "degraded"
        health_status["job_manager_status"]["worker_status"] = "One or more workers are not running properly."
        logger.error("Health check: One or more job workers are not running properly.")


    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)


# --- Model Management Endpoint ---
@app.get("/api/models", response_model=List[schemas.ModelInfo]) # Assuming ModelInfo schema will be created
async def get_available_models():
    """
    Lists available AI models, primarily from Google Generative AI,
    with caching and fallback to a hardcoded list.
    Includes pricing information URL.
    """
    try:
        # This service function will handle caching and fallback internally
        models = await services.list_available_models()
        return models
    except Exception as e:
        logger.error(f"Error fetching available models: {e}", exc_info=True)
        # Fallback to a minimal list or error response if services.list_available_models itself fails severely
        # However, list_available_models is designed to have its own fallback.
        # If it still raises an exception, it's a more critical issue.
        raise HTTPException(status_code=500, detail=f"Could not retrieve model list: {str(e)}")
async def get_system_metrics():
    with app_state["LOCK"]:
        metrics = app_state["METRICS"].copy()
        active_files = len(app_state["TEMP_FILES"])

    success_rate = (metrics['successful_conversions'] / max(metrics['total_requests'], 1)) * 100
    
    return schemas.SystemMetrics(
        total_requests=metrics['total_requests'],
        successful_conversions=metrics['successful_conversions'],
        ai_conversions=metrics['ai_conversions'],
        direct_conversions=metrics['direct_conversions'],
        failed_conversions=metrics['failed_conversions'],
        success_rate=round(success_rate, 2),
        average_processing_time=round(metrics['average_processing_time'], 2),
        active_files=active_files,
        temp_directory=app_state["TEMP_DIR"]
    )


@app.get("/")
async def root():
    return {"message": "Welcome to the IntelliExtract Agno AI API for Financial Document Processing. See /docs for more info."}


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    health_status = {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "temp_directory_exists": os.path.exists(app_state.get("TEMP_DIR", "")),
        "agent_pool_size": len(services.AGENT_POOL),
        "active_files": len(app_state.get("TEMP_FILES", {})),
    }
    
    # Check Google API key availability
    health_status["google_api_configured"] = bool(settings.GOOGLE_API_KEY)
    
    # Check temp directory is writable
    try:
        test_file = os.path.join(app_state.get("TEMP_DIR", "/tmp"), "health_check.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        health_status["temp_directory_writable"] = True
    except:
        health_status["temp_directory_writable"] = False
        health_status["status"] = "degraded"
    
    # Return appropriate status code
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)