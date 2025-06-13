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
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool

from . import schemas, services
from .core.config import settings

# --- Application State and Logging ---
app_state = {}

logging.basicConfig(level=settings.LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Lifespan Management (Startup and Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    # Startup
    app_state["TEMP_DIR"] = tempfile.mkdtemp(prefix=settings.TEMP_DIR_PREFIX)
    app_state["TEMP_FILES"] = {}
    app_state["METRICS"] = {
        'total_requests': 0, 'successful_conversions': 0, 'ai_conversions': 0,
        'direct_conversions': 0, 'failed_conversions': 0,
        'average_processing_time': 0.0
    }
    app_state["LOCK"] = threading.Lock()
    logger.info(f"Application startup complete. Temp directory: {app_state['TEMP_DIR']}")
    
    yield  # Application is now running
    
    # Shutdown
    logger.info("Application shutdown. Cleaning up temp directory...")
    # Clean up agent pool to free memory
    services.cleanup_agent_pool()
    shutil.rmtree(app_state["TEMP_DIR"])
    logger.info("Cleanup complete.")

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# --- Utility Functions ---
def update_metrics(processing_time: float, method: str, success: bool):
    with app_state["LOCK"]:
        metrics = app_state["METRICS"]
        metrics['total_requests'] += 1
        if success:
            metrics['successful_conversions'] += 1
            if method == 'ai':
                metrics['ai_conversions'] += 1
            else:
                metrics['direct_conversions'] += 1
            
            total_time = metrics['average_processing_time'] * (metrics['successful_conversions'] - 1)
            metrics['average_processing_time'] = (total_time + processing_time) / metrics['successful_conversions']
        else:
            metrics['failed_conversions'] += 1

# --- API Endpoints ---
@app.post("/process", response_model=schemas.ProcessResponse)
async def process_json_data(request: schemas.ProcessRequest, background_tasks: BackgroundTasks):
    start_time = time.time()
    processing_method = "direct"  # Default method
    
    # Create a unique temp directory for THIS request to avoid conflicts
    request_id = str(uuid.uuid4())[:8]
    request_temp_dir = os.path.join(app_state["TEMP_DIR"], f"request_{request_id}")
    os.makedirs(request_temp_dir, exist_ok=True)
    
    logger.info(f"Created isolated temp directory for request: {request_temp_dir}")
    
    # Get all Excel files before processing in THIS request's directory
    files_before = set()
    patterns = [
        os.path.join(request_temp_dir, "*.xlsx"),
        os.path.join(request_temp_dir, "**", "*.xlsx"),
    ]
    for pattern in patterns:
        files_before.update(glob.glob(pattern, recursive=True))
    
    logger.info(f"Files before processing in {request_temp_dir}: {list(files_before)}")
    
    # Schedule cleanup of request directory and agents after processing with delay
    def cleanup_request_resources():
        try:
            # Use configurable delay from settings
            import time
            time.sleep(settings.CLEANUP_DELAY_SECONDS)  # Configurable delay before cleanup (default: 5 minutes)
            
            # Clean up temp directory
            if os.path.exists(request_temp_dir):
                shutil.rmtree(request_temp_dir)
                logger.info(f"Cleaned up request temp directory: {request_temp_dir}")
            
            # Note: Agents are now pooled by model and reused across requests
            # No per-request cleanup needed - agents persist for efficiency
            logger.debug(f"Agent pool maintained for reuse (size: {len(services.AGENT_POOL)})")
                
        except Exception as e:
            logger.warning(f"Failed to cleanup request resources for {request_temp_dir}: {e}")
    
    background_tasks.add_task(cleanup_request_resources)

    try:
        # --- AI Processing Logic ---
        if request.processing_mode in ["auto", "ai_only"]:
            try:
                # Use async version for better performance with isolated temp directory
                ai_response_content, actual_session_id = await services.convert_with_agno_async(
                    request.json_data,
                    request.file_name,
                    request.description,
                    request.model or settings.DEFAULT_AI_MODEL,  # Use default model if none specified
                    request_temp_dir,  # Use isolated directory for this request
                    request.user_id,   # Pass user ID for session management
                    request.session_id # Pass session ID for conversation continuity
                )
                
                newest_file = services.find_newest_file(request_temp_dir, files_before)
                logger.info(f"AI processing completed. Looking for newest file...")
                
                if newest_file:
                    logger.info(f"Found newest file: {newest_file}")
                    processing_method = "ai"
                    file_id = str(uuid.uuid4())
                    original_filename = os.path.basename(newest_file)
                    
                    with app_state["LOCK"]:
                        app_state["TEMP_FILES"][file_id] = {'path': newest_file, 'filename': original_filename}

                    processing_time = time.time() - start_time
                    update_metrics(processing_time, processing_method, True)
                    
                    return schemas.ProcessResponse(
                        success=True, file_id=file_id, file_name=original_filename,
                        download_url=f"/download/{file_id}", ai_analysis=ai_response_content,
                        processing_method=processing_method, processing_time=processing_time,
                        data_size=len(request.json_data),
                        user_id=request.user_id or f"user_{request_id}",  # Return user_id for session continuity
                        session_id=actual_session_id                     # Return actual session_id used
                    )
                
                logger.warning(f"AI processing completed but no new file was found. Falling back to direct conversion.")
                if request.processing_mode == "ai_only":
                    raise HTTPException(status_code=500, detail="AI processing was requested, but no file was generated.")

            except Exception as e:
                logger.warning(f"AI processing failed: {e}. Falling back to direct conversion.")
                if request.processing_mode == "ai_only":
                    raise HTTPException(status_code=500, detail=f"AI-only processing failed: {e}")

        # --- Direct Conversion (Fallback or direct_only mode) ---
        logger.info("Using direct conversion...")
        # Use async version for better performance with isolated temp directory
        file_id, xlsx_filename, file_path = await services.direct_json_to_excel_async(
            request.json_data, request.file_name, request.chunk_size, request_temp_dir
        )
        
        with app_state["LOCK"]:
            app_state["TEMP_FILES"][file_id] = {'path': file_path, 'filename': xlsx_filename}

        processing_time = time.time() - start_time
        update_metrics(processing_time, processing_method, True)

        return schemas.ProcessResponse(
            success=True, file_id=file_id, file_name=xlsx_filename,
            download_url=f"/download/{file_id}", processing_method=processing_method,
            processing_time=processing_time, data_size=len(request.json_data),
            user_id=request.user_id or f"user_{request_id}",  # Return user_id for consistency
            session_id=request.session_id or f"session_{request_id}"  # Return session_id for direct conversions
        )

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        update_metrics(time.time() - start_time, processing_method, False)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.get("/download/{file_id}")
async def download_file(file_id: str):
    with app_state["LOCK"]:
        file_info = app_state["TEMP_FILES"].get(file_id)
    
    if not file_info or not os.path.exists(file_info['path']):
        raise HTTPException(status_code=404, detail="File not found or has expired.")
        
    return FileResponse(
        path=file_info['path'],
        filename=file_info['filename'],
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


@app.get("/metrics", response_model=schemas.SystemMetrics)
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