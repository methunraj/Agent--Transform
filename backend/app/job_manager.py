import asyncio
import uuid
import json # For serializing Pydantic models to/from Redis
from enum import Enum
from datetime import datetime, timezone # Ensure timezone awareness for consistency
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Any, Dict, List

import aioredis
from .core.config import settings # Import app settings for Redis config
from . import schemas # For payload typing, e.g. schemas.ProcessRequest


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Job(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    current_step: Optional[str] = None
    # cancellation_token: Optional[Any] = None -> This will be managed differently with Redis
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    result: Optional[Dict[str, Any]] = None # Store results as dict for easier JSON serialization
    name: Optional[str] = None
    payload: Optional[schemas.ProcessRequest] = None # To store the input data for the job

    # For Pydantic v2, model_dump replaces dict(), and model_parse_raw replaces parse_raw_as()
    # For Pydantic v1, use .dict() and parse_obj_as() or parse_raw_as()

    @field_validator('created_at', 'updated_at', mode='before')
    def ensure_datetime_utc(cls, v):
        if isinstance(v, str):
            dt = datetime.fromisoformat(v.replace('Z', '+00:00'))
            return dt.astimezone(timezone.utc) if dt.tzinfo is None else dt
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    class Config:
        arbitrary_types_allowed = True # Keep for asyncio.Event if used locally by workers
        json_encoders = {
            datetime: lambda v: v.isoformat(), # Ensure datetimes are ISO formatted in JSON
        }


class JobManager:
    def __init__(self):
        self.redis_client: Optional[aioredis.Redis] = None
        # self.jobs: Dict[str, Job] = {} # No longer primary store; Redis is. Can be a local cache if needed.
        # self.job_queue: asyncio.Queue[Job] = asyncio.Queue() # Replaced by Redis list
        self.workers: List[asyncio.Task] = []
        self._job_update_callbacks: List[callable] = []
        self._local_cancellation_tokens: Dict[str, asyncio.Event] = {} # For jobs processed by this instance

    async def connect_redis(self):
        """Establishes connection to Redis."""
        if not self.redis_client:
            try:
                self.redis_client = await aioredis.from_url(
                    f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
                    password=settings.REDIS_PASSWORD,
                    encoding="utf-8",
                    decode_responses=True # Decode responses from bytes to string
                )
                await self.redis_client.ping() # Verify connection
                print("Successfully connected to Redis.")
            except Exception as e:
                print(f"Failed to connect to Redis: {e}")
                self.redis_client = None # Ensure client is None if connection fails
                raise # Re-raise exception to be handled by lifespan or caller

    async def close_redis(self):
        """Closes Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
            print("Redis connection closed.")

    def register_job_update_callback(self, callback: callable):
        """Registers a callback function to be called when a job is updated (for local instance)."""
        self._job_update_callbacks.append(callback)

    async def _notify_job_update(self, job: Job):
        """Notifies all registered callbacks about a job update (for local instance)."""
        if not job: return
        for callback in self._job_update_callbacks:
            try:
                # Ensure callbacks are awaited if they are coroutines
                if asyncio.iscoroutinefunction(callback):
                    await callback(job)
                else:
                    callback(job) # Should not be used for async callbacks
            except Exception as e:
                print(f"Error in job update callback for job {job.id}: {e}")

    def _get_redis_job_key(self, job_id: str) -> str:
        return f"{settings.REDIS_JOB_KEY_PREFIX}{job_id}"

    async def create_job(self, name: Optional[str] = None, payload: Optional[schemas.ProcessRequest] = None) -> str:
        if not self.redis_client:
            raise ConnectionError("Redis client not connected. Call connect_redis() first.")

        job = Job(name=name, payload=payload) # cancellation_token is not part of Job model anymore
        job_json = job.model_dump_json() # Pydantic v2
        # For Pydantic v1: job_json = job.json()

        redis_key = self._get_redis_job_key(job.id)

        # Use a pipeline for atomic operations
        async with self.redis_client.pipeline(transaction=True) as pipe:
            pipe.set(redis_key, job_json, ex=settings.REDIS_JOB_TTL_SECONDS)
            pipe.lpush(settings.REDIS_JOB_QUEUE_KEY, job.id)
            await pipe.execute()

        await self._notify_job_update(job) # Notify local listeners
        print(f"Job created in Redis: {job.id}, Name: {name}, Status: {job.status}")
        return job.id

    async def get_job(self, job_id: str) -> Optional[Job]:
        if not self.redis_client:
            # Fallback or raise error if Redis is critical
            print("Warning: Redis client not connected in get_job.")
            return None

        redis_key = self._get_redis_job_key(job_id)
        job_json = await self.redis_client.get(redis_key)

        if job_json:
            try:
                job_data = json.loads(job_json) # Standard json.loads
                # For Pydantic v1: job = Job.parse_obj(job_data)
                # For Pydantic v2, model_validate is preferred over parse_obj for dicts
                job = Job.model_validate(job_data) # Pydantic v2
                return job
            except (json.JSONDecodeError, TypeError) as e: # Catch PydanticValidationError too if using parse_obj
                print(f"Error deserializing job {job_id} from Redis: {e}. Data: {job_json[:200]}")
                return None
        return None

    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        progress: Optional[float] = None,
        current_step: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None, # Ensure result is a dict
    ):
        if not self.redis_client:
            print(f"Warning: Redis client not connected in update_job_status for job {job_id}.")
            # Optionally, could try to update a local cache if one were maintained, but would diverge from Redis.
            return

        job = await self.get_job(job_id)
        if job:
            job.status = status
            if progress is not None: job.progress = progress
            if current_step is not None: job.current_step = current_step
            if result is not None: job.result = result # result is now Dict[str, Any]
            job.updated_at = datetime.now(timezone.utc)

            job_json = job.model_dump_json() # Pydantic v2
            # For Pydantic v1: job_json = job.json()
            redis_key = self._get_redis_job_key(job_id)

            # Preserve TTL when updating:
            # Use SET with KEEPTTL option if available with aioredis and Redis server version
            # Or, get current TTL and re-apply EXPIRE. Simpler for now is just SETEX if full update.
            # If only a few fields change, consider HSET for efficiency.
            # For simplicity of Job model as single JSON, SET with EX is fine.
            await self.redis_client.set(redis_key, job_json, ex=settings.REDIS_JOB_TTL_SECONDS)

            await self._notify_job_update(job)
            print(f"Job {job_id} updated in Redis: Status={status}, Progress={progress}, Step='{current_step}'")
        else:
            print(f"Job {job_id} not found in Redis for update.")

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancels a job. For multi-instance workers, true cancellation requires
        a Pub/Sub mechanism or a shared cancellation flag in Redis that workers check.
        This implementation sets status to CANCELLED in Redis and manages a local asyncio.Event.
        """
        job = await self.get_job(job_id)
        if job and job.status not in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            # Set local cancellation event for workers on this instance
            if job_id in self._local_cancellation_tokens:
                self._local_cancellation_tokens[job_id].set()
            else: # If job processed by another instance, create and set (won't affect other instance directly)
                event = asyncio.Event()
                event.set()
                self._local_cancellation_tokens[job_id] = event

            await self.update_job_status(job_id, JobStatus.CANCELLED, current_step="Cancellation requested by user")
            print(f"Job {job_id} cancellation requested and status updated in Redis.")
            # TODO: Implement Pub/Sub for multi-instance cancellation signal if needed.
            return True
        elif job:
            print(f"Job {job_id} is already in a terminal state: {job.status}")
            return False
        else:
            print(f"Job {job_id} not found for cancellation.")
            return False

    async def _get_local_cancellation_event(self, job_id: str) -> asyncio.Event:
        """Gets or creates a local asyncio.Event for job cancellation signalling for this instance."""
        if job_id not in self._local_cancellation_tokens:
            self._local_cancellation_tokens[job_id] = asyncio.Event()
        return self._local_cancellation_tokens[job_id]

    async def _remove_local_cancellation_event(self, job_id: str):
        if job_id in self._local_cancellation_tokens:
            del self._local_cancellation_tokens[job_id]

    async def _process_job_placeholder(self, job: Job):
        """
        Placeholder for actual job processing logic.
        This should be replaced by actual task execution.
        It now needs to use the local cancellation event.
        """
        print(f"Starting processing for job {job.id} ({job.name}) via placeholder")
        await self.update_job_status(job.id, JobStatus.PROCESSING, progress=0.1, current_step="Initializing Placeholder")

        cancellation_event = await self._get_local_cancellation_event(job.id)

        steps = ["Step 1: Validation", "Step 2: Preprocessing", "Step 3: Core Processing", "Step 4: Postprocessing", "Step 5: Finalizing"]
        total_steps = len(steps)

        for i, step_name in enumerate(steps):
            if cancellation_event.is_set():
                await self.update_job_status(job.id, JobStatus.CANCELLED, current_step="Processing cancelled by user (placeholder)")
                print(f"Job {job.id} (placeholder) cancelled during {step_name}")
                return

            await self.update_job_status(job.id, JobStatus.PROCESSING, progress=((i + 1) / total_steps) * 0.9, current_step=step_name + " (Placeholder)")

            try:
                # Simulate work, but allow cancellation to interrupt sleep
                await asyncio.wait_for(asyncio.sleep(2), timeout=2.1)
            except asyncio.TimeoutError:
                pass # Expected if sleep completed

            if cancellation_event.is_set(): # Check again after work/sleep
                await self.update_job_status(job.id, JobStatus.CANCELLED, current_step="Processing cancelled by user (placeholder)")
                print(f"Job {job.id} (placeholder) cancelled after {step_name}")
                return

        await self.update_job_status(job.id, JobStatus.COMPLETED, progress=1.0, current_step="Processing complete (placeholder)", result={"message": f"Placeholder job {job.id} successfully processed"})
        print(f"Job {job.id} (placeholder) completed.")


    async def _worker(self):
        if not self.redis_client:
            print("Worker cannot start: Redis client not connected.")
            return # Or raise an exception

        print("Worker started, waiting for jobs from Redis...")
        while True:
            try:
                # BRPOP returns a tuple (list_name, item_value) or None if timeout occurs (if timeout is set)
                # Using 0 for timeout means block indefinitely.
                # Example: ('intelliextract_job_queue', 'job_id_string')
                packed_job_id = await self.redis_client.brpop(settings.REDIS_JOB_QUEUE_KEY, timeout=0)

                if packed_job_id is None: # Should not happen with timeout=0, but defensive
                    continue

                job_id = packed_job_id[1] # Get the job_id string
                print(f"Worker picked up job ID: {job_id} from Redis queue.")

                job = await self.get_job(job_id)

                if not job:
                    print(f"Worker: Job ID {job_id} found in queue but not in Redis storage. Skipping.")
                    continue

                if job.status == JobStatus.PENDING or job.status == JobStatus.PROCESSING: # Allow re-processing if it was stuck in PROCESSING
                    # Actual processing logic will replace _process_job_placeholder
                    # It will need access to job.payload and job_id (for cancellation event)
                    # And it should handle its own exceptions, updating job status to FAILED in Redis.

                    # Create/get local cancellation event for this job
                    cancellation_event = await self._get_local_cancellation_event(job.id)
                    if cancellation_event.is_set(): # Job might have been cancelled while in queue
                        print(f"Job {job.id} was cancelled before processing could start by this worker.")
                        await self.update_job_status(job.id, JobStatus.CANCELLED, current_step="Cancelled before worker processing")
                        self._remove_local_cancellation_event(job.id) # Clean up event
                        continue

                    # TODO: Replace placeholder with actual call to processing logic
                    # e.g., await process_actual_task(job, self)
                    await self._process_job_placeholder(job)

                elif job.status == JobStatus.CANCELLED:
                    print(f"Worker: Job {job_id} was already cancelled. No action taken.")
                else: # COMPLETED or FAILED
                    print(f"Worker: Job {job_id} already in terminal state {job.status}. No action taken.")

                # Clean up local cancellation token once job is terminal or handled
                self._remove_local_cancellation_event(job.id)

            except asyncio.CancelledError:
                print("Worker task cancelled.")
                break
            except aioredis.RedisError as e:
                print(f"Worker: Redis error: {e}. Retrying connection or pausing...")
                await asyncio.sleep(5) # Wait before retrying to connect or fetch
                # Consider re-establishing Redis connection if it's a connection error.
                if self.redis_client is None or not self.redis_client.closed:
                    await self.connect_redis() # Try to reconnect
            except Exception as e:
                # This is a general catch-all for unexpected errors in the worker loop itself.
                # Errors within job processing should be handled by that logic and update job status.
                job_id_for_error = job_id if 'job_id' in locals() else 'unknown'
                print(f"Error in worker loop for job {job_id_for_error}: {e}")
                # If a job was fetched, try to mark it as failed in Redis.
                if 'job' in locals() and job and isinstance(job, Job):
                    try:
                        await self.update_job_status(job.id, JobStatus.FAILED, current_step=f"Critical worker error: {str(e)}", result={"error": str(e)})
                    except Exception as ue:
                        print(f"Failed to update job status to FAILED for job {job.id} after worker error: {ue}")
                await asyncio.sleep(1) # Prevent rapid looping on persistent error


    def start_workers(self, num_workers: int = 2):
        if not self.redis_client:
            print("Cannot start workers: Redis client not connected.")
            # Or raise an error, or attempt connection. For now, just print and return.
            return

        if not self.workers: # Only start if not already started
            self.workers = [asyncio.create_task(self._worker()) for _ in range(num_workers)]
            print(f"Started {num_workers} Redis-backed workers.")
        else:
            print("Workers already started.")

    async def stop_workers(self):
        print("Stopping workers...")
        for worker_task in self.workers:
            if not worker_task.done():
                worker_task.cancel()

        # Wait for all worker tasks to complete or be cancelled
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)

        self.workers = []
        # Clean up any remaining local cancellation tokens, though ideally jobs would clear them
        self._local_cancellation_tokens.clear()
        print("All workers stopped.")

    async def get_job_summary(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = await self.get_job(job_id) # This now fetches from Redis
        if job:
            return {
                "id": job.id,
                "name": job.name,
                "status": job.status.value if isinstance(job.status, Enum) else job.status, # Handle if status is already string
                "progress": job.progress,
                "current_step": job.current_step,
                "created_at": job.created_at.isoformat() if isinstance(job.created_at, datetime) else str(job.created_at),
                "updated_at": job.updated_at.isoformat() if isinstance(job.updated_at, datetime) else str(job.updated_at),
                # Result is not typically part of summary, but can be added if needed.
                # "result": job.result
            }
        return None

# Example usage (for testing purposes, not part of the class)
async def main_test():
    job_manager = JobManager()
    await job_manager.connect_redis() # Connect to Redis first

    if not job_manager.redis_client:
        print("Failed to connect to Redis for test. Exiting.")
        return

    job_manager.start_workers(1) # Start one worker for testing

    # Simulate creating a few jobs
    print("Creating test jobs...")
    payload_example = schemas.ProcessRequest(json_data="{\"key\": \"value\"}", file_name="test.json") # Minimal valid payload
    job_id1 = await job_manager.create_job(name="Document Conversion Task", payload=payload_example)
    job_id2 = await job_manager.create_job(name="Data Analysis Report", payload=payload_example) # Also needs payload

    print(f"Job 1 ID: {job_id1}")
    print(f"Job 2 ID: {job_id2}")

    # Let them process for a bit
    await asyncio.sleep(5)

    # Check status
    job1_status = await job_manager.get_job_summary(job_id1)
    print(f"Job 1 Status: {job1_status}")

    # Cancel job 2 if it's still processing
    job2 = await job_manager.get_job(job_id2)
    if job2 and job2.status == JobStatus.PROCESSING:
        print(f"Attempting to cancel job {job_id2}")
        await job_manager.cancel_job(job_id2)

    await asyncio.sleep(15) # Let processing complete or cancellation take effect

    job1_final_status = await job_manager.get_job_summary(job_id1)
    job2_final_status = await job_manager.get_job_summary(job_id2)

    print(f"Job 1 Final Status: {job1_final_status}")
    print(f"Job 2 Final Status: {job2_final_status}")

    await job_manager.stop_workers()

if __name__ == "__main__":
    # This is how you would run the test in an asyncio environment
    # Note: This won't run directly by executing the file if it's imported elsewhere in FastAPI
    # It's primarily for isolated testing of the JobManager.
    try:
        asyncio.run(main_test())
    except KeyboardInterrupt:
        print("Test run interrupted.")
    finally:
        print("Test run finished.")
