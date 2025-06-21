import asyncio
import uuid
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List

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
    cancellation_token: Optional[Any] = None  # Will be an asyncio.Event
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    result: Optional[Any] = None
    name: Optional[str] = None # Optional: for naming specific job types

    def __init__(self, **data):
        super().__init__(**data)
        self.cancellation_token = asyncio.Event()

    class Config:
        arbitrary_types_allowed = True

class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.job_queue: asyncio.Queue[Job] = asyncio.Queue()
        self.workers: List[asyncio.Task] = []
        self._job_update_callbacks: List[callable] = []

    def register_job_update_callback(self, callback: callable):
        """Registers a callback function to be called when a job is updated."""
        self._job_update_callbacks.append(callback)

    async def _notify_job_update(self, job: Job):
        """Notifies all registered callbacks about a job update."""
        for callback in self._job_update_callbacks:
            try:
                await callback(job)
            except Exception as e:
                # Log error, but don't let one callback failure stop others
                print(f"Error in job update callback: {e}")


    async def create_job(self, name: Optional[str] = None) -> str:
        job = Job(name=name)
        job.cancellation_token = asyncio.Event() # Ensure event is created
        self.jobs[job.id] = job
        await self.job_queue.put(job)
        await self._notify_job_update(job)
        print(f"Job created: {job.id}, Name: {name}, Status: {job.status}")
        return job.id

    async def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        progress: Optional[float] = None,
        current_step: Optional[str] = None,
        result: Optional[Any] = None,
    ):
        job = await self.get_job(job_id)
        if job:
            job.status = status
            if progress is not None:
                job.progress = progress
            if current_step is not None:
                job.current_step = current_step
            if result is not None:
                job.result = result
            job.updated_at = datetime.utcnow()
            self.jobs[job.id] = job # Re-assign to ensure update if Job is immutable in some contexts
            await self._notify_job_update(job)
            print(f"Job {job_id} updated: Status={status}, Progress={progress}, Step='{current_step}'")
        else:
            print(f"Job {job_id} not found for update.")


    async def cancel_job(self, job_id: str) -> bool:
        job = await self.get_job(job_id)
        if job and job.status not in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            if job.cancellation_token: # Check if it's not None
                 job.cancellation_token.set() # type: ignore
            await self.update_job_status(job_id, JobStatus.CANCELLED, current_step="Cancellation requested")
            # Add any resource cleanup logic here if necessary
            print(f"Job {job_id} cancellation requested.")
            return True
        elif job:
            print(f"Job {job_id} is already in a terminal state: {job.status}")
            return False
        else:
            print(f"Job {job_id} not found for cancellation.")
            return False

    async def _process_job_placeholder(self, job: Job):
        """
        Placeholder for actual job processing logic.
        This should be replaced by actual task execution.
        """
        print(f"Starting processing for job {job.id} ({job.name})")
        await self.update_job_status(job.id, JobStatus.PROCESSING, progress=0.1, current_step="Initializing")

        steps = ["Step 1: Data validation", "Step 2: Preprocessing", "Step 3: Core processing", "Step 4: Postprocessing", "Step 5: Finalizing"]
        total_steps = len(steps)

        for i, step_name in enumerate(steps):
            if job.cancellation_token and job.cancellation_token.is_set(): # type: ignore
                await self.update_job_status(job.id, JobStatus.CANCELLED, current_step="Processing cancelled by user")
                print(f"Job {job.id} cancelled during {step_name}")
                return

            await self.update_job_status(job.id, JobStatus.PROCESSING, progress=((i + 1) / total_steps) * 0.9, current_step=step_name)
            await asyncio.sleep(2) # Simulate work for this step

            if job.cancellation_token and job.cancellation_token.is_set(): # type: ignore
                await self.update_job_status(job.id, JobStatus.CANCELLED, current_step="Processing cancelled by user")
                print(f"Job {job.id} cancelled after {step_name}")
                return

        # Simulate a chance of failure for demonstration
        # import random
        # if random.random() < 0.1: # 10% chance of failure
        #     await self.update_job_status(job.id, JobStatus.FAILED, progress=0.95, current_step="Critical error during finalization", result={"error": "Simulated random failure"})
        #     print(f"Job {job.id} failed.")
        #     return

        await self.update_job_status(job.id, JobStatus.COMPLETED, progress=1.0, current_step="Processing complete", result={"message": "Job successfully processed", "output_file": f"/path/to/output_for_{job.id}.xlsx"})
        print(f"Job {job.id} completed.")


    async def _worker(self):
        print("Worker started...")
        while True:
            try:
                job = await self.job_queue.get()
                print(f"Worker picked up job: {job.id}")
                if job.status == JobStatus.PENDING:
                    # This is where you'd call your actual processing function
                    # For now, we use the placeholder
                    await self._process_job_placeholder(job)
                self.job_queue.task_done()
                print(f"Worker finished job: {job.id}")
            except asyncio.CancelledError:
                print("Worker cancelled.")
                break
            except Exception as e:
                print(f"Error in worker processing job {job.id if 'job' in locals() else 'unknown'}: {e}")
                if 'job' in locals() and job: # Check if job is defined
                    try:
                        await self.update_job_status(job.id, JobStatus.FAILED, current_step=f"Worker error: {str(e)}", result={"error": str(e)})
                    except Exception as ue: # ue for update error
                        print(f"Failed to update job status to FAILED for job {job.id}: {ue}")
                # Continue to the next job, or handle as per retry policy
                if self.job_queue.empty(): # Avoid busy loop if queue is empty and error persists
                    await asyncio.sleep(1)


    def start_workers(self, num_workers: int = 2):
        if not self.workers: # Only start if not already started
            self.workers = [asyncio.create_task(self._worker()) for _ in range(num_workers)]
            print(f"Started {num_workers} workers.")
        else:
            print("Workers already started.")

    async def stop_workers(self):
        print("Stopping workers...")
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers = []
        print("All workers stopped.")

    async def get_job_summary(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = await self.get_job(job_id)
        if job:
            return {
                "id": job.id,
                "name": job.name,
                "status": job.status.value,
                "progress": job.progress,
                "current_step": job.current_step,
                "created_at": job.created_at.isoformat(),
                "updated_at": job.updated_at.isoformat(),
            }
        return None

# Example usage (for testing purposes, not part of the class)
async def main_test():
    job_manager = JobManager()
    job_manager.start_workers(2)

    # Simulate creating a few jobs
    job_id1 = await job_manager.create_job(name="Document Conversion Task")
    job_id2 = await job_manager.create_job(name="Data Analysis Report")

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
