import { useState } from 'react';
import { useToast } from '@/components/ui/use-toast'; // Assuming a toast component for notifications

interface UseJobCancellationReturn {
  isCancelling: boolean;
  cancelJob: (jobId: string) => Promise<boolean>;
}

/**
 * Custom hook to handle job cancellation.
 *
 * @returns {UseJobCancellationReturn} Object containing cancellation state and function.
 */
export const useJobCancellation = (): UseJobCancellationReturn => {
  const [isCancelling, setIsCancelling] = useState<boolean>(false);
  const { toast } = useToast();

  /**
   * Sends a request to the backend to cancel a job.
   *
   * @param {string} jobId The ID of the job to cancel.
   * @returns {Promise<boolean>} True if cancellation was successfully requested, false otherwise.
   */
  const cancelJob = async (jobId: string): Promise<boolean> => {
    if (!jobId) {
      console.error('Job ID is required to cancel a job.');
      toast({
        title: 'Error',
        description: 'Job ID is missing. Cannot cancel job.',
        variant: 'destructive',
      });
      return false;
    }

    setIsCancelling(true);
    let success = false;

    try {
      // The backend API for job cancellation is POST /job/{job_id}/cancel
      // This frontend hook calls an internal Next.js API route, which then calls the backend.
      // This is a common pattern to avoid exposing the backend URL directly to the client
      // and to handle potential CORS issues or add frontend-specific logic.
      // Let's assume the Next.js API route is /api/jobs/cancel

      const response = await fetch(`/api/jobs/cancel`, { // TODO: Create this Next.js API route
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ jobId }),
      });

      if (response.ok) {
        const data = await response.json();
        toast({
          title: 'Job Cancellation Requested',
          description: `Cancellation requested for job ${jobId}. Status: ${data.status}`,
        });
        success = true;
        // Optionally, trigger a state update or event to refresh job status elsewhere in the UI
      } else {
        const errorData = await response.json();
        console.error(`Failed to cancel job ${jobId}:`, errorData.detail || response.statusText);
        toast({
          title: 'Cancellation Failed',
          description: `Could not cancel job ${jobId}: ${errorData.detail || response.statusText}`,
          variant: 'destructive',
        });
      }
    } catch (error) {
      console.error(`Error cancelling job ${jobId}:`, error);
      toast({
        title: 'Error',
        description: `An unexpected error occurred while cancelling job ${jobId}.`,
        variant: 'destructive',
      });
    } finally {
      setIsCancelling(false);
    }

    return success;
  };

  return {
    isCancelling,
    cancelJob,
  };
};
