'use client';

import React, { useEffect } from 'react';
import { useJobProgress, JobProgressState } from '@/hooks/useJobProgress';
import { useJobCancellation } from '@/hooks/useJobCancellation'; // Assuming this path is correct
import { Progress } from '@/components/ui/progress'; // Assuming Shadcn UI progress component
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card'; // For layout

interface JobProgressDisplayProps {
  jobId: string | null; // Allow null for when no job is active
  onJobCompletion?: (jobDetails: JobProgressState) => void;
  onJobFailure?: (jobDetails: JobProgressState) => void;
  onJobCancellation?: (jobDetails: JobProgressState) => void;
}

const JobProgressDisplay: React.FC<JobProgressDisplayProps> = ({
  jobId,
  onJobCompletion,
  onJobFailure,
  onJobCancellation,
}) => {
  const { jobProgress, connect, disconnect } = useJobProgress({ jobId, autoConnect: true });
  const { cancelJob, isCancelling } = useJobCancellation();

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };

  useEffect(() => {
    if (jobProgress.status === 'COMPLETED' && onJobCompletion) {
      onJobCompletion(jobProgress);
    } else if (jobProgress.status === 'FAILED' && onJobFailure) {
      onJobFailure(jobProgress);
    } else if (jobProgress.status === 'CANCELLED' && onJobCancellation) {
      onJobCancellation(jobProgress);
    }
  }, [jobProgress.status, onJobCompletion, onJobFailure, onJobCancellation, jobProgress]);

  // Automatically disconnect when the component unmounts or jobId changes and is null
  useEffect(() => {
    return () => {
      if (jobProgress.isConnected) {
        disconnect();
      }
    };
  }, [jobProgress.isConnected, disconnect]);


  if (!jobId || !jobProgress.isConnected && jobProgress.status !== 'COMPLETED' && jobProgress.status !== 'FAILED' && jobProgress.status !== 'CANCELLED' && !jobProgress.error) {
    // Show nothing or a placeholder if no job ID or not yet connected (unless it's a terminal state or error)
    // This prevents flashing of "Initializing" when jobId is null.
    if (jobId && !jobProgress.isConnected && !jobProgress.error && jobProgress.currentStep === 'Initializing...') {
        return (
            <Card className="w-full max-w-md">
                <CardHeader>
                    <CardTitle>Job Progress</CardTitle>
                    <CardDescription>Attempting to connect to job updates for {jobId}...</CardDescription>
                </CardHeader>
            </Card>
        );
    }
    return null;
  }

  const canCancel = jobProgress.status === 'PENDING' || jobProgress.status === 'PROCESSING';

  return (
    <Card className="w-full max-w-md">
      <CardHeader>
        <CardTitle>Job Progress: {jobProgress.jobId || 'N/A'}</CardTitle>
        <CardDescription>
          Status: <span className={`font-semibold ${
            jobProgress.status === 'COMPLETED' ? 'text-green-600' :
            jobProgress.status === 'FAILED' || jobProgress.status === 'CANCELLED' ? 'text-red-600' :
            'text-blue-600'
          }`}>{jobProgress.status || 'N/A'}</span>
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <div className="flex justify-between mb-1">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              {jobProgress.currentStep || 'Waiting...'}
            </span>
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              {jobProgress.progress.toFixed(0)}%
            </span>
          </div>
          <Progress value={jobProgress.progress} className="w-full" />
        </div>

        <div className="text-sm text-gray-500 dark:text-gray-400">
          Time Elapsed: {formatTime(jobProgress.timeElapsed)}
        </div>

        {jobProgress.error && (
          <div className="text-sm text-red-600 dark:text-red-400 p-2 bg-red-50 dark:bg-red-900/30 rounded-md">
            <strong>Error:</strong> {jobProgress.error}
          </div>
        )}

        {jobProgress.status === 'COMPLETED' && jobProgress.result && (
          <div className="text-sm text-green-600 dark:text-green-400 p-2 bg-green-50 dark:bg-green-900/30 rounded-md">
            <strong>Result:</strong> {typeof jobProgress.result === 'object' ? JSON.stringify(jobProgress.result) : jobProgress.result.toString()}
            {/* You might want to display parts of the result more specifically, e.g., a download link */}
            {typeof jobProgress.result === 'object' && jobProgress.result.download_url && (
                 <p>Download: <a href={jobProgress.result.download_url} target="_blank" rel="noopener noreferrer" className="underline">Click here</a></p>
            )}
          </div>
        )}
      </CardContent>
      {canCancel && (
        <CardFooter>
          <Button
            onClick={() => jobId && cancelJob(jobId)}
            variant="destructive"
            disabled={isCancelling || !jobId}
            className="w-full"
          >
            {isCancelling ? 'Cancelling...' : 'Cancel Job'}
          </Button>
        </CardFooter>
      )}
    </Card>
  );
};

export default JobProgressDisplay;
