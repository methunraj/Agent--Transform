import { useState, useEffect, useCallback, useRef } from 'react';
import { useToast } from '@/components/ui/use-toast'; // Assuming a toast component

export interface JobProgressState {
  jobId: string | null;
  status: string | null;
  progress: number; // Percentage 0-100
  currentStep: string | null;
  result: any | null; // To hold final job result or error details
  error: string | null; // Specific error message for progress display
  timeElapsed: number; // Seconds
  isConnected: boolean;
}

const initialJobProgressState: JobProgressState = {
  jobId: null,
  status: null,
  progress: 0,
  currentStep: 'Initializing...',
  result: null,
  error: null,
  timeElapsed: 0,
  isConnected: false,
};

interface UseJobProgressOptions {
  jobId: string | null;
  connectionType?: 'sse' | 'websocket'; // Default to 'sse'
  autoConnect?: boolean; // Default to true if jobId is provided
}

export const useJobProgress = ({
  jobId,
  connectionType = 'sse',
  autoConnect = true,
}: UseJobProgressOptions) => {
  const [jobProgress, setJobProgress] = useState<JobProgressState>(initialJobProgressState);
  const { toast } = useToast();
  const eventSourceRef = useRef<EventSource | null>(null);
  const websocketRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const startTimeRef = useRef<number | null>(null);

  const updateJobState = useCallback((data: Partial<JobProgressState>) => {
    setJobProgress(prev => ({ ...prev, ...data }));
  }, []);

  const resetJobProgress = useCallback(() => {
    setJobProgress(initialJobProgressState);
    if (timerRef.current) clearInterval(timerRef.current);
    startTimeRef.current = null;
  }, []);

  const stopConnections = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
      console.log(`SSE connection closed for job ${jobProgress.jobId}`);
    }
    if (websocketRef.current) {
      websocketRef.current.close();
      websocketRef.current = null;
      console.log(`WebSocket connection closed for job ${jobProgress.jobId}`);
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    updateJobState({ isConnected: false });
  }, [jobProgress.jobId, updateJobState]);

  const connect = useCallback(() => {
    if (!jobId) {
      // console.warn('Cannot connect: Job ID is null.');
      // resetJobProgress(); // Reset if no job ID
      return;
    }

    if (jobProgress.isConnected && jobProgress.jobId === jobId) {
      // console.log(`Already connected to job ${jobId}`);
      return;
    }

    stopConnections(); // Ensure previous connections are closed before starting new one
    resetJobProgress(); // Reset state for the new job
    updateJobState({ jobId, isConnected: true, currentStep: 'Connecting...' });
    startTimeRef.current = Date.now();

    timerRef.current = setInterval(() => {
      if (startTimeRef.current) {
        const elapsed = Math.floor((Date.now() - startTimeRef.current) / 1000);
        updateJobState({ timeElapsed: elapsed });
      }
    }, 1000);

    if (connectionType === 'sse') {
      console.log(`Connecting to SSE for job ${jobId}...`);
      // Backend SSE endpoint: GET /job/{job_id}/stream
      const sseUrl = `/api/jobs/stream/${jobId}`; // Use Next.js API route as proxy

      eventSourceRef.current = new EventSource(sseUrl);

      eventSourceRef.current.onopen = () => {
        console.log(`SSE connection opened for job ${jobId}`);
        updateJobState({ isConnected: true, currentStep: 'Connected, waiting for updates...' });
      };

      eventSourceRef.current.addEventListener('job_update', (event) => {
        try {
          const data = JSON.parse(event.data);
          updateJobState({
            status: data.status,
            progress: data.progress * 100, // Assuming progress is 0-1, convert to 0-100
            currentStep: data.current_step,
            result: data.result,
            error: data.status === 'FAILED' ? (data.result?.error || 'Job failed') : null,
          });
          if (data.status === 'COMPLETED' || data.status === 'FAILED' || data.status === 'CANCELLED') {
            stopConnections();
          }
        } catch (e) {
          console.error('Failed to parse SSE job_update data:', e);
        }
      });

      eventSourceRef.current.addEventListener('error', (event) => {
        console.error(`SSE error for job ${jobId}:`, event);
        const es = event.target as EventSource;
        if (es.readyState === EventSource.CLOSED) {
            updateJobState({ isConnected: false, error: 'Connection closed by server.', currentStep: 'Disconnected' });
            toast({ title: 'Stream Error', description: 'Connection to job updates lost.', variant: 'destructive' });
        } else {
            updateJobState({ isConnected: false, error: 'Connection error.', currentStep: 'Connection error' });
        }
        stopConnections();
      });

    } else { // WebSocket
      console.log(`Connecting to WebSocket for job ${jobId}...`);
      // Backend WebSocket endpoint: /job/{job_id}/ws
      // Construct WebSocket URL (ws:// or wss://)
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      // Assuming Next.js API route proxy for WS at /api/jobs/ws/[jobId] (or similar)
      // For direct connection (if backend and frontend are on same host or CORS is set up):
      // const backendHost = process.env.NEXT_PUBLIC_BACKEND_WS_HOST || window.location.host;
      // const wsUrl = `${wsProtocol}//${backendHost}/job/${jobId}/ws`;
      // For now, let's assume direct connection for simplicity or a well-configured proxy
      // This might need a Next.js API route that handles WebSocket proxying if direct connection is an issue.
      const wsUrl = `${wsProtocol}//${window.location.host}/api/jobs/ws/${jobId}`; // Placeholder for proxied WS

      websocketRef.current = new WebSocket(wsUrl);

      websocketRef.current.onopen = () => {
        console.log(`WebSocket connection opened for job ${jobId}`);
        updateJobState({ isConnected: true, currentStep: 'Connected, waiting for updates...' });
      };

      websocketRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data as string);
          updateJobState({
            status: data.status,
            progress: data.progress * 100, // Assuming progress is 0-1
            currentStep: data.current_step,
            result: data.result,
            error: data.status === 'FAILED' ? (data.result?.error || 'Job failed') : null,
          });
          if (data.status === 'COMPLETED' || data.status === 'FAILED' || data.status === 'CANCELLED') {
            stopConnections();
          }
        } catch (e) {
          console.error('Failed to parse WebSocket message data:', e);
        }
      };

      websocketRef.current.onclose = (event) => {
        console.log(`WebSocket connection closed for job ${jobId}. Code: ${event.code}, Reason: ${event.reason}`);
        updateJobState({ isConnected: false, currentStep: 'Disconnected' });
        if (!event.wasClean && jobProgress.status !== 'COMPLETED' && jobProgress.status !== 'FAILED' && jobProgress.status !== 'CANCELLED') {
          toast({ title: 'Connection Lost', description: 'Lost connection to job updates.', variant: 'destructive' });
          updateJobState({ error: 'Connection lost' });
        }
        if (timerRef.current) clearInterval(timerRef.current);
      };

      websocketRef.current.onerror = (error) => {
        console.error(`WebSocket error for job ${jobId}:`, error);
        updateJobState({ isConnected: false, error: 'Connection error.', currentStep: 'Connection error' });
        toast({ title: 'Stream Error', description: 'Error connecting to job updates.', variant: 'destructive' });
        stopConnections();
      };
    }
  }, [jobId, connectionType, updateJobState, stopConnections, toast, jobProgress.isConnected, jobProgress.status, resetJobProgress]);

  useEffect(() => {
    if (jobId && autoConnect) {
      connect();
    } else if (!jobId) {
        // If jobId becomes null (e.g. user navigates away or job finishes and is cleared)
        stopConnections();
        resetJobProgress();
    }
    // Cleanup function to close connections when the hook unmounts or jobId changes
    return () => {
      stopConnections();
    };
  }, [jobId, autoConnect, connect, stopConnections, resetJobProgress]);

  return {
    jobProgress,
    connect,
    disconnect: stopConnections, // Expose disconnect as stopConnections
  };
};

// Note:
// For SSE, you'll need a Next.js API route at `/pages/api/jobs/stream/[jobId].ts` (or app router equivalent)
// that proxies the request to the backend's `/job/{job_id}/stream` SSE endpoint.
// For WebSockets, direct connection is simpler if same-origin or CORS allows.
// If proxying WebSockets via Next.js API routes, it's more complex and might require a custom server
// or specific Vercel configurations if deploying there.
// The placeholder `/api/jobs/ws/${jobId}` implies a proxy is needed.
// For this implementation, I'm focusing on the hook logic. The actual API route for SSE proxy is required.
// A WebSocket proxy is more involved.
