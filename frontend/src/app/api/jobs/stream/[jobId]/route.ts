import { NextRequest } from 'next/server';
import { PassThrough } from 'stream';

export const dynamic = 'force-dynamic'; // Opt out of caching for this route

export async function GET(
  request: NextRequest,
  { params }: { params: { jobId: string } }
) {
  const { jobId } = params;

  if (!jobId) {
    return new Response('Job ID is required', { status: 400 });
  }

  const backendUrl = process.env.AGNO_BACKEND_URL || 'http://localhost:8001';
  const sseBackendUrl = `${backendUrl}/job/${jobId}/stream`;

  try {
    const backendResponse = await fetch(sseBackendUrl, {
      method: 'GET',
      headers: {
        'Accept': 'text/event-stream',
        // Add any other necessary headers, like Authorization if required
      },
      // Important: Set duplex to 'half' for streaming with undici/fetch in Next.js
      // This allows the response body to be streamed.
      // @ts-ignore - duplex is a valid option for undici but might not be in all TS lib.dom versions
      duplex: 'half',
    });

    if (!backendResponse.ok || !backendResponse.body) {
      const errorText = await backendResponse.text();
      console.error(`Backend SSE error for job ${jobId}: ${backendResponse.status} ${errorText}`);
      return new Response(errorText || `Error fetching SSE stream from backend: ${backendResponse.status}`, {
        status: backendResponse.status,
      });
    }

    // Create a PassThrough stream to pipe the backend's SSE stream to the client
    const stream = new PassThrough();
    const reader = backendResponse.body.getReader();

    const pump = async () => {
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            stream.end();
            break;
          }
          stream.write(value);
        }
      } catch (error) {
        console.error(`Error pumping SSE stream for job ${jobId}:`, error);
        stream.destroy(error as Error);
      }
    };

    pump();

    return new Response(stream as any, { // Cast stream to any to satisfy Response type
      status: 200,
      headers: {
        'Content-Type': 'text/event-stream; charset=utf-8',
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no', // Useful for Nginx environments
      },
    });

  } catch (error: any) {
    console.error(`Error in /api/jobs/stream/${jobId} route:`, error);
    return new Response(`Internal server error: ${error.message || 'Unknown error'}`, {
      status: 500,
    });
  }
}
