import { NextRequest, NextResponse } from 'next/server';

/**
 * API route handler for cancelling a job.
 * Proxies the request to the backend job cancellation endpoint.
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { jobId } = body;

    if (!jobId) {
      return NextResponse.json({ detail: 'Job ID is required' }, { status: 400 });
    }

    const backendUrl = process.env.AGNO_BACKEND_URL || 'http://localhost:8001';
    const cancelUrl = `${backendUrl}/job/${jobId}/cancel`;

    const backendResponse = await fetch(cancelUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        // Add any other necessary headers, like Authorization if required
      },
      // Note: The backend /job/{job_id}/cancel endpoint doesn't expect a body,
      // but fetch POST might require it or send an empty one by default.
      // Explicitly sending an empty body if that's what the backend expects,
      // or remove if backend strictly disallows body for this POST.
      // For now, let's assume no body is needed for the backend call.
    });

    const responseData = await backendResponse.json();

    if (!backendResponse.ok) {
      // Forward the error status and detail from the backend
      return NextResponse.json(
        { detail: responseData.detail || `Backend error: ${backendResponse.status}` },
        { status: backendResponse.status }
      );
    }

    // Forward the successful response from the backend
    return NextResponse.json(responseData, { status: backendResponse.status });

  } catch (error: any) {
    console.error('Error in /api/jobs/cancel route:', error);
    return NextResponse.json(
      { detail: `Internal server error: ${error.message || 'Unknown error'}` },
      { status: 500 }
    );
  }
}
