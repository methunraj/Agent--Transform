import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic'; // Ensure fresh data, backend handles caching

/**
 * API route handler for fetching available AI models.
 * Proxies the request to the backend /api/models endpoint.
 */
export async function GET(request: NextRequest) {
  try {
    const backendUrl = process.env.AGNO_BACKEND_URL || 'http://localhost:8001';
    const modelsBackendUrl = `${backendUrl}/api/models`;

    const backendResponse = await fetch(modelsBackendUrl, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        // Add any other necessary headers if your backend requires them (e.g., API key for this endpoint itself)
      },
      // cache: 'no-store', // Let the backend handle caching logic
    });

    const responseData = await backendResponse.json();

    if (!backendResponse.ok) {
      console.error(`Backend error fetching models: ${backendResponse.status}`, responseData);
      return NextResponse.json(
        { detail: responseData.detail || `Backend error: ${backendResponse.status}` },
        { status: backendResponse.status }
      );
    }

    return NextResponse.json(responseData, { status: backendResponse.status });

  } catch (error: any) {
    console.error('Error in /api/ai-models route:', error);
    return NextResponse.json(
      { detail: `Internal server error: ${error.message || 'Unknown error'}` },
      { status: 500 }
    );
  }
}
