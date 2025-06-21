import { NextRequest, NextResponse } from 'next/server';
import { request } from 'undici';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { extractedData, fileName, llmProvider, model, apiKey, temperature } = body;

    // Validate required fields
    // For googleAI provider, API key might be optional (using default key)
    if (!extractedData || !fileName || !model) {
      return NextResponse.json(
        { success: false, error: 'Missing required fields: extractedData, fileName, or model' },
        { status: 400 }
      );
    }
    
    // Use API key from request or get from environment for googleAI
    const finalApiKey = apiKey || (llmProvider === 'googleAI' ? process.env.GOOGLE_API_KEY : null);
    
    if (!finalApiKey) {
      return NextResponse.json(
        { success: false, error: 'API key is required' },
        { status: 400 }
      );
    }

    // Forward request to Python backend
    const pythonBackendUrl = process.env.AGNO_BACKEND_URL || 'http://localhost:8001';
    
    console.log(`Attempting to connect to Python backend at: ${pythonBackendUrl}`);
    
    try {
      const requestTimeoutMs = 2700 * 1000; // 45 minutes in milliseconds
      console.log(`Starting Agno processing request with ${requestTimeoutMs / 60000} minute timeout...`);
      
      // Use undici with proper timeout controls
      // Note: The /process endpoint in the backend is deprecated and has its own timeout handling.
      // This proxy should ideally call the new /process-async endpoint and then
      // the client would poll /job/{job_id}/status.
      // However, the current frontend JobContext calls this /api/agno-process expecting a direct response.
      // For now, we'll update the timeout, but this flow might need a larger refactor
      // if the backend /process truly becomes slow or is removed.
      // The backend's /process endpoint (deprecated) internally creates a job and polls.
      // So, this timeout here is for the entire duration of that polling by the backend's /process.
      const response = await request(`${pythonBackendUrl}/process`, { // This still calls the deprecated backend endpoint
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          json_data: extractedData,
          file_name: fileName,
          description: `Extracted from ${fileName} using ${llmProvider}/${model}`,
          // api_key: finalApiKey, // The /process endpoint in backend doesn't expect api_key directly in body
                                 // It relies on GOOGLE_API_KEY env var.
          model: model,
          // Ensure other ProcessRequest fields are passed if needed by the backend /process
           processing_mode: "auto", // Example: or get from client request if available
           user_id: `user_${Math.random().toString(36).substr(2, 9)}`, // Example user_id
           session_id: `session_${Math.random().toString(36).substr(2, 9)}` // Example session_id
        }),
        headersTimeout: requestTimeoutMs,
        bodyTimeout: requestTimeoutMs,
      });

      if (response.statusCode !== 200) {
        const errorText = await response.body.text();
        console.error('Python backend error:', errorText);
        return NextResponse.json(
          { success: false, error: 'Agno processing failed' },
          { status: 500 }
        );
      }

      const result = await response.body.json() as any;
      
      // If successful, add the backend URL to the download URL for frontend access
      if (result.success && result.download_url) {
        // The download URL is already properly formatted as /download/{file_id}
        result.download_url = `${pythonBackendUrl}${result.download_url}`;
        
        // Add token usage information if needed (Python backend doesn't currently provide this)
        // You can add token tracking in the Python backend if needed
        result.token_usage = {
          input_tokens: 0,  // Placeholder - implement in Python if needed
          output_tokens: 0,
          total_tokens: 0
        };
        
        // Add processing cost if needed
        result.processingCost = 0;  // Placeholder - calculate based on actual usage
      }
      
      return NextResponse.json(result);
      
    } catch (fetchError) {
      // Handle timeout and other fetch errors
      
      if (fetchError instanceof Error && fetchError.name === 'AbortError') {
        console.error('Python backend request timed out after 20 minutes');
        return NextResponse.json(
          { success: false, error: 'Processing timed out after 20 minutes' },
          { status: 408 }
        );
      }
      
      console.error('Python backend fetch error:', fetchError);
      return NextResponse.json(
        { success: false, error: 'Failed to connect to processing service' },
        { status: 500 }
      );
    }

  } catch (error) {
    console.error('Agno API route error:', error);
    return NextResponse.json(
      { success: false, error: 'Internal server error' },
      { status: 500 }
    );
  }
}