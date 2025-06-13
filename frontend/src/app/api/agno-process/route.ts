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
      console.log('Starting Agno processing request with 20 minute timeout...');
      
      // Use undici with proper timeout controls
      const response = await request(`${pythonBackendUrl}/process`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          json_data: extractedData,  // Changed from extracted_data to match Python API
          file_name: fileName,
          description: `Extracted from ${fileName} using ${llmProvider}/${model}`,
          api_key: finalApiKey,  // Use the resolved API key
          model: model  // Python backend will use this model for Agno processing
        }),
        // Set proper timeouts for long-running operations
        headersTimeout: 20 * 60 * 1000, // 20 minutes for headers
        bodyTimeout: 20 * 60 * 1000,    // 20 minutes for body
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