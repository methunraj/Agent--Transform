import {genkit} from 'genkit';
import {googleAI} from '@genkit-ai/googleai';
import { GoogleGenAI } from '@google/genai';

// Create googleAI instance for backward compatibility with existing flows
const googleAIPlugin = googleAI();

// Legacy Genkit AI instance for existing flows
export const ai = genkit({
  plugins: [googleAIPlugin],
  model: 'googleai/gemini-2.5-flash-preview-05-20', // Updated to latest model
});

// New Google Gen AI client for direct SDK usage
export const googleAIClient = new GoogleGenAI({
  apiKey: process.env.GOOGLE_API_KEY || '',
  httpOptions: { apiVersion: 'v1beta' }
});

// Default model configuration
export const DEFAULT_MODEL = 'gemini-2.5-flash-preview-05-20';
export const FALLBACK_MODEL = 'gemini-2.0-flash';

// Token counting function using the new Google AI SDK
export async function countTokens(model: string, contents: any) {
  try {
    // Convert legacy model names to new format
    const normalizedModel = normalizeModelName(model);
    
    const result = await googleAIClient.models.countTokens({
      model: normalizedModel,
      contents: typeof contents === 'string' ? contents : JSON.stringify(contents)
    });
    
    // Log token details for debugging
    console.log(`[TOKEN COUNT] Model: ${normalizedModel}`);
    console.log(`[TOKEN COUNT] Total tokens: ${result.totalTokens}`);
    
    return result;
  } catch (error) {
    console.error('[TOKEN COUNT ERROR]:', error);
    // Fallback to legacy method for backward compatibility
    console.log(`[TOKEN METADATA] Model: ${model}`);
    console.log(`[TOKEN METADATA] Content type: ${typeof contents}`);
    
    if (typeof contents === 'string') {
      console.log(`[TOKEN METADATA] Content length: ${contents.length} chars`);
      console.log(`[TOKEN METADATA] Content sample: ${contents.substring(0, 100)}${contents.length > 100 ? '...' : ''}`);
    } else {
      console.log(`[TOKEN METADATA] Content: ${JSON.stringify(contents).substring(0, 200)}...`);
    }
    
    return { totalTokens: 0 };
  }
}

// Helper function to normalize model names
function normalizeModelName(model: string): string {
  // Remove 'googleai/' prefix if present
  if (model.startsWith('googleai/')) {
    model = model.replace('googleai/', '');
  }
  
  // Map legacy model names to current ones
  const modelMap: Record<string, string> = {
    // Legacy mappings
    'gemini-1.5-flash-latest': 'gemini-2.5-flash-preview-05-20',
    'gemini-1.5-pro-latest': 'gemini-2.5-pro-preview-05-06',
    'gemini-2.5-flash-preview-04-17': 'gemini-2.5-flash-preview-05-20',
    'gemini-2.5-flash': 'gemini-2.5-flash-preview-05-20',
    'gemini-2.5-pro': 'gemini-2.5-pro-preview-05-06',
    'gemini-1.5-flash': 'gemini-2.5-flash-preview-05-20',
    'gemini-1.5-pro': 'gemini-2.5-pro-preview-05-06'
  };
  
  return modelMap[model] || model;
}

// Available model constants for easy reference
export const MODELS = {
  // Core text and multimodal models for data extraction
  FLASH_PREVIEW: 'gemini-2.5-flash-preview-05-20',
  PRO_PREVIEW: 'gemini-2.5-pro-preview-05-06',
  FLASH_2_0: 'gemini-2.0-flash',
  FLASH_LITE: 'gemini-2.0-flash-lite',
} as const;

// Generate content with the new Google AI SDK
export async function generateContent({
  model = DEFAULT_MODEL,
  contents,
  config
}: {
  model?: string;
  contents: string;
  config?: {
    maxOutputTokens?: number;
    temperature?: number;
    topK?: number;
    topP?: number;
    stopSequences?: string[];
    systemInstruction?: string;
    safetySettings?: any[];
  };
}) {
  const normalizedModel = normalizeModelName(model);
  
  try {
    const response = await googleAIClient.models.generateContent({
      model: normalizedModel,
      contents,
      ...(config && { config })
    });
    return response;
  } catch (error) {
    console.error('[GENERATION ERROR]:', error);
    
    // Try fallback model if primary model fails
    const normalizedFallback = normalizeModelName(FALLBACK_MODEL);
    if (normalizedModel !== normalizedFallback) {
      console.log(`[FALLBACK] Trying with ${normalizedFallback}`);
      return generateContent({
        model: FALLBACK_MODEL,
        contents,
        config
      });
    }
    
    throw error;
  }
}

// Generate streaming content with the new Google AI SDK
export async function generateContentStream(options: Parameters<typeof generateContent>[0]) {
  try {
    const { model = DEFAULT_MODEL, contents, config } = options;
    const normalizedModel = normalizeModelName(model);
    
    const stream = await googleAIClient.models.generateContentStream({
      model: normalizedModel,
      contents,
      ...(config && { config })
    });
    return stream;
  } catch (error) {
    console.error('[STREAMING ERROR]:', error);
    throw error;
  }
}

// File upload utility using the new Google AI SDK
export async function uploadFile(filePath: string, mimeType: string, displayName?: string) {
  try {
    const response = await googleAIClient.files.upload({
      file: filePath,
      config: { 
        mimeType,
        ...(displayName && { displayName })
      }
    });
    
    console.log(`[FILE UPLOAD] Uploaded: ${response.name}`);
    return response;
  } catch (error) {
    console.error('[FILE UPLOAD ERROR]:', error);
    throw error;
  }
}

// Context caching utilities using the new Google AI SDK
export async function createCache({
  model = DEFAULT_MODEL,
  contents,
  systemInstruction,
  ttl = '3600s'
}: {
  model?: string;
  contents: any[];
  systemInstruction?: string;
  ttl?: string;
}) {
  try {
    const normalizedModel = normalizeModelName(model);
    
    const cache = await googleAIClient.caches.create({
      model: normalizedModel,
      config: {
        contents,
        ...(systemInstruction && { systemInstruction }),
        ttl
      }
    });
    
    console.log(`[CACHE CREATED] Name: ${cache.name}`);
    return cache;
  } catch (error) {
    console.error('[CACHE CREATE ERROR]:', error);
    throw error;
  }
}

export async function listCaches() {
  try {
    const caches = await googleAIClient.caches.list();
    return caches;
  } catch (error) {
    console.error('[CACHE LIST ERROR]:', error);
    throw error;
  }
}

export async function deleteCache(name: string) {
  try {
    await googleAIClient.caches.delete({ name });
    console.log(`[CACHE DELETED] Name: ${name}`);
  } catch (error) {
    console.error('[CACHE DELETE ERROR]:', error);
    throw error;
  }
}
