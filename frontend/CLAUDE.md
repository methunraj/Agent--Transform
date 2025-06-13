# Claude Development Guidelines for Google Gen AI API

## API Documentation Reference
For all Google Gen AI API related development, strictly refer to the documentation in `/api-context/` directory. These documents contain the latest and most accurate information for implementing Google Gen AI features.

## Core API Rules and Guidelines

### 1. SDK Usage
- **Always use** `@google/genai` SDK - the latest official Google Gen AI SDK
- **Avoid** older Google AI SDKs or deprecated packages
- Use TypeScript for type safety and better developer experience

### 2. Model Selection
- **gemini-2.5-flash-preview-05-20**: Latest model with adaptive thinking and cost efficiency - use for most tasks
- **gemini-2.5-pro-preview-05-06**: Enhanced thinking and reasoning - use for complex analysis and coding
- **gemini-2.0-flash**: Next generation features with speed and thinking - use for real-time applications
- **gemini-2.0-flash-lite**: Cost efficient with low latency - use for simple, fast tasks

### 3. Authentication and Client Setup
```typescript
import { GoogleGenAI } from '@google/genai';

const ai = new GoogleGenAI({
  apiKey: process.env.GOOGLE_API_KEY,
  httpOptions: { apiVersion: 'v1beta' } // Use v1beta unless you need v1alpha features
});
```

### 4. Error Handling Requirements
- Always implement proper error handling with try-catch blocks
- Handle rate limits with exponential backoff
- Validate inputs before API calls
- Gracefully handle blocked content and safety filter responses

### 5. Cost Optimization Rules
- Use `countTokens()` to monitor token usage before expensive operations
- Implement context caching for content ≥32K tokens (≥2K for older models)
- Use streaming responses for long content generation
- Cache responses when appropriate to reduce API calls

### 6. Performance Best Practices
- Use streaming for real-time user experiences: `generateContentStream()`
- Implement context caching for repeated large content
- Use appropriate model for task complexity (don't use pro models for simple tasks)
- Batch similar requests when possible

### 7. Content and Safety Guidelines
- Always configure safety settings appropriately for your use case
- Implement content moderation for user-generated inputs
- Handle blocked content gracefully with fallback responses
- Validate file uploads against size and format limits

### 8. File Upload Limits
- **Images**: Max 20MB, 3072x3072 pixels
- **Audio**: Max 20MB, 9.5 hours duration  
- **Video**: Max 2GB, 2 hours duration
- Always validate file sizes before upload

### 9. Function Calling Standards
- Define functions with proper JSON Schema validation
- Use descriptive function names and parameter descriptions
- Implement proper parameter validation
- Handle function execution errors gracefully
- Support AUTO, REQUIRED, or NONE function calling modes

### 10. Structured Output Requirements
- Use JSON Schema for structured output validation
- Specify MIME type as 'application/json' for JSON responses
- Validate generated JSON against expected schema
- Handle schema validation errors appropriately

### 11. Context Management
- Respect context window limits:
  - gemini-2.0-flash: 1M tokens
  - gemini-1.5-flash: 1M tokens
  - gemini-1.5-pro: 2M tokens
- Use system instructions for consistent behavior
- Maintain conversation context efficiently

### 12. Rate Limiting
- Respect rate limits (15 RPM for free tier)
- Implement exponential backoff for rate limit errors
- Monitor usage to avoid quota exhaustion
- Use caching to reduce API calls

## Development Patterns

### Basic Text Generation
```typescript
const response = await ai.models.generateContent({
  model: 'gemini-2.5-flash-preview-05-20',
  contents: 'Your prompt here',
  config: {
    temperature: 0.7,
    maxOutputTokens: 1000
  }
});
```

### Multi-modal Input
```typescript
// First upload the file
const file = await ai.files.upload({
  file: 'path/to/image.jpg',
  config: { mimeType: 'image/jpeg' }
});

const response = await ai.models.generateContent({
  model: 'gemini-2.5-flash-preview-05-20',
  contents: [
    { text: 'Analyze this image:' },
    { fileData: { mimeType: file.mimeType, fileUri: file.uri } }
  ]
});
```

### Function Calling
```typescript
const tools = [{
  functionDeclarations: [{
    name: 'function_name',
    description: 'Function description',
    parameters: {
      type: 'object',
      properties: {
        param: { type: 'string', description: 'Parameter description' }
      },
      required: ['param']
    }
  }]
}];

const response = await ai.models.generateContent({
  model: 'gemini-2.0-flash',
  contents: [{ role: 'user', parts: [{ text: 'Use the function' }] }],
  tools,
  toolConfig: { functionCallingConfig: { mode: 'AUTO' } }
});
```

### Context Caching
```typescript
const cache = await ai.caches.create({
  model: 'gemini-2.5-flash-preview-05-20',
  config: {
    contents: [{ text: 'Large context content...' }],
    systemInstruction: 'You are an expert analyzer.',
    ttl: '3600s'
  }
});

const response = await ai.models.generateContent({
  model: 'gemini-2.5-flash-preview-05-20',
  contents: 'Question about cached content',
  config: { cachedContent: cache.name }
});
```

## Documentation References
- **Text Generation**: `/api-context/text-generation.md`
- **Image Understanding**: `/api-context/image-understanding.md`
- **Video Understanding**: `/api-context/video-understanding.md`
- **Audio Understanding**: `/api-context/audio-understanding.md`
- **Function Calling**: `/api-context/function-calling.md`
- **Structured Output**: `/api-context/structured-output.md`
- **Context Caching**: `/api-context/context-caching.md`
- **Code Execution**: `/api-context/code-execution.md`
- **Live API**: `/api-context/live-api.md`
- **System Instructions**: `/api-context/system-instructions.md`
- **Token Counting**: `/api-context/token-counting.md`
- **Quickstart Guide**: `/api-context/quickstart.md`

## Important Notes
- Always check the api-context documentation for the most up-to-date examples and best practices
- Test all implementations thoroughly, especially error handling
- Monitor API usage and costs in production
- Keep API keys secure and never commit them to version control
- Follow Google's terms of service and content policies