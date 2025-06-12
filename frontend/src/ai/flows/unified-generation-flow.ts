'use server';
/**
 * @fileOverview Unified AI generation flow for creating complete extraction configurations.
 */

import { ai } from '@/ai/genkit';
import { z } from 'genkit';

// Input schema for unified generation
const UnifiedGenerationInputSchema = z.object({
  userIntent: z.string().describe("User's description of what they want to extract"),
  exampleCount: z.number().int().min(1).max(5).describe("Number of examples to generate"),
  llmProvider: z.string().optional().describe("LLM provider"),
  modelName: z.string().optional().describe("Model name to use"),
  temperature: z.number().optional().describe("Generation temperature"),
});

export type UnifiedGenerationInput = z.infer<typeof UnifiedGenerationInputSchema>;

// Output schema for unified generation
const UnifiedGenerationOutputSchema = z.object({
  schema: z.string().describe("Generated JSON schema"),
  systemPrompt: z.string().describe("Generated system prompt"),
  userPromptTemplate: z.string().describe("Generated user prompt template"),
  examples: z.array(z.object({
    input: z.string(),
    output: z.string()
  })).describe("Generated few-shot examples"),
  reasoning: z.string().optional().describe("AI's reasoning for the generation"),
  confidence: z.number().optional().describe("Confidence score 0-1"),
});

export type UnifiedGenerationOutput = z.infer<typeof UnifiedGenerationOutputSchema>;

// Generation prompt template - very simple
const generationPrompt = ai.definePrompt({
  name: 'unifiedGenerationPrompt',
  input: { 
    schema: z.object({
      userIntent: z.string(),
      exampleCount: z.number(),
    })
  },
  output: { 
    schema: z.object({
      schema: z.string(),
      systemPrompt: z.string(),
      userPromptTemplate: z.string(),
      examples: z.array(z.object({ input: z.string(), output: z.string() })),
      reasoning: z.string(),
    })
  },
  prompt: `Create a complete data extraction configuration based on this request:

USER REQUEST: {{{userIntent}}}
EXAMPLES TO GENERATE: {{{exampleCount}}}

Generate these components:

1. JSON SCHEMA: A valid JSON schema that defines the structure for the data to extract.

2. SYSTEM PROMPT: Instructions for the AI model on how to perform the extraction.

3. USER PROMPT TEMPLATE: A template that includes these placeholders:
   - {{document_content_text}} for text content
   - {{json_schema_text}} for the schema
   - {{examples_list}} for the examples

4. EXAMPLES: Sample input and output pairs that demonstrate the extraction.

5. REASONING: Brief explanation of your design choices.`,
  config: {
    temperature: 0.7,
  },
});

// Main unified generation flow
const unifiedGenerationFlow = ai.defineFlow(
  {
    name: 'unifiedGenerationFlow',
    inputSchema: UnifiedGenerationInputSchema,
    outputSchema: UnifiedGenerationOutputSchema,
  },
  async (input) => {
    const promptData = {
      userIntent: input.userIntent,
      exampleCount: input.exampleCount,
    };

    const callConfig: Record<string, any> = {};
    if (input.temperature !== undefined) {
      callConfig.temperature = input.temperature;
    }

    let modelToUse: string | undefined = undefined;
    if (input.modelName) {
      if (input.llmProvider === 'googleAI' && !input.modelName.startsWith('googleai/')) {
        modelToUse = `googleai/${input.modelName}`;
      } else {
        modelToUse = input.modelName;
      }
    }

    const { output } = await generationPrompt(promptData, {
      model: modelToUse,
      config: Object.keys(callConfig).length > 0 ? callConfig : undefined,
    });

    if (!output) {
      throw new Error("Generation failed: No output from LLM.");
    }

    return {
      schema: output.schema,
      systemPrompt: output.systemPrompt,
      userPromptTemplate: output.userPromptTemplate,
      examples: output.examples,
      reasoning: output.reasoning,
      confidence: 0.85,
    };
  }
);

export async function generateUnifiedConfiguration(input: UnifiedGenerationInput): Promise<UnifiedGenerationOutput> {
  return unifiedGenerationFlow(input);
}