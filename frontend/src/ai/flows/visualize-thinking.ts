'use server';

/**
 * @fileOverview A flow that visualizes the model's thinking process for better insights into complex extractions.
 *
 * - visualizeThinking - A function that enables visualizing the model's thinking process by streaming chunks.
 * - VisualizeThinkingInput - The input type for the visualizeThinking function.
 * - VisualizeThinkingOutput - The return type for the visualizeThinking function (final complete output).
 * - VisualizeThinkingStreamChunk - The type for streamed chunks.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const VisualizeThinkingInputSchema = z.object({
  query: z.string().describe('The query or prompt context for which to visualize the thinking process.'),
  explanationDetailBudget: z.number().int().min(0).optional().describe('A numeric budget to guide the detail level of the thinking explanation (e.g., 0 for brief, higher for more detail).'),
  llmProvider: z.string().optional(), 
  modelName: z.string().optional(),
  temperature: z.number().min(0.0).max(2.0).optional().describe("Controls randomness for the thinking visualization generation. Range: 0.0 - 2.0."),
});
export type VisualizeThinkingInput = z.infer<typeof VisualizeThinkingInputSchema>;

const VisualizeThinkingOutputSchema = z.object({
  thinkingProcess: z.string().describe('The complete visualized thinking process of the model.'),
});
export type VisualizeThinkingOutput = z.infer<typeof VisualizeThinkingOutputSchema>;

const VisualizeThinkingStreamChunkSchema = z.object({
  chunk: z.string().describe('A chunk of the thinking process text.'),
});
export type VisualizeThinkingStreamChunk = z.infer<typeof VisualizeThinkingStreamChunkSchema>;


export const visualizeThinking = ai.defineFlow(
  {
    name: 'visualizeThinkingFlow',
    inputSchema: VisualizeThinkingInputSchema,
    outputSchema: VisualizeThinkingOutputSchema, 
    streamSchema: VisualizeThinkingStreamChunkSchema, 
  },
  async (input, DANGEROUS_API_may_change_in_future_streamingCallback) => {
    
    const promptInputContext = {
      query: input.query,
      explanationDetailBudget: input.explanationDetailBudget,
    };

    const callConfig: Record<string, any> = {};
    if (input.temperature !== undefined) {
      callConfig.temperature = input.temperature;
    }
   
    if (input.llmProvider === 'googleAI' && input.explanationDetailBudget !== undefined && input.modelName && modelsSupportingThinkingBudget.includes(input.modelName.replace('googleai/',''))) {
        callConfig.thinkingConfig = { 
            thinkingBudget: input.explanationDetailBudget
        };
    }
    
    let modelToUse: string | undefined = undefined;
    if (input.modelName) {
      if (input.llmProvider === 'googleAI' && !input.modelName.startsWith('googleai/')) {
        modelToUse = `googleai/${input.modelName}`;
      } else {
        modelToUse = input.modelName;
      }
    }

    const { stream: promptStream, response: promptResponsePromise } = 
      // @ts-ignore - Temporarily ignore type errors while API stabilizes
      ai.generateStream({
          prompt: visualizeThinkingPrompt, 
          input: promptInputContext,      
          model: modelToUse,
          config: Object.keys(callConfig).length > 0 ? callConfig : undefined,
        }
      );
    
    let accumulatedProcess = "";
    for await (const chunk of promptStream) { 
      // Try to access chunk.output first, then fallback to chunk.text
      // @ts-ignore - Temporarily ignore type errors while API stabilizes 
      const textChunkFromOutput = chunk.output?.thinkingProcess ?? chunk.text ?? '';
      
      if (textChunkFromOutput) {
        if (DANGEROUS_API_may_change_in_future_streamingCallback) {
          DANGEROUS_API_may_change_in_future_streamingCallback({ chunk: textChunkFromOutput });
        }
        accumulatedProcess += textChunkFromOutput;
      }
    }

    const finalResponse = await promptResponsePromise; 
    // @ts-ignore - Temporarily ignore type errors while API stabilizes
    const completeThinkingProcess = finalResponse?.output?.thinkingProcess || accumulatedProcess;
    
    return { thinkingProcess: completeThinkingProcess };
  }
);

const modelsSupportingThinkingBudget = [
  'gemini-2.5-flash-preview-05-20', 
  'gemini-2.5-pro-preview-05-06',   
  'gemini-2.0-flash', 
  'gemini-2.0-flash-lite',
];

const visualizeThinkingPrompt = ai.definePrompt({
  name: 'visualizeThinkingPrompt',
  input: {schema: z.object({ query: z.string(), explanationDetailBudget: z.number().optional() }) },
  output: {schema: VisualizeThinkingOutputSchema}, 
  prompt: `You are an AI assistant that explains the thinking process of another AI model.
For the following query context:
"{{{query}}}"

{{#if explanationDetailBudget}}
Please provide an explanation of the steps and reasoning the model likely takes to arrive at an answer related to this query context.
Your response should be approximately {{explanationDetailBudget}} tokens. A higher budget implies more detail is desired. If the budget is 0 or very low, provide a very brief overview.
{{else}}
Please provide a concise explanation of the steps and reasoning the model likely takes to arrive at an answer related to this query context.
{{/if}}

Focus on clarity and logical flow. If the query implies a task (e.g., data extraction), describe how the model might break down the task, identify relevant information, and structure the output.
If the query is about a concept, explain how the model might access and synthesize information.
Output the explanation directly as plain text, suitable for streaming. Do not wrap in JSON or any other structure.
The output must be a single string assigned to the 'thinkingProcess' field of the JSON response.`, 
  config: { 
    temperature: 0.5, 
    safetySettings: [
      { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_NONE' },
      { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_NONE' },
      { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_MEDIUM_AND_ABOVE'},
      { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_NONE'},
    ]
  },
});
