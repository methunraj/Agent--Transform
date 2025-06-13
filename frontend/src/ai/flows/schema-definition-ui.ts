'use server';

/**
 * @fileOverview A schema generator AI agent.
 *
 * - generateSchema - A function that handles the schema generation process.
 * - GenerateSchemaInput - The input type for the generateSchema function.
 * - GenerateSchemaOutput - The return type for the generateSchema function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const GenerateSchemaInputSchema = z.object({
  intent: z.string().describe('The intent of the desired schema.'),
});
export type GenerateSchemaInput = z.infer<typeof GenerateSchemaInputSchema>;

const GenerateSchemaOutputSchema = z.object({
  schema: z.string().describe('The generated JSON schema.'),
});
export type GenerateSchemaOutput = z.infer<typeof GenerateSchemaOutputSchema>;

export async function generateSchema(input: GenerateSchemaInput): Promise<GenerateSchemaOutput> {
  return generateSchemaFlow(input);
}

const prompt = ai.definePrompt({
  name: 'generateSchemaPrompt',
  input: {schema: GenerateSchemaInputSchema},
  output: {schema: GenerateSchemaOutputSchema},
  prompt: `You are an expert JSON schema generator. You will generate a JSON schema based on the user's intent.

Intent: {{{intent}}}

Ensure that the schema is valid JSON and includes descriptions for each field.
`,
});

const generateSchemaFlow = ai.defineFlow(
  {
    name: 'generateSchemaFlow',
    inputSchema: GenerateSchemaInputSchema,
    outputSchema: GenerateSchemaOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);
