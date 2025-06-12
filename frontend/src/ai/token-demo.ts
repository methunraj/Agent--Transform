'use server';

/**
 * This file provides a simple way to test token counting and usage logging.
 * Run it with: npx tsx src/ai/token-demo.ts
 */

import { ai } from './genkit';
import { countTokens } from './genkit';

async function demonstrateTokenCounting() {
  console.log('\n===== TOKEN COUNTING DEMONSTRATION =====\n');
  
  // Test simple string token counting
  const simpleText = 'The quick brown fox jumps over the lazy dog.';
  console.log('Testing with simple text:');
  await countTokens('gemini-1.5-flash-latest', simpleText);
  
  // Test longer text token counting
  const longerText = `
    This is a longer piece of text that might represent a document.
    It has multiple lines and includes various types of content.
    The goal is to see how token counting works with more substantial text.
    
    - Item 1: Some information
    - Item 2: More information
    - Item 3: Additional details
    
    This helps us understand how tokens are counted and reported.
  `;
  console.log('\nTesting with longer text:');
  await countTokens('gemini-1.5-flash-latest', longerText);
  
  // Test token counting with more complex content
  const complexContent = {
    system: 'You are a helpful assistant',
    user: 'Please analyze this document and extract key information',
    document: 'This is a sample document with important data points like 12.5% growth rate and $1.2M revenue'
  };
  console.log('\nTesting with complex content:');
  await countTokens('gemini-1.5-flash-latest', complexContent);
  
  console.log('\n===== DEMONSTRATION COMPLETE =====\n');
}

// Run the demonstration
demonstrateTokenCounting().catch(error => {
  console.error('Error in token counting demonstration:', error);
}); 