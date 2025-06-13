'use client';
import type { Dispatch, SetStateAction } from 'react';
import { createContext, useContext, useState, useMemo, useEffect, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import type { Example } from '@/types';

// Types for the unified configuration
export interface SavedSchema {
  id: string;
  name: string;
  schemaJson: string;
  createdAt: number;
}

export interface SavedPromptSet {
  id: string;
  name: string;
  systemPrompt: string;
  userPromptTemplate: string;
  examples: Example[];
  createdAt: number;
}

export interface LLMConfiguration {
  provider: string;
  model: string;
  apiKey: string;
  temperature: number;
  thinkingBudget?: number;
  pricePerMillionInputTokens?: number;
  pricePerMillionOutputTokens?: number;
  isConfigured: boolean;
  isValid: boolean | null;
}

export interface GenerationInput {
  userIntent: string;
  exampleCount: number;
  includeValidation: boolean;
  includeComprehensiveExamples?: boolean;
}

export interface GenerationResult {
  id: string;
  input: GenerationInput;
  schema: string;
  systemPrompt: string;
  userPromptTemplate: string;
  examples: Example[];
  reasoning?: string;
  confidence?: number;
  timestamp: number;
}

export interface CompleteConfiguration {
  id: string;
  name: string;
  llmConfig: LLMConfiguration;
  schema: string;
  systemPrompt: string;
  userPromptTemplate: string;
  examples: Example[];
  isGenerated: boolean;
  createdAt: number;
}

interface ConfigurationContextType {
  // LLM Configuration
  llmConfig: LLMConfiguration;
  updateLLMConfig: (updates: Partial<LLMConfiguration>) => void;
  validateLLMConnection: () => Promise<boolean>;
  
  // AI Generation
  isGenerating: boolean;
  generationHistory: GenerationResult[];
  generateFromPrompt: (input: GenerationInput) => Promise<void>;
  clearGenerationHistory: () => void;
  
  // Schema Management
  schemaJson: string;
  setSchemaJson: Dispatch<SetStateAction<string>>;
  savedSchemas: SavedSchema[];
  saveSchema: (name: string) => void;
  loadSchema: (id: string) => void;
  deleteSchema: (id: string) => void;
  isSchemaGenerated: boolean;
  
  // Prompt Management
  systemPrompt: string;
  setSystemPrompt: Dispatch<SetStateAction<string>>;
  userPromptTemplate: string;
  setUserPromptTemplate: Dispatch<SetStateAction<string>>;
  examples: Example[];
  setExamples: Dispatch<SetStateAction<Example[]>>;
  savedPromptSets: SavedPromptSet[];
  savePromptSet: (name: string) => void;
  loadPromptSet: (id: string) => void;
  deletePromptSet: (id: string) => void;
  arePromptsGenerated: boolean;
  
  // Unified Configuration Management
  completeConfigurations: CompleteConfiguration[];
  saveCompleteConfiguration: (name: string) => void;
  loadCompleteConfiguration: (id: string) => void;
  deleteCompleteConfiguration: (id: string) => void;
  resetConfiguration: () => void;
  isConfigurationComplete: boolean;
}

// Default values
const defaultLLMConfig: LLMConfiguration = {
  provider: 'googleAI',
  model: 'gemini-2.5-flash-preview-05-20',
  apiKey: '',
  temperature: 0.3,
  thinkingBudget: undefined,
  pricePerMillionInputTokens: undefined,
  pricePerMillionOutputTokens: undefined,
  isConfigured: false,
  isValid: null,
};

// Function to get predefined examples for various document types
const getPredefinedExamples = (): Example[] => [
  // Example 1: Invoice
  {
    input: `INVOICE

Acme Corp
123 Business Ave
San Francisco, CA 94107

Bill To:
Tech Solutions Inc.
456 Innovation Drive
Palo Alto, CA 94301

Invoice #: INV-2023-0456
Date: 2023-09-15
Due Date: 2023-10-15

Description                   Quantity    Unit Price    Amount
--------------------------------------------------------------
Cloud Storage (500GB)             1        $99.99       $99.99
Premium Support Package           1       $199.99      $199.99
API Access Tokens              100         $0.50       $50.00
--------------------------------------------------------------
Subtotal                                              $349.98
Tax (8.5%)                                             $29.75
Total                                                 $379.73

Payment Terms: Net 30
Payment Methods: Bank Transfer, Credit Card

Thank you for your business!
`,
    output: JSON.stringify({
      "documentType": "invoice",
      "title": "Acme Corp Invoice",
      "date": "2023-09-15",
      "metadata": {
        "author": "Acme Corp",
        "recipient": "Tech Solutions Inc.",
        "documentId": "INV-2023-0456",
        "creationDate": "2023-09-15"
      },
      "financialData": {
        "totalAmount": 379.73,
        "currency": "USD",
        "taxAmount": 29.75,
        "lineItems": [
          {
            "description": "Cloud Storage (500GB)",
            "quantity": 1,
            "unitPrice": 99.99,
            "amount": 99.99
          },
          {
            "description": "Premium Support Package",
            "quantity": 1,
            "unitPrice": 199.99,
            "amount": 199.99
          },
          {
            "description": "API Access Tokens",
            "quantity": 100,
            "unitPrice": 0.50,
            "amount": 50.00
          }
        ],
        "paymentTerms": "Net 30"
      },
      "contentData": {
        "summary": "Invoice from Acme Corp to Tech Solutions Inc. for cloud services and support.",
        "keywords": ["invoice", "cloud storage", "support package", "API tokens"]
      },
      "entities": [
        {
          "name": "Acme Corp",
          "type": "organization",
          "role": "sender"
        },
        {
          "name": "Tech Solutions Inc.",
          "type": "organization",
          "role": "recipient"
        }
      ]
    }, null, 2)
  },
  // Example 2: Technical Article
  {
    input: `# Understanding Machine Learning Fundamentals

By Dr. Sarah Johnson
Published: June 12, 2023

## Introduction

Machine Learning (ML) has become an essential tool in modern technology. This article provides an overview of key ML concepts for beginners and intermediate practitioners.

## Supervised Learning

Supervised learning involves training a model on labeled data. The algorithm learns to map inputs to outputs based on example input-output pairs. Common applications include:

- Classification (predicting categories)
- Regression (predicting continuous values)
- Recommendation systems

## Unsupervised Learning

Unsupervised learning works with unlabeled data, finding patterns or structures without explicit guidance. Key techniques include:

- Clustering (grouping similar data points)
- Dimensionality reduction (simplifying data while preserving information)
- Anomaly detection (identifying outliers)

## Key Algorithms

1. **Linear Regression**: Predicts continuous values using a linear approach
2. **Decision Trees**: Tree-like models for classification and regression
3. **Neural Networks**: Inspired by biological neural networks, powerful for complex patterns
4. **K-means**: Clustering algorithm that partitions data into k clusters

## Conclusion

Understanding these fundamentals provides a solid foundation for more advanced ML concepts. In future articles, we'll explore model evaluation, hyperparameter tuning, and ethical considerations in AI.

## References

1. Mitchell, T. (1997). Machine Learning. McGraw Hill.
2. Goodfellow, I., et al. (2016). Deep Learning. MIT Press.
`,
    output: JSON.stringify({
      "documentType": "article",
      "title": "Understanding Machine Learning Fundamentals",
      "date": "2023-06-12",
      "metadata": {
        "author": "Dr. Sarah Johnson",
        "recipient": null,
        "documentId": null,
        "creationDate": "2023-06-12"
      },
      "contentData": {
        "summary": "An overview of machine learning concepts including supervised learning, unsupervised learning, and key algorithms for beginners and intermediate practitioners.",
        "keywords": ["machine learning", "supervised learning", "unsupervised learning", "algorithms", "classification", "regression", "clustering"]
      },
      "entities": [
        {
          "name": "Dr. Sarah Johnson",
          "type": "person",
          "role": "author"
        },
        {
          "name": "Mitchell, T.",
          "type": "person",
          "role": "referenced author"
        },
        {
          "name": "Goodfellow, I.",
          "type": "person",
          "role": "referenced author"
        }
      ],
      "financialData": null
    }, null, 2)
  },
  // Example 3: Business Email
  {
    input: `From: michael.chen@techvision.com
To: sarah.patel@innovatech.org
Subject: Partnership Proposal for Q4 2023
Date: August 28, 2023, 10:15 AM

Dear Sarah,

I hope this email finds you well. Following our conversation at the Tech Summit last month, I'm reaching out to formally propose a strategic partnership between TechVision and InnovaTech for Q4 2023.

Proposed Collaboration Points:

1. Joint webinar series on emerging AI technologies (October-November)
2. Co-developed white paper on industry applications
3. Shared booth at the December Tech Expo in San Francisco

Budget Considerations:
- Webinar platform and promotion: $5,000 (split equally)
- White paper research and production: $8,000 (TechVision: 60%, InnovaTech: 40%)
- Expo presence: $12,000 (split equally)

I believe this partnership would benefit both organizations by expanding our market reach and positioning us as thought leaders in the AI space. Our preliminary market analysis suggests we could reach approximately 15,000 new potential clients through these combined efforts.

Please let me know if you'd like to discuss this proposal further. I'm available for a call next Tuesday or Wednesday afternoon.

Best regards,

Michael Chen
Business Development Director
TechVision Inc.
Phone: (555) 123-4567
`,
    output: JSON.stringify({
      "documentType": "email",
      "title": "Partnership Proposal for Q4 2023",
      "date": "2023-08-28",
      "metadata": {
        "author": "Michael Chen",
        "recipient": "Sarah Patel",
        "documentId": null,
        "creationDate": "2023-08-28"
      },
      "financialData": {
        "totalAmount": 25000,
        "currency": "USD",
        "lineItems": [
          {
            "description": "Webinar platform and promotion",
            "amount": 5000
          },
          {
            "description": "White paper research and production",
            "amount": 8000
          },
          {
            "description": "Expo presence",
            "amount": 12000
          }
        ],
        "paymentTerms": null
      },
      "contentData": {
        "summary": "A partnership proposal from TechVision to InnovaTech for Q4 2023, including joint webinars, a co-developed white paper, and shared expo presence with associated budget considerations.",
        "keywords": ["partnership", "proposal", "webinar", "white paper", "expo", "budget", "AI technologies"]
      },
      "entities": [
        {
          "name": "Michael Chen",
          "type": "person",
          "role": "sender"
        },
        {
          "name": "Sarah Patel",
          "type": "person",
          "role": "recipient"
        },
        {
          "name": "TechVision",
          "type": "organization",
          "role": "sender organization"
        },
        {
          "name": "InnovaTech",
          "type": "organization",
          "role": "recipient organization"
        },
        {
          "name": "Tech Expo",
          "type": "event",
          "role": "mentioned event"
        }
      ]
    }, null, 2)
  }
];

const defaultSchema = JSON.stringify(
  {
    $schema: 'http://json-schema.org/draft-07/schema#',
    title: 'ExtractedData',
    description: 'Schema for comprehensive data extraction from various document types.',
    type: 'object',
    properties: {
      documentType: {
        type: 'string',
        description: 'The type of document (e.g., invoice, report, article, email, contract, resume)',
        enum: ['invoice', 'financial_report', 'article', 'email', 'contract', 'letter', 'resume', 'technical_document', 'legal_document', 'medical_record', 'other']
      },
      title: {
        type: 'string',
        description: 'The title or heading of the document',
      },
      date: {
        type: ['string', 'null'],
        format: 'date',
        description: 'The main date mentioned in the document (YYYY-MM-DD format if possible)',
      },
      metadata: {
        type: 'object',
        description: 'Additional metadata about the document',
        properties: {
          author: {
            type: ['string', 'null'],
            description: 'Author or sender of the document'
          },
          recipient: {
            type: ['string', 'null'],
            description: 'Intended recipient of the document'
          },
          documentId: {
            type: ['string', 'null'],
            description: 'Any reference or ID number in the document'
          },
          creationDate: {
            type: ['string', 'null'],
            format: 'date',
            description: 'When the document was created (if different from main date)'
          }
        }
      },
      financialData: {
        type: 'object',
        description: 'Financial information if present in the document',
        properties: {
          totalAmount: {
            type: ['number', 'null'],
            description: 'Total amount mentioned (e.g., invoice total)'
          },
          currency: {
            type: ['string', 'null'],
            description: 'Currency code or symbol (USD, EUR, $, etc.)'
          },
          taxAmount: {
            type: ['number', 'null'],
            description: 'Tax amount if specified'
          },
          lineItems: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                description: { type: 'string' },
                quantity: { type: ['number', 'null'] },
                unitPrice: { type: ['number', 'null'] },
                amount: { type: ['number', 'null'] }
              }
            },
            description: 'Individual line items in an invoice or financial document'
          },
          paymentTerms: {
            type: ['string', 'null'],
            description: 'Payment terms or due date information'
          }
        }
      },
      contentData: {
        type: 'object',
        description: 'Structured content information from the document',
        properties: {
          sections: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                title: { type: ['string', 'null'] },
                content: { type: 'string' }
              }
            },
            description: 'Major sections or parts of the document'
          },
          tables: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                title: { type: ['string', 'null'] },
                headers: { 
                  type: 'array',
                  items: { type: 'string' }
                },
                data: {
                  type: 'array',
                  items: {
                    type: 'array',
                    items: { type: 'string' }
                  }
                }
              }
            },
            description: 'Tables found in the document'
          }
        }
      },
      summary: {
        type: 'string',
        description: 'A comprehensive summary of the document content',
      },
      keywords: {
        type: 'array',
        items: {
          type: 'string'
        },
        description: 'A list of keywords from the document'
      },
      entities: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            name: { type: 'string' },
            type: { type: 'string' },
            value: { type: 'string' }
          }
        },
        description: 'Named entities mentioned in the document (people, organizations, locations, etc.)'
      }
    },
    required: ['documentType', 'title', 'summary'],
  },
  null,
  2
);

const defaultSystemPrompt = `You are a precise data extraction assistant specialized in analyzing various document types including invoices, financial reports, articles, emails, contracts, resumes, technical documents, and more.

Your task is to extract structured information from documents according to the provided schema. Return the data in the requested format.

Guidelines for extraction:
1. First identify the document type to guide your extraction strategy
2. Extract all relevant fields based on the document content and structure
3. For financial documents, pay special attention to monetary values, dates, line items, and payment terms
4. For text-heavy documents like articles or reports, identify key sections, tables, and entities
5. For correspondence like emails or letters, capture sender, recipient, and key message points
6. For technical or legal documents, focus on structured sections, definitions, and specialized terminology
7. If information for a field is not available in the document, use null for that field
8. Maintain the exact structure of the provided schema
9. Format dates in YYYY-MM-DD format when possible
10. Extract numerical values as numbers, not strings, for financial fields
11. Identify and extract tables with their headers and data when present
12. Recognize named entities (people, organizations, locations) throughout the document

Focus solely on extracting data as per the schema. Do not add any conversational text or explanations outside of the JSON output.`;

const defaultUserPromptTemplate = `Based on the provided document content and the JSON schema, please extract the relevant information in a highly structured format.

Document Content will be provided by the system (using {{document_content_text}} or {{media url=document_media_url}}).
JSON Schema will be provided by the system (using {{json_schema_text}}).
{{#if examples_list.length}}
Here are some examples:
{{#each examples_list}}
---
Input: {{{this.input}}}
Output: {{{this.output}}}
---
{{/each}}
{{/if}}

Your task is to:
1. Analyze the document thoroughly to understand its type and content structure
2. Classify the document into one of the document types defined in the schema
3. Extract all relevant information according to the schema, focusing on:
   - Document type-specific fields (financial data for invoices, content structure for articles, etc.)
   - Metadata (authors, recipients, dates, reference numbers)
   - Content organization (sections, tables, lists)
   - Named entities (people, organizations, locations)
4. Structure the extracted data precisely according to the schema
5. Ensure all required fields are populated
6. Use null for optional fields where information is not present
7. Format dates, numbers, and other data types according to schema specifications

Return the extracted data in the requested format. The output should be complete and well-structured.`;

const ConfigurationContext = createContext<ConfigurationContextType | undefined>(undefined);

export function ConfigurationProvider({ children }: { children: React.ReactNode }) {
  // LLM Configuration State
  const [llmConfig, setLLMConfig] = useState<LLMConfiguration>(defaultLLMConfig);
  
  // AI Generation State
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationHistory, setGenerationHistory] = useState<GenerationResult[]>([]);
  
  // Schema State
  const [schemaJson, setSchemaJson] = useState<string>(defaultSchema);
  const [savedSchemas, setSavedSchemas] = useState<SavedSchema[]>([]);
  const [isSchemaGenerated, setIsSchemaGenerated] = useState(false);
  
  // Predefined examples for different document types
  const getPredefinedExamples = (): Example[] => [
    // Example 1: Invoice
    {
      input: `INVOICE
Invoice #: INV-2023-0456
Date: May 15, 2023

From:
TechSupplies Inc.
123 Business Ave
San Francisco, CA 94107
Tax ID: 98-7654321

To:
Acme Corporation
789 Corporate Blvd
New York, NY 10001

Description                 Qty    Unit Price    Amount
------------------------------------------------------
Dell XPS 15 Laptop          2      $1,899.99     $3,799.98
HP LaserJet Pro Printer     1      $349.99       $349.99
USB-C Docking Station       2      $129.99       $259.98

Subtotal:                                        $4,409.95
Tax (8.875%):                                    $391.38
Total:                                           $4,801.33

Payment Terms: Net 30
Due Date: June 14, 2023

Please make payment to:
Bank: First National Bank
Account: 987-654321
Routing: 123456789`,

      output: `{
  "documentType": "invoice",
  "title": "TechSupplies Inc. Invoice #INV-2023-0456",
  "date": "2023-05-15",
  "metadata": {
    "author": "TechSupplies Inc.",
    "recipient": "Acme Corporation",
    "documentId": "INV-2023-0456",
    "creationDate": "2023-05-15"
  },
  "financialData": {
    "totalAmount": 4801.33,
    "currency": "USD",
    "taxAmount": 391.38,
    "lineItems": [
      {
        "description": "Dell XPS 15 Laptop",
        "quantity": 2,
        "unitPrice": 1899.99,
        "amount": 3799.98
      },
      {
        "description": "HP LaserJet Pro Printer",
        "quantity": 1,
        "unitPrice": 349.99,
        "amount": 349.99
      },
      {
        "description": "USB-C Docking Station",
        "quantity": 2,
        "unitPrice": 129.99,
        "amount": 259.98
      }
    ],
    "paymentTerms": "Net 30, Due Date: June 14, 2023"
  },
  "contentData": {
    "sections": [],
    "tables": [
      {
        "title": "Invoice Items",
        "headers": ["Description", "Qty", "Unit Price", "Amount"],
        "data": [
          ["Dell XPS 15 Laptop", "2", "$1,899.99", "$3,799.98"],
          ["HP LaserJet Pro Printer", "1", "$349.99", "$349.99"],
          ["USB-C Docking Station", "2", "$129.99", "$259.98"]
        ]
      }
    ]
  },
  "summary": "Invoice from TechSupplies Inc. to Acme Corporation for computer equipment including laptops, printer, and docking stations totaling $4,801.33 with payment due by June 14, 2023.",
  "keywords": ["invoice", "TechSupplies Inc.", "Acme Corporation", "computer equipment", "payment"],
  "entities": [
    {
      "name": "TechSupplies Inc.",
      "type": "organization",
      "value": "sender"
    },
    {
      "name": "Acme Corporation",
      "type": "organization",
      "value": "recipient"
    },
    {
      "name": "First National Bank",
      "type": "organization",
      "value": "payment recipient"
    }
  ]
}`
    },
    // Example 2: Technical Article
    {
      input: `# Introduction to Machine Learning

Author: Dr. Sarah Johnson
Published: March 10, 2023
Journal: AI Quarterly Review

## Abstract
This article provides an overview of machine learning concepts, techniques, and applications. It is intended for beginners who want to understand the fundamentals of this rapidly evolving field.

## 1. What is Machine Learning?
Machine Learning (ML) is a subset of artificial intelligence that focuses on developing systems that can learn from and make decisions based on data. Unlike traditional programming, where explicit instructions are provided, ML algorithms build models based on sample data to make predictions or decisions without being explicitly programmed to do so.

## 2. Types of Machine Learning

### 2.1 Supervised Learning
In supervised learning, algorithms are trained using labeled data. The algorithm learns to map inputs to outputs based on example input-output pairs. Common applications include:
- Classification (e.g., spam detection)
- Regression (e.g., price prediction)

### 2.2 Unsupervised Learning
Unsupervised learning algorithms work with unlabeled data. They identify patterns and relationships within the data without prior training. Applications include:
- Clustering (e.g., customer segmentation)
- Dimensionality reduction

### 2.3 Reinforcement Learning
Reinforcement learning involves training algorithms to make sequences of decisions by rewarding desired behaviors and punishing undesired ones. It's commonly used in:
- Game playing
- Robotics
- Autonomous vehicles

## 3. Common Algorithms

| Algorithm Type | Examples | Common Applications |
|----------------|----------|---------------------|
| Classification | Decision Trees, Random Forest, SVM | Spam detection, Image recognition |
| Regression | Linear Regression, Ridge Regression | Price prediction, Risk assessment |
| Clustering | K-means, Hierarchical Clustering | Customer segmentation, Anomaly detection |

## Conclusion
Machine learning continues to transform industries and drive innovation. As computing power increases and algorithms improve, we can expect even more sophisticated applications in the future.

## References
1. Mitchell, T. (1997). Machine Learning. McGraw Hill.
2. Goodfellow, I., et al. (2016). Deep Learning. MIT Press.
3. Russell, S., & Norvig, P. (2020). Artificial Intelligence: A Modern Approach. Pearson.`,

      output: `{
  "documentType": "article",
  "title": "Introduction to Machine Learning",
  "date": "2023-03-10",
  "metadata": {
    "author": "Dr. Sarah Johnson",
    "recipient": null,
    "documentId": null,
    "creationDate": "2023-03-10"
  },
  "financialData": {
    "totalAmount": null,
    "currency": null,
    "taxAmount": null,
    "lineItems": [],
    "paymentTerms": null
  },
  "contentData": {
    "sections": [
      {
        "title": "Abstract",
        "content": "This article provides an overview of machine learning concepts, techniques, and applications. It is intended for beginners who want to understand the fundamentals of this rapidly evolving field."
      },
      {
        "title": "1. What is Machine Learning?",
        "content": "Machine Learning (ML) is a subset of artificial intelligence that focuses on developing systems that can learn from and make decisions based on data. Unlike traditional programming, where explicit instructions are provided, ML algorithms build models based on sample data to make predictions or decisions without being explicitly programmed to do so."
      },
      {
        "title": "2. Types of Machine Learning",
        "content": "Includes supervised learning, unsupervised learning, and reinforcement learning sections with their applications."
      },
      {
        "title": "3. Common Algorithms",
        "content": "Describes various algorithm types, examples, and their common applications."
      },
      {
        "title": "Conclusion",
        "content": "Machine learning continues to transform industries and drive innovation. As computing power increases and algorithms improve, we can expect even more sophisticated applications in the future."
      }
    ],
    "tables": [
      {
        "title": "Common Algorithms",
        "headers": ["Algorithm Type", "Examples", "Common Applications"],
        "data": [
          ["Classification", "Decision Trees, Random Forest, SVM", "Spam detection, Image recognition"],
          ["Regression", "Linear Regression, Ridge Regression", "Price prediction, Risk assessment"],
          ["Clustering", "K-means, Hierarchical Clustering", "Customer segmentation, Anomaly detection"]
        ]
      }
    ]
  },
  "summary": "An introductory article on machine learning written by Dr. Sarah Johnson and published in AI Quarterly Review on March 10, 2023. The article covers fundamental concepts of machine learning, including types (supervised, unsupervised, reinforcement learning) and common algorithms with their applications.",
  "keywords": ["machine learning", "artificial intelligence", "supervised learning", "unsupervised learning", "reinforcement learning", "algorithms"],
  "entities": [
    {
      "name": "Dr. Sarah Johnson",
      "type": "person",
      "value": "author"
    },
    {
      "name": "AI Quarterly Review",
      "type": "organization",
      "value": "publisher"
    },
    {
      "name": "Mitchell, T.",
      "type": "person",
      "value": "reference author"
    },
    {
      "name": "Goodfellow, I.",
      "type": "person",
      "value": "reference author"
    },
    {
      "name": "Russell, S.",
      "type": "person",
      "value": "reference author"
    },
    {
      "name": "Norvig, P.",
      "type": "person",
      "value": "reference author"
    }
  ]
}`
    },
    // Example 3: Email
    {
      input: `From: john.smith@company.com
To: sarah.jones@client.org
CC: michael.brown@company.com, legal@company.com
Subject: Proposal for Website Redesign Project
Date: April 5, 2023

Dear Sarah,

I hope this email finds you well. Following our meeting last week, I'm pleased to submit our formal proposal for the redesign of client.org.

As discussed, our proposal includes:

1. Complete redesign of the main website with responsive design
2. Integration with your existing CMS
3. SEO optimization
4. Analytics setup and dashboard
5. Training for your team

The project timeline is estimated at 12 weeks from kickoff to launch, with the following cost breakdown:

- Design phase: $15,000
- Development: $35,000
- Testing and QA: $8,000
- Training and documentation: $7,000
- Total project cost: $65,000

We propose a payment schedule as follows:
- 30% upon project kickoff
- 30% at design approval
- 30% at development completion
- 10% upon final delivery

Please review the attached detailed proposal document and let me know if you have any questions or would like to discuss any aspects further.

We're excited about the opportunity to work with Client Organization and believe we can deliver a website that will significantly improve user engagement and conversion rates.

Best regards,

John Smith
Senior Project Manager
Company Solutions Inc.
Phone: (555) 123-4567
www.company.com`,

      output: `{
  "documentType": "email",
  "title": "Proposal for Website Redesign Project",
  "date": "2023-04-05",
  "metadata": {
    "author": "John Smith",
    "recipient": "Sarah Jones",
    "documentId": null,
    "creationDate": "2023-04-05"
  },
  "financialData": {
    "totalAmount": 65000,
    "currency": "USD",
    "taxAmount": null,
    "lineItems": [
      {
        "description": "Design phase",
        "quantity": null,
        "unitPrice": null,
        "amount": 15000
      },
      {
        "description": "Development",
        "quantity": null,
        "unitPrice": null,
        "amount": 35000
      },
      {
        "description": "Testing and QA",
        "quantity": null,
        "unitPrice": null,
        "amount": 8000
      },
      {
        "description": "Training and documentation",
        "quantity": null,
        "unitPrice": null,
        "amount": 7000
      }
    ],
    "paymentTerms": "30% upon project kickoff, 30% at design approval, 30% at development completion, 10% upon final delivery"
  },
  "contentData": {
    "sections": [
      {
        "title": "Proposal Items",
        "content": "1. Complete redesign of the main website with responsive design\n2. Integration with your existing CMS\n3. SEO optimization\n4. Analytics setup and dashboard\n5. Training for your team"
      },
      {
        "title": "Project Timeline",
        "content": "12 weeks from kickoff to launch"
      },
      {
        "title": "Payment Schedule",
        "content": "- 30% upon project kickoff\n- 30% at design approval\n- 30% at development completion\n- 10% upon final delivery"
      }
    ],
    "tables": [
      {
        "title": "Cost Breakdown",
        "headers": ["Item", "Cost"],
        "data": [
          ["Design phase", "$15,000"],
          ["Development", "$35,000"],
          ["Testing and QA", "$8,000"],
          ["Training and documentation", "$7,000"],
          ["Total project cost", "$65,000"]
        ]
      }
    ]
  },
  "summary": "Email from John Smith at Company Solutions Inc. to Sarah Jones at client.org regarding a proposal for website redesign project. The proposal outlines a 12-week project with a total cost of $65,000, covering design, development, testing, and training phases with a structured payment schedule.",
  "keywords": ["proposal", "website redesign", "project", "cost", "timeline", "payment schedule"],
  "entities": [
    {
      "name": "John Smith",
      "type": "person",
      "value": "sender"
    },
    {
      "name": "Sarah Jones",
      "type": "person",
      "value": "primary recipient"
    },
    {
      "name": "Michael Brown",
      "type": "person",
      "value": "cc recipient"
    },
    {
      "name": "Company Solutions Inc.",
      "type": "organization",
      "value": "sender company"
    },
    {
      "name": "Client Organization",
      "type": "organization",
      "value": "client"
    }
  ]
}`
    }
  ];

  // Prompt State
  const [systemPrompt, setSystemPrompt] = useState<string>(defaultSystemPrompt);
  const [userPromptTemplate, setUserPromptTemplate] = useState<string>(defaultUserPromptTemplate);
  const [examples, setExamples] = useState<Example[]>([]);
  const [savedPromptSets, setSavedPromptSets] = useState<SavedPromptSet[]>([]);
  const [arePromptsGenerated, setArePromptsGenerated] = useState(false);
  
  // Complete Configurations State
  const [completeConfigurations, setCompleteConfigurations] = useState<CompleteConfiguration[]>([]);

  // Load saved data on mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      // Load LLM config
      const savedLLMConfig = localStorage.getItem('intelliextract_llm_config');
      if (savedLLMConfig) {
        try {
          const parsed = JSON.parse(savedLLMConfig);
          setLLMConfig(prev => ({ ...prev, ...parsed }));
        } catch (e) {
          console.error('Error loading LLM config:', e);
        }
      }
      
      // Load schemas
      const savedSchemasData = localStorage.getItem('intelliextract_savedSchemas');
      if (savedSchemasData) {
        try {
          setSavedSchemas(JSON.parse(savedSchemasData));
        } catch (e) {
          console.error('Error loading schemas:', e);
        }
      }
      
      // Load prompt sets
      const savedPromptSetsData = localStorage.getItem('intelliextract_savedPromptSets');
      if (savedPromptSetsData) {
        try {
          setSavedPromptSets(JSON.parse(savedPromptSetsData));
        } catch (e) {
          console.error('Error loading prompt sets:', e);
        }
      }
      
      // Load complete configurations
      const savedCompleteConfigs = localStorage.getItem('intelliextract_completeConfigurations');
      if (savedCompleteConfigs) {
        try {
          setCompleteConfigurations(JSON.parse(savedCompleteConfigs));
        } catch (e) {
          console.error('Error loading complete configurations:', e);
        }
      }
      
      // Load generation history
      const savedGenerationHistory = localStorage.getItem('intelliextract_generationHistory');
      if (savedGenerationHistory) {
        try {
          setGenerationHistory(JSON.parse(savedGenerationHistory));
        } catch (e) {
          console.error('Error loading generation history:', e);
        }
      }
    }
  }, []);

  // Save data to localStorage when state changes
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('intelliextract_llm_config', JSON.stringify(llmConfig));
    }
  }, [llmConfig]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('intelliextract_savedSchemas', JSON.stringify(savedSchemas));
    }
  }, [savedSchemas]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('intelliextract_savedPromptSets', JSON.stringify(savedPromptSets));
    }
  }, [savedPromptSets]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('intelliextract_completeConfigurations', JSON.stringify(completeConfigurations));
    }
  }, [completeConfigurations]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('intelliextract_generationHistory', JSON.stringify(generationHistory));
    }
  }, [generationHistory]);

  // LLM Configuration Methods
  const updateLLMConfig = useCallback((updates: Partial<LLMConfiguration>) => {
    setLLMConfig(prev => {
      const updated = { ...prev, ...updates };
      // Auto-determine if configured
      updated.isConfigured = !!(updated.model && (updated.apiKey || updated.provider === 'googleAI'));
      return updated;
    });
  }, []);

  const validateLLMConnection = useCallback(async (): Promise<boolean> => {
    // Simple validation - in a real app you'd test the actual connection
    const isValid = llmConfig.model && (llmConfig.apiKey.length > 15 || llmConfig.provider === 'googleAI');
    setLLMConfig(prev => ({ ...prev, isValid }));
    return isValid;
  }, [llmConfig.model, llmConfig.apiKey, llmConfig.provider]);

  // AI Generation Methods
  const generateFromPrompt = useCallback(async (input: GenerationInput) => {
    if (!llmConfig.isConfigured) {
      throw new Error('LLM must be configured before generating content');
    }

    setIsGenerating(true);
    try {
      // Use the actual AI generation flow
      const { generateUnifiedConfiguration } = await import('@/ai/flows/unified-generation-flow');
      
      const generationInput = {
        userIntent: input.userIntent,
        exampleCount: input.exampleCount,
        llmProvider: llmConfig.provider,
        modelName: llmConfig.model,
        temperature: llmConfig.temperature,
      };
      
      const result = await generateUnifiedConfiguration(generationInput);

      // Create a comprehensive set of examples by combining AI-generated examples
      // with our predefined examples for various document types
      const combinedExamples = [...result.examples];
      
      // Add predefined examples if requested or if AI didn't generate enough examples
      if (input.includeComprehensiveExamples || combinedExamples.length < 2) {
        // Get predefined examples and add them to the mix
        const predefinedExamples = getPredefinedExamples();
        
        // Add examples that are relevant to the user's intent
        // This is a simple relevance check - in a real app, you might use more sophisticated matching
        const userIntentLower = input.userIntent.toLowerCase();
        predefinedExamples.forEach(example => {
          // Check if this example might be relevant to the user's intent
          const isRelevant = example.output.toLowerCase().includes(userIntentLower) ||
                            userIntentLower.includes('comprehensive') ||
                            userIntentLower.includes('all types') ||
                            userIntentLower.includes('various');
          
          if (isRelevant && combinedExamples.length < 5) { // Limit to 5 total examples
            combinedExamples.push(example);
          }
        });
      }

      const generatedResult: GenerationResult = {
        id: uuidv4(),
        input,
        schema: result.schema,
        systemPrompt: result.systemPrompt,
        userPromptTemplate: result.userPromptTemplate,
        examples: combinedExamples,
        reasoning: result.reasoning,
        confidence: result.confidence,
        timestamp: Date.now(),
      };

      // Update state with generated content
      setSchemaJson(generatedResult.schema);
      setSystemPrompt(generatedResult.systemPrompt);
      setUserPromptTemplate(generatedResult.userPromptTemplate);
      setExamples(generatedResult.examples);
      
      // Mark as generated
      setIsSchemaGenerated(true);
      setArePromptsGenerated(true);
      
      // Add to history
      setGenerationHistory(prev => [generatedResult, ...prev]);
      
    } catch (error) {
      console.error('Generation failed:', error);
      throw error;
    } finally {
      setIsGenerating(false);
    }
  }, [llmConfig.isConfigured, llmConfig.provider, llmConfig.model, llmConfig.temperature]);

  const clearGenerationHistory = useCallback(() => {
    setGenerationHistory([]);
  }, []);

  // Schema Management Methods
  const saveSchema = useCallback((name: string) => {
    if (!name.trim()) {
      throw new Error("Schema name cannot be empty.");
    }
    if (savedSchemas.some(s => s.name === name.trim())) {
      throw new Error(`Schema with name "${name.trim()}" already exists.`);
    }
    const newSchema: SavedSchema = { 
      id: uuidv4(), 
      name: name.trim(), 
      schemaJson, 
      createdAt: Date.now() 
    };
    setSavedSchemas(prev => [...prev, newSchema].sort((a,b) => a.name.localeCompare(b.name)));
  }, [schemaJson, savedSchemas]);

  const loadSchema = useCallback((id: string) => {
    const schema = savedSchemas.find(s => s.id === id);
    if (schema) {
      setSchemaJson(schema.schemaJson);
      setIsSchemaGenerated(false);
    }
  }, [savedSchemas]);

  const deleteSchema = useCallback((id: string) => {
    setSavedSchemas(prev => prev.filter(s => s.id !== id));
  }, []);

  // Prompt Management Methods
  const savePromptSet = useCallback((name: string) => {
    if (!name.trim()) {
      throw new Error("Prompt set name cannot be empty.");
    }
    if (savedPromptSets.some(ps => ps.name === name.trim())) {
      throw new Error(`Prompt set with name "${name.trim()}" already exists.`);
    }
    const newSet: SavedPromptSet = {
      id: uuidv4(),
      name: name.trim(),
      systemPrompt,
      userPromptTemplate,
      examples,
      createdAt: Date.now(),
    };
    setSavedPromptSets(prev => [...prev, newSet].sort((a,b) => a.name.localeCompare(b.name)));
  }, [systemPrompt, userPromptTemplate, examples, savedPromptSets]);

  const loadPromptSet = useCallback((id: string) => {
    const promptSet = savedPromptSets.find(ps => ps.id === id);
    if (promptSet) {
      setSystemPrompt(promptSet.systemPrompt);
      setUserPromptTemplate(promptSet.userPromptTemplate);
      setExamples(promptSet.examples);
      setArePromptsGenerated(false);
    }
  }, [savedPromptSets]);

  const deletePromptSet = useCallback((id: string) => {
    setSavedPromptSets(prev => prev.filter(ps => ps.id !== id));
  }, []);

  // Complete Configuration Management
  const saveCompleteConfiguration = useCallback((name: string) => {
    if (!name.trim()) {
      throw new Error("Configuration name cannot be empty.");
    }
    if (completeConfigurations.some(c => c.name === name.trim())) {
      throw new Error(`Configuration with name "${name.trim()}" already exists.`);
    }
    const newConfig: CompleteConfiguration = {
      id: uuidv4(),
      name: name.trim(),
      llmConfig,
      schema: schemaJson,
      systemPrompt,
      userPromptTemplate,
      examples,
      isGenerated: isSchemaGenerated && arePromptsGenerated,
      createdAt: Date.now(),
    };
    setCompleteConfigurations(prev => [...prev, newConfig].sort((a,b) => a.name.localeCompare(b.name)));
  }, [llmConfig, schemaJson, systemPrompt, userPromptTemplate, examples, isSchemaGenerated, arePromptsGenerated, completeConfigurations]);

  const loadCompleteConfiguration = useCallback((id: string) => {
    const config = completeConfigurations.find(c => c.id === id);
    if (config) {
      setLLMConfig(config.llmConfig);
      setSchemaJson(config.schema);
      setSystemPrompt(config.systemPrompt);
      setUserPromptTemplate(config.userPromptTemplate);
      setExamples(config.examples);
      setIsSchemaGenerated(config.isGenerated);
      setArePromptsGenerated(config.isGenerated);
    }
  }, [completeConfigurations]);

  const deleteCompleteConfiguration = useCallback((id: string) => {
    setCompleteConfigurations(prev => prev.filter(c => c.id !== id));
  }, []);

  const resetConfiguration = useCallback(() => {
    setSchemaJson(defaultSchema);
    setSystemPrompt(defaultSystemPrompt);
    setUserPromptTemplate(defaultUserPromptTemplate);
    setExamples([]);
    setIsSchemaGenerated(false);
    setArePromptsGenerated(false);
  }, []);

  // Computed properties
  const isConfigurationComplete = useMemo(() => {
    return !!(
      llmConfig.isConfigured &&
      schemaJson &&
      systemPrompt &&
      userPromptTemplate
    );
  }, [llmConfig.isConfigured, schemaJson, systemPrompt, userPromptTemplate]);

  const value = useMemo(() => ({
    // LLM Configuration
    llmConfig,
    updateLLMConfig,
    validateLLMConnection,
    
    // AI Generation
    isGenerating,
    generationHistory,
    generateFromPrompt,
    clearGenerationHistory,
    
    // Schema Management
    schemaJson,
    setSchemaJson,
    savedSchemas,
    saveSchema,
    loadSchema,
    deleteSchema,
    isSchemaGenerated,
    
    // Prompt Management
    systemPrompt,
    setSystemPrompt,
    userPromptTemplate,
    setUserPromptTemplate,
    examples,
    setExamples,
    savedPromptSets,
    savePromptSet,
    loadPromptSet,
    deletePromptSet,
    arePromptsGenerated,
    
    // Unified Configuration Management
    completeConfigurations,
    saveCompleteConfiguration,
    loadCompleteConfiguration,
    deleteCompleteConfiguration,
    resetConfiguration,
    isConfigurationComplete,
  }), [
    llmConfig, updateLLMConfig, validateLLMConnection,
    isGenerating, generationHistory, generateFromPrompt, clearGenerationHistory,
    schemaJson, savedSchemas, saveSchema, loadSchema, deleteSchema, isSchemaGenerated,
    systemPrompt, userPromptTemplate, examples, savedPromptSets, savePromptSet, loadPromptSet, deletePromptSet, arePromptsGenerated,
    completeConfigurations, saveCompleteConfiguration, loadCompleteConfiguration, deleteCompleteConfiguration, resetConfiguration, isConfigurationComplete
  ]);

  return <ConfigurationContext.Provider value={value}>{children}</ConfigurationContext.Provider>;
}

export function useConfiguration() {
  const context = useContext(ConfigurationContext);
  if (context === undefined) {
    throw new Error('useConfiguration must be used within a ConfigurationProvider');
  }
  return context;
}

// Mock generation functions (to be replaced with actual AI generation)
function generateMockSchema(input: GenerationInput): string {
  const baseSchema = {
    $schema: 'http://json-schema.org/draft-07/schema#',
    title: `${input.domain.charAt(0).toUpperCase() + input.domain.slice(1)}Data`,
    description: `Schema for extracting ${input.domain} data`,
    type: 'object',
    properties: {},
    required: [] as string[]
  };

  // Add domain-specific properties
  switch (input.domain) {
    case 'invoice':
      baseSchema.properties = {
        invoiceNumber: { type: 'string', description: 'Invoice number' },
        date: { type: 'string', format: 'date', description: 'Invoice date' },
        customerName: { type: 'string', description: 'Customer name' },
        totalAmount: { type: 'number', description: 'Total amount' },
        items: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              description: { type: 'string' },
              quantity: { type: 'number' },
              price: { type: 'number' }
            }
          }
        }
      };
      baseSchema.required = ['invoiceNumber', 'totalAmount'];
      break;
    case 'resume':
      baseSchema.properties = {
        name: { type: 'string', description: 'Full name' },
        email: { type: 'string', format: 'email', description: 'Email address' },
        phone: { type: 'string', description: 'Phone number' },
        experience: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              company: { type: 'string' },
              position: { type: 'string' },
              duration: { type: 'string' }
            }
          }
        },
        skills: { type: 'array', items: { type: 'string' } }
      };
      baseSchema.required = ['name', 'email'];
      break;
    default:
      baseSchema.properties = {
        title: { type: 'string', description: 'Document title' },
        content: { type: 'string', description: 'Main content' },
        metadata: { type: 'object', description: 'Additional metadata' }
      };
      baseSchema.required = ['title'];
  }

  return JSON.stringify(baseSchema, null, 2);
}

function generateMockSystemPrompt(input: GenerationInput): string {
  return `You are an expert data extraction assistant specialized in processing ${input.domain} documents. 
Your task is to carefully analyze documents and extract structured information according to the provided JSON schema.

Key guidelines:
- Extract information accurately and completely
- Follow the schema structure exactly
- Use null for missing information unless schema specifies defaults
- Maintain data type consistency
- Focus on precision and attention to detail

Return the extracted data in the requested format.`;
}

function generateMockUserPrompt(input: GenerationInput): string {
  return `Please extract ${input.domain} information from the provided document according to the JSON schema.

Document: {{#if document_content_text}}{{document_content_text}}{{else}}{{media url=document_media_url}}{{/if}}

Schema: {{json_schema_text}}

{{#if examples_list.length}}
Examples:
{{#each examples_list}}
Input: {{{this.input}}}
Output: {{{this.output}}}
{{/each}}
{{/if}}

Extract the information and return the result in the requested format.`;
}

function generateMockExamples(input: GenerationInput): Example[] {
  switch (input.domain) {
    case 'invoice':
      return [
        {
          input: 'Invoice #INV-001 dated 2024-01-15 for Acme Corp, Total: $1,250.00',
          output: '{"invoiceNumber": "INV-001", "date": "2024-01-15", "customerName": "Acme Corp", "totalAmount": 1250.00}'
        }
      ];
    case 'resume':
      return [
        {
          input: 'John Smith, john@email.com, Software Engineer at Tech Corp (2020-2023)',
          output: '{"name": "John Smith", "email": "john@email.com", "experience": [{"company": "Tech Corp", "position": "Software Engineer", "duration": "2020-2023"}]}'
        }
      ];
    default:
      return [
        {
          input: 'Sample document with title "Project Report" containing analysis data',
          output: '{"title": "Project Report", "content": "analysis data"}'
        }
      ];
  }
}