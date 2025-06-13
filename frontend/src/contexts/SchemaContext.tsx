'use client';
import type { Dispatch, SetStateAction } from 'react';
import { createContext, useContext, useState, useMemo, useEffect, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';

export interface SavedSchema {
  id: string;
  name: string;
  schemaJson: string;
  createdAt: number;
}

const defaultSchema = JSON.stringify(
  {
    $schema: 'http://json-schema.org/draft-07/schema#',
    title: 'ExtractedData',
    description: 'Schema for data to be extracted from a document.',
    type: 'object',
    properties: {
      title: {
        type: 'string',
        description: 'The title or heading of the document',
      },
      date: {
        type: ['string', 'null'],
        format: 'date',
        description: 'The main date mentioned in the document (YYYY-MM-DD format if possible)',
      },
      summary: {
        type: 'string',
        description: 'A brief summary of the document content',
      },
      keywords: {
        type: 'array',
        items: {
          type: 'string'
        },
        description: 'A list of keywords from the document'
      }
    },
    required: ['title', 'summary'],
  },
  null,
  2
);

interface SchemaContextType {
  schemaJson: string; // Currently active schema in editor for session use
  setSchemaJson: Dispatch<SetStateAction<string>>;
  savedSchemas: SavedSchema[];
  saveSchemaToList: (name: string, schemaToSave: string) => void;
  loadSchemaFromList: (schemaId: string) => SavedSchema | null;
  deleteSchemaFromList: (schemaId: string) => void;
  updateSchemaInList: (schemaId: string, name: string, schemaToUpdate: string) => void;
}

const SchemaContext = createContext<SchemaContextType | undefined>(undefined);

export function SchemaProvider({ children }: { children: React.ReactNode }) {
  const [schemaJson, setSchemaJson] = useState<string>(defaultSchema);
  const [savedSchemas, setSavedSchemas] = useState<SavedSchema[]>([]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const storedSchemas = localStorage.getItem('intelliextract_savedSchemas');
      if (storedSchemas) {
        setSavedSchemas(JSON.parse(storedSchemas));
      }
    }
  }, []);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('intelliextract_savedSchemas', JSON.stringify(savedSchemas));
    }
  }, [savedSchemas]);

  const saveSchemaToList = useCallback((name: string, schemaToSave: string) => {
    if (!name.trim()) {
      throw new Error("Schema name cannot be empty.");
    }
    // Check for duplicate name before saving as new, allow update via updateSchemaInList
    if (savedSchemas.some(s => s.name === name.trim())) {
        throw new Error(`Schema with name "${name.trim()}" already exists. Use a different name or update the existing one.`);
    }
    const newSchema: SavedSchema = { 
      id: uuidv4(), 
      name: name.trim(), 
      schemaJson: schemaToSave, 
      createdAt: Date.now() 
    };
    setSavedSchemas(prev => [...prev, newSchema].sort((a,b) => a.name.localeCompare(b.name)));
  }, [savedSchemas]);

  const updateSchemaInList = useCallback((schemaId: string, name: string, schemaToUpdate: string) => {
    if (!name.trim()) {
      throw new Error("Schema name cannot be empty.");
    }
    setSavedSchemas(prev => 
      prev.map(s => 
        s.id === schemaId ? { ...s, name: name.trim(), schemaJson: schemaToUpdate, createdAt: Date.now() } : s
      ).sort((a,b) => a.name.localeCompare(b.name))
    );
  }, []);

  const loadSchemaFromList = useCallback((schemaId: string): SavedSchema | null => {
    const schemaToLoad = savedSchemas.find(s => s.id === schemaId);
    if (schemaToLoad) {
      setSchemaJson(schemaToLoad.schemaJson); // Update the active editor content
      return schemaToLoad;
    }
    return null;
  }, [savedSchemas, setSchemaJson]);

  const deleteSchemaFromList = useCallback((schemaId: string) => {
    setSavedSchemas(prev => prev.filter(s => s.id !== schemaId));
  }, []);

  const value = useMemo(() => ({ 
    schemaJson, setSchemaJson, 
    savedSchemas, saveSchemaToList, loadSchemaFromList, deleteSchemaFromList, updateSchemaInList
  }), [schemaJson, setSchemaJson, savedSchemas, saveSchemaToList, loadSchemaFromList, deleteSchemaFromList, updateSchemaInList]);

  return <SchemaContext.Provider value={value}>{children}</SchemaContext.Provider>;
}

export function useSchema() {
  const context = useContext(SchemaContext);
  if (context === undefined) {
    throw new Error('useSchema must be used within a SchemaProvider');
  }
  return context;
}
