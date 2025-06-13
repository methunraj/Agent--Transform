'use client';

import { createContext, useState, useContext, useEffect, useCallback, Dispatch, SetStateAction } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Example, defaultSystemPrompt, defaultUserPromptTemplate } from './ConfigurationContext';

export interface SavedPromptSet {
  id: string;
  name: string;
  systemPrompt: string;
  userPromptTemplate: string;
  examples: Example[];
  createdAt: number;
}

interface PromptContextType {
  systemPrompt: string;
  setSystemPrompt: Dispatch<SetStateAction<string>>;
  userPromptTemplate: string;
  setUserPromptTemplate: Dispatch<SetStateAction<string>>;
  examples: Example[];
  setExamples: Dispatch<SetStateAction<Example[]>>;
  
  savedPromptSets: SavedPromptSet[];
  savePromptSetToList: (name: string, systemPrompt: string, userPromptTemplate: string, examples: Example[]) => void;
  loadPromptSetFromList: (promptSetId: string) => SavedPromptSet | null;
  deletePromptSetFromList: (promptSetId: string) => void;
  updatePromptSetInList: (promptSetId: string, name: string, systemPrompt: string, userPromptTemplate: string, examples: Example[]) => void;
}

const PromptContext = createContext<PromptContextType | undefined>(undefined);

export function PromptProvider({ children }: { children: React.ReactNode }) {
  const [systemPrompt, setSystemPrompt] = useState<string>(defaultSystemPrompt);
  const [userPromptTemplate, setUserPromptTemplate] = useState<string>(defaultUserPromptTemplate);
  const [examples, setExamples] = useState<Example[]>([]);
  const [savedPromptSets, setSavedPromptSets] = useState<SavedPromptSet[]>([]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      // Changed the localStorage key to avoid conflict with ConfigurationContext
      const storedSets = localStorage.getItem('intelliextract_promptContext_savedPromptSets');
      if (storedSets) {
        setSavedPromptSets(JSON.parse(storedSets));
      }
    }
  }, []);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      // Changed the localStorage key to avoid conflict with ConfigurationContext
      localStorage.setItem('intelliextract_promptContext_savedPromptSets', JSON.stringify(savedPromptSets));
    }
  }, [savedPromptSets]);

  const savePromptSetToList = useCallback((name: string, newSystemPrompt: string, newUserPromptTemplate: string, newExamples: Example[]) => {
    if (!name.trim()) throw new Error("Prompt set name cannot be empty.");
    if (savedPromptSets.some(ps => ps.name === name.trim())) {
      throw new Error(`Prompt set with name "${name.trim()}" already exists.`);
    }
    
    const newSet: SavedPromptSet = {
      id: uuidv4(),
      name: name.trim(),
      systemPrompt: newSystemPrompt,
      userPromptTemplate: newUserPromptTemplate,
      examples: newExamples,
      createdAt: Date.now()
    };
    
    setSavedPromptSets(prev => [...prev, newSet].sort((a, b) => a.name.localeCompare(b.name)));
  }, [savedPromptSets]);

  const loadPromptSetFromList = useCallback((promptSetId: string): SavedPromptSet | null => {
    const set = savedPromptSets.find(ps => ps.id === promptSetId);
    if (set) {
      setSystemPrompt(set.systemPrompt);
      setUserPromptTemplate(set.userPromptTemplate);
      setExamples(set.examples);
      return set;
    }
    return null;
  }, [savedPromptSets]);

  const deletePromptSetFromList = useCallback((promptSetId: string) => {
    setSavedPromptSets(prev => prev.filter(ps => ps.id !== promptSetId));
  }, []);

  const updatePromptSetInList = useCallback((promptSetId: string, name: string, newSystemPrompt: string, newUserPromptTemplate: string, newExamples: Example[]) => {
    if (!name.trim()) throw new Error("Prompt set name cannot be empty.");
    
    // Check if another prompt set (not this one) already has this name
    if (savedPromptSets.some(ps => ps.id !== promptSetId && ps.name === name.trim())) {
      throw new Error(`Another prompt set with name "${name.trim()}" already exists.`);
    }
    
    setSavedPromptSets(prev => {
      const updated = prev.map(ps => {
        if (ps.id === promptSetId) {
          return {
            ...ps,
            name: name.trim(),
            systemPrompt: newSystemPrompt,
            userPromptTemplate: newUserPromptTemplate,
            examples: newExamples,
          };
        }
        return ps;
      });
      return updated.sort((a, b) => a.name.localeCompare(b.name));
    });
  }, [savedPromptSets]);

  return (
    <PromptContext.Provider value={{
      systemPrompt,
      setSystemPrompt,
      userPromptTemplate,
      setUserPromptTemplate,
      examples,
      setExamples,
      savedPromptSets,
      savePromptSetToList,
      loadPromptSetFromList,
      deletePromptSetFromList,
      updatePromptSetInList
    }}>
      {children}
    </PromptContext.Provider>
  );
}

export function usePrompts() {
  const context = useContext(PromptContext);
  if (context === undefined) {
    throw new Error('usePrompts must be used within a PromptProvider');
  }
  return context;
}
