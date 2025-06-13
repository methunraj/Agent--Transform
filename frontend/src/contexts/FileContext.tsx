'use client';
import type { Dispatch, SetStateAction } from 'react';
import { createContext, useContext, useState, useCallback, useMemo } from 'react';
import type { AppFile } from '@/types';
import { v4 as uuidv4 } from 'uuid'; 

interface FileContextType {
  files: AppFile[];
  addFile: (fileData: Omit<AppFile, 'id' | 'dataUri' | 'textContent'> & { rawFile: File }) => Promise<void>;
  addTextFile: (name: string, textContent: string) => void;
  removeFile: (id: string) => void;
  clearFiles: () => void;
}

const FileContext = createContext<FileContextType | undefined>(undefined);

async function readFileAsDataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsText(file);
  });
}


export function FileProvider({ children }: { children: React.ReactNode }) {
  const [files, setFiles] = useState<AppFile[]>([]);

  const addFile = useCallback(async (fileData: Omit<AppFile, 'id' | 'dataUri' | 'textContent'> & { rawFile: File }) => {
    const dataUri = await readFileAsDataURL(fileData.rawFile);
    let textContent: string | undefined = undefined;
    
    const knownTextMimeTypes = ['text/plain', 'text/csv', 'text/html', 'text/xml', 'application/json', 'application/javascript'];
    const knownTextExtensions = ['.txt', '.csv', '.md', '.json', '.js', '.ts', '.html', '.xml', '.log'];
    
    const isTextMime = knownTextMimeTypes.includes(fileData.type);
    const isTextExtension = knownTextExtensions.some(ext => fileData.name.toLowerCase().endsWith(ext));

    if (isTextMime || (fileData.type === '' && isTextExtension)) { // fileData.type can be empty for some uploads
      try {
        textContent = await readFileAsText(fileData.rawFile);
      } catch (e) {
        console.warn(`Could not read text content for ${fileData.name}:`, e);
      }
    }
    // For PDFs, DOCX, etc., text extraction is complex client-side.
    // The Genkit flow will handle these using multimodal capabilities or server-side extraction if `textContent` is not provided.

    setFiles(prevFiles => {
        // Prevent duplicate file additions if same file object is processed (e.g. from multiple drop events)
        // This check is basic and might not catch all cases (e.g., same file selected twice).
        // A more robust check might involve a hash of the file content if performance allows.
        // For now, checking by name and size as a simple heuristic.
        if (prevFiles.some(f => f.name === fileData.name && f.size === fileData.size)) {
            console.warn(`File ${fileData.name} seems to be a duplicate, skipping addition.`);
            return prevFiles;
        }
        return [...prevFiles, { ...fileData, id: uuidv4(), dataUri, textContent }];
    });
  }, []);

  const addTextFile = useCallback((name: string, textContentValue: string) => {
    const blob = new Blob([textContentValue], { type: 'text/plain' });
    const reader = new FileReader();
    reader.onloadend = () => {
        const base64DataUri = reader.result as string;
        const newFile: AppFile = {
            id: uuidv4(),
            name,
            type: 'text/plain',
            size: textContentValue.length,
            dataUri: base64DataUri,
            textContent: textContentValue,
        };
        setFiles(prevFiles => [...prevFiles, newFile]);
    };
    reader.readAsDataURL(blob);
  }, []);


  const removeFile = useCallback((id: string) => {
    setFiles(prevFiles => prevFiles.filter(file => file.id !== id));
  }, []);

  const clearFiles = useCallback(() => {
    setFiles([]);
  }, []);

  const value = useMemo(() => ({ files, addFile, addTextFile, removeFile, clearFiles }), [files, addFile, addTextFile, removeFile, clearFiles]);

  return <FileContext.Provider value={value}>{children}</FileContext.Provider>;
}

export function useFiles() {
  const context = useContext(FileContext);
  if (context === undefined) {
    throw new Error('useFiles must be used within a FileProvider');
  }
  return context;
}
