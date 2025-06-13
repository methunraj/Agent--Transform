'use client';

import { useState, useRef, useTransition } from 'react';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { useToast } from '@/hooks/use-toast';
import { FolderOpen, Loader2 } from 'lucide-react';
import { useFiles } from '@/contexts/FileContext';
import type { AppFile } from '@/types';

export function DirectorySelector() {
  const [recursive, setRecursive] = useState(true);
  const { toast } = useToast();
  const { addFile } = useFiles();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isProcessing, startProcessingTransition] = useTransition();
  const [selectedDirectoryName, setSelectedDirectoryName] = useState<string | null>(null);

  const handleFileSelection = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      setSelectedDirectoryName(files[0].webkitRelativePath.split('/')[0] || 'Selected Directory');
      startProcessingTransition(async () => {
        let filesAddedCount = 0;
        let filesSkippedCount = 0;
        const maxFilesToAdd = 200; 

        for (let i = 0; i < Math.min(files.length, maxFilesToAdd); i++) {
          const file = files[i];
          
          if (file.size > 100 * 1024 * 1024) { 
            toast({ title: "File Too Large", description: `${file.name} is too large (max 100MB) and was skipped.`, variant: "destructive", duration: 2000 });
            filesSkippedCount++;
            continue;
          }
          
          try {
            await addFile({ name: file.name, type: file.type, size: file.size, rawFile: file });
            filesAddedCount++;
          } catch (error) {
            console.error("Error processing file from directory:", file.name, error);
            toast({ title: "Error Adding File", description: `Could not add ${file.name} to the queue.`, variant: "destructive"});
            filesSkippedCount++;
          }
        }
        
        let toastMessage = `${filesAddedCount} file(s) successfully added to the job queue.`;
        if (files.length > maxFilesToAdd) {
            toastMessage += ` Processed the first ${maxFilesToAdd} files to prevent performance issues. Select a smaller directory or upload in batches if more are needed.`;
        }
        if (filesSkippedCount > 0) {
            toastMessage += ` ${filesSkippedCount} file(s) were skipped (e.g., too large or error during staging).`;
        }
        
        toast({
          title: filesAddedCount > 0 ? "Directory Files Staged" : "No New Files Staged",
          description: toastMessage,
          duration: filesAddedCount > 0 ? 3000 : 5000,
        });
        if (fileInputRef.current) {
            fileInputRef.current.value = ''; 
        }
      });
    } else {
        setSelectedDirectoryName(null);
    }
  };

  const triggerFileSelect = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  return (
    <div className="space-y-4">
      <input
        type="file"
        // @ts-ignore 
        webkitdirectory=""
        directory=""
        multiple
        ref={fileInputRef}
        onChange={handleFileSelection}
        className="hidden"
        disabled={isProcessing}
      />
      <div>
        <div className="flex items-center gap-2 mb-2">
         <FolderOpen className="h-5 w-5 text-muted-foreground" />
         <Label htmlFor="directory-selector-button" className="text-sm font-medium">
            Select Directory for Processing
         </Label>
        </div>
        <p className="text-xs text-muted-foreground mb-3">
          Click the button below to select a directory. Supported files from the directory will be added to the job queue.
          {selectedDirectoryName && ` Current selection target: ${selectedDirectoryName}`}
        </p>
      </div>
      <div className="flex items-center space-x-2">
        <Checkbox
          id="recursive-scan"
          checked={recursive} 
          onCheckedChange={(checked) => setRecursive(checked as boolean)}
          disabled={true} 
        />
        <Label htmlFor="recursive-scan" className="text-sm font-medium opacity-70">
          Include subdirectories (browser default)
        </Label>
      </div>
      <Button id="directory-selector-button" onClick={triggerFileSelect} className="w-full" disabled={isProcessing}>
        {isProcessing ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <FolderOpen className="mr-2 h-4 w-4" /> }
        {isProcessing ? 'Scanning Directory...' : 'Select and Add Directory Files'}
      </Button>
       {isProcessing && <p className="text-xs text-center text-muted-foreground mt-2">Adding files from directory. This may take a moment for large directories...</p>}
    </div>
  );
}
