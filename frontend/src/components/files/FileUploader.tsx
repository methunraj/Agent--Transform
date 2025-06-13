
'use client';

import { useState, useTransition } from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { useToast } from '@/hooks/use-toast';
import { UploadCloud, Loader2 } from 'lucide-react';
import { useFiles } from '@/contexts/FileContext';

export function FileUploader() {
  const [stagedFiles, setStagedFiles] = useState<File[]>([]);
  const { addFile } = useFiles();
  const { toast } = useToast();
  const [isProcessing, startProcessingTransition] = useTransition();

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      const newFiles = Array.from(event.target.files);
      if (stagedFiles.length + newFiles.length > 10) { // Example limit
        toast({ title: "Too many files", description: "You can stage a maximum of 10 files at once.", variant: "destructive"});
        return;
      }
      setStagedFiles(prevFiles => [...prevFiles, ...newFiles]);
      toast({ title: "Files Staged", description: `${newFiles.length} file(s) added to staging area.`});
      event.target.value = ''; // Clear the input for re-selection of same file
    }
  };

  const handleAddStagedFilesToJob = () => {
    if (stagedFiles.length === 0) return;

    startProcessingTransition(async () => {
      let filesAddedCount = 0;
      for (const file of stagedFiles) {
        try {
          // Basic size check (e.g., 100MB)
          if (file.size > 100 * 1024 * 1024) {
            toast({ title: "File Too Large", description: `${file.name} is too large (max 100MB).`, variant: "destructive" });
            continue;
          }
          await addFile({ name: file.name, type: file.type, size: file.size, rawFile: file });
          filesAddedCount++;
        } catch (error) {
          console.error("Error processing file:", file.name, error);
          toast({ title: "Error Processing File", description: `Could not process ${file.name}.`, variant: "destructive"});
        }
      }
      if (filesAddedCount > 0) {
        toast({ title: "Files Added to Job", description: `${filesAddedCount} file(s) successfully added to the job queue.`});
      }
      setStagedFiles([]); // Clear staged files after attempting to add them
    });
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-col items-center justify-center w-full p-6 border-2 border-dashed rounded-lg border-border hover:border-primary transition-colors">
        <UploadCloud className="w-12 h-12 text-muted-foreground mb-4" />
        <Label htmlFor="file-upload" className="text-primary font-semibold cursor-pointer mb-2">
          Click to upload or drag and drop
        </Label>
        <p className="text-xs text-muted-foreground">TXT, JSON, PDF, DOCX, JPG, PNG (MAX. 100MB)</p>
        <Input 
          id="file-upload" 
          type="file" 
          multiple 
          onChange={handleFileChange} 
          className="hidden"
          accept=".txt,.json,.pdf,.docx,.jpg,.jpeg,.png"
          disabled={isProcessing}
        />
      </div>
      {stagedFiles.length > 0 && (
        <div>
          <h4 className="font-medium mb-2">Staged files ({stagedFiles.length}):</h4>
          <ul className="list-disc list-inside text-sm space-y-1 max-h-32 overflow-y-auto">
            {stagedFiles.map((file, index) => (
              <li key={index}>{file.name} ({(file.size / 1024).toFixed(2)} KB)</li>
            ))}
          </ul>
           <Button variant="outline" size="sm" className="mt-2" onClick={() => setStagedFiles([])} disabled={isProcessing}>Clear Staged Files</Button>
        </div>
      )}
       <Button 
        className="w-full mt-2" 
        disabled={stagedFiles.length === 0 || isProcessing}
        onClick={handleAddStagedFilesToJob}
       >
        {isProcessing ? (
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        ) : null}
        {isProcessing ? 'Processing...' : 'Add Staged Files to Job'}
      </Button>
    </div>
  );
}
