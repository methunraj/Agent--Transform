
'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { FileText, Image as LucideImage, Trash2, ListFilter } from 'lucide-react'; // Renamed Image to LucideImage
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuCheckboxItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useFiles } from '@/contexts/FileContext';
import type { AppFile } from '@/types';

const getFileTypeVisual = (mimeType: string): 'text' | 'image' | 'pdf' | 'doc' | 'unknown' => {
  if (mimeType.startsWith('text/')) return 'text';
  if (mimeType.startsWith('image/')) return 'image';
  if (mimeType === 'application/pdf') return 'pdf';
  if (mimeType.includes('wordprocessingml') || mimeType === 'application/msword') return 'doc';
  return 'unknown';
};

export function SelectedFilesList() {
  const { files, removeFile, clearFiles: clearAllFiles } = useFiles();
  const [filter, setFilter] = useState<'all' | 'text' | 'image' | 'pdf' | 'doc' | 'unknown'>('all');

  const filteredFiles = files.filter(file => {
    if (filter === 'all') return true;
    return getFileTypeVisual(file.type) === filter;
  });

  if (files.length === 0) {
    return <p className="text-muted-foreground text-center py-4">No files selected for processing yet.</p>;
  }

  return (
    <div className="space-y-4">
        <div className="flex justify-between items-center">
            <h3 className="text-md font-semibold">
                {filteredFiles.length} file(s) selected {filter !== 'all' ? `(matching filter: ${filter})` : ''}
            </h3>
            <div className="flex gap-2">
            <DropdownMenu>
                <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm">
                    <ListFilter className="mr-2 h-4 w-4" />
                    Filter
                </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                <DropdownMenuLabel>Filter by type</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuCheckboxItem checked={filter === 'all'} onCheckedChange={() => setFilter('all')}>All</DropdownMenuCheckboxItem>
                <DropdownMenuCheckboxItem checked={filter === 'text'} onCheckedChange={() => setFilter('text')}>Text</DropdownMenuCheckboxItem>
                <DropdownMenuCheckboxItem checked={filter === 'image'} onCheckedChange={() => setFilter('image')}>Image</DropdownMenuCheckboxItem>
                <DropdownMenuCheckboxItem checked={filter === 'pdf'} onCheckedChange={() => setFilter('pdf')}>PDF</DropdownMenuCheckboxItem>
                <DropdownMenuCheckboxItem checked={filter === 'doc'} onCheckedChange={() => setFilter('doc')}>Document</DropdownMenuCheckboxItem>
                <DropdownMenuCheckboxItem checked={filter === 'unknown'} onCheckedChange={() => setFilter('unknown')}>Other</DropdownMenuCheckboxItem>
                </DropdownMenuContent>
            </DropdownMenu>
            <Button variant="outline" size="sm" onClick={clearAllFiles} disabled={files.length === 0}>
                Clear All
            </Button>
            </div>
        </div>
      <ScrollArea className="h-72 w-full rounded-md border p-2">
        {filteredFiles.length === 0 && files.length > 0 ? (
            <p className="text-muted-foreground text-center py-4">No files match the current filter.</p>
        ) : (
        <ul className="space-y-2">
          {filteredFiles.map((file) => (
            <li key={file.id} className="flex items-center justify-between p-3 bg-secondary/30 rounded-md hover:bg-secondary/60 transition-colors">
              <div className="flex items-center gap-3">
                {getFileTypeVisual(file.type) === 'image' ? <LucideImage className="h-5 w-5 text-muted-foreground" /> : <FileText className="h-5 w-5 text-muted-foreground" />}
                <div className="flex flex-col">
                  <span className="text-sm font-medium truncate max-w-xs" title={file.name}>{file.name}</span>
                  <span className="text-xs text-muted-foreground">{file.type} - {(file.size / 1024).toFixed(2)} KB</span>
                </div>
              </div>
              <Button variant="ghost" size="icon" onClick={() => removeFile(file.id)} className="text-muted-foreground hover:text-destructive">
                <Trash2 className="h-4 w-4" />
              </Button>
            </li>
          ))}
        </ul>
        )}
      </ScrollArea>
    </div>
  );
}
