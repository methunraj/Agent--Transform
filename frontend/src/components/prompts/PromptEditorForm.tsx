'use client';

import { useState, useTransition, useEffect } from 'react';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Trash2, Save, Loader2, UploadCloud } from 'lucide-react';
import { usePrompts, type SavedPromptSet } from '@/contexts/PromptContext';
import type { Example } from '@/types';
import { useToast } from '@/hooks/use-toast';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";

export function PromptEditorForm() {
  const { 
    systemPrompt: contextSystemPrompt, setSystemPrompt: setContextSystemPrompt,
    userPromptTemplate: contextUserPromptTemplate, setUserPromptTemplate: setContextUserPromptTemplate,
    examples: contextExamples, setExamples: setContextExamples,
    savedPromptSets,
    savePromptSetToList,
    loadPromptSetFromList,
    deletePromptSetFromList,
    updatePromptSetInList
  } = usePrompts();

  const [localSystemPrompt, setLocalSystemPrompt] = useState(contextSystemPrompt);
  const [localUserPromptTemplate, setLocalUserPromptTemplate] = useState(contextUserPromptTemplate);
  const [localExamples, setLocalExamples] = useState<Example[]>(contextExamples);
  
  const [currentExampleInput, setCurrentExampleInput] = useState('');
  const [currentExampleOutput, setCurrentExampleOutput] = useState('');
  
  const [isProcessing, startProcessingTransition] = useTransition();
  const { toast } = useToast();

  const [selectedPromptSetId, setSelectedPromptSetId] = useState<string | undefined>(undefined);
  const [currentPromptSetName, setCurrentPromptSetName] = useState('');

  useEffect(() => {
    if (!selectedPromptSetId) {
        setLocalSystemPrompt(contextSystemPrompt);
        setLocalUserPromptTemplate(contextUserPromptTemplate);
        setLocalExamples(contextExamples);
    }
  }, [contextSystemPrompt, contextUserPromptTemplate, contextExamples, selectedPromptSetId]);

  useEffect(() => {
    if (selectedPromptSetId) {
      const loaded = loadPromptSetFromList(selectedPromptSetId);
      if (loaded) {
        setLocalSystemPrompt(loaded.systemPrompt);
        setLocalUserPromptTemplate(loaded.userPromptTemplate);
        setLocalExamples(loaded.examples);
        setCurrentPromptSetName(loaded.name);
      } else {
        setSelectedPromptSetId(undefined);
        setCurrentPromptSetName('');
      }
    }
  }, [selectedPromptSetId, loadPromptSetFromList]);


  const addExample = () => {
    if (currentExampleInput.trim() && currentExampleOutput.trim()) {
      setLocalExamples([...localExamples, { input: currentExampleInput, output: currentExampleOutput }]);
      setCurrentExampleInput('');
      setCurrentExampleOutput('');
      toast({ title: "Example Added Locally", description: "This example is added to the current editor session."});
    } else {
      toast({ title: "Missing Example Fields", description: "Both example input and output are required.", variant: "destructive" });
    }
  };

  const removeExample = (index: number) => {
    setLocalExamples(localExamples.filter((_, i) => i !== index));
    toast({ title: "Example Removed Locally", description: "This example is removed from the current editor session."});
  };

  const handleApplyPromptsToSession = () => {
    startProcessingTransition(() => {
      setContextSystemPrompt(localSystemPrompt);
      setContextUserPromptTemplate(localUserPromptTemplate);
      setContextExamples(localExamples);
      toast({
        title: "Prompts Applied",
        description: "Editor prompts (system, user template, examples) are now active for the current session.",
      });
    });
  };

  const handleSavePromptSetToList = () => {
    if (!currentPromptSetName.trim()) {
      toast({ title: 'Name Required', description: 'Please enter a name for the prompt set.', variant: 'destructive'});
      return;
    }
    startProcessingTransition(() => {
      try {
        if (selectedPromptSetId && savedPromptSets.find(ps => ps.id === selectedPromptSetId && ps.name === currentPromptSetName.trim())) {
          updatePromptSetInList(selectedPromptSetId, currentPromptSetName.trim(), localSystemPrompt, localUserPromptTemplate, localExamples);
          toast({ title: 'Prompt Set Updated', description: `Prompt set "${currentPromptSetName.trim()}" has been updated.` });
        } else {
          savePromptSetToList(currentPromptSetName.trim(), localSystemPrompt, localUserPromptTemplate, localExamples);
          toast({ title: 'Prompt Set Saved', description: `Prompt set "${currentPromptSetName.trim()}" saved to list.` });
           const newSaved = savedPromptSets.find(ps => ps.name === currentPromptSetName.trim());
           if (newSaved) setSelectedPromptSetId(newSaved.id);
        }
      } catch (error) {
        const specificError = error instanceof Error ? error.message : String(error);
        toast({ title: 'Save Failed', description: `Could not save prompt set. Error: ${specificError}`, variant: 'destructive'});
      }
    });
  };
  
  const handleDeletePromptSetFromList = () => {
    if (!selectedPromptSetId) {
      toast({ title: 'No Prompt Set Selected', description: 'Please select a prompt set to delete.', variant: 'destructive'});
      return;
    }
    startProcessingTransition(() => {
      try {
        const setToDelete = savedPromptSets.find(ps => ps.id === selectedPromptSetId);
        deletePromptSetFromList(selectedPromptSetId);
        toast({ title: 'Prompt Set Deleted', description: `Prompt set "${setToDelete?.name}" deleted.` });
        setSelectedPromptSetId(undefined);
        setCurrentPromptSetName('');
        // Reset local editor to context defaults
        setLocalSystemPrompt(contextSystemPrompt);
        setLocalUserPromptTemplate(contextUserPromptTemplate);
        setLocalExamples(contextExamples);
      } catch (error) {
        const specificError = error instanceof Error ? error.message : String(error);
        toast({ title: 'Delete Failed', description: `Could not delete prompt set. Error: ${specificError}`, variant: 'destructive'});
      }
    });
  };

  const handleSelectPromptSet = (promptSetId: string) => {
     if (!promptSetId) {
      setSelectedPromptSetId(undefined);
      setCurrentPromptSetName('');
      setLocalSystemPrompt(contextSystemPrompt);
      setLocalUserPromptTemplate(contextUserPromptTemplate);
      setLocalExamples(contextExamples);
      return;
    }
    setSelectedPromptSetId(promptSetId);
    // Effect will load data
  };

  return (
    <div className="space-y-8">
      {/* Prompt Editors Section */}
      <Card>
        <CardHeader>
          <CardTitle>Prompt Editors</CardTitle>
          <CardDescription>Define system and user prompts. Apply them to the current session for extraction, or save/load sets from your list.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div>
            <Label htmlFor="system-prompt" className="text-base font-semibold">System Prompt</Label>
            <p className="text-sm text-muted-foreground mb-1">Instructions for the LLM's general behavior.</p>
            <Textarea id="system-prompt" value={localSystemPrompt} onChange={(e) => setLocalSystemPrompt(e.target.value)} rows={6} disabled={isProcessing} />
          </div>
          <div>
            <Label htmlFor="user-prompt-template" className="text-base font-semibold">User Prompt Template</Label>
            <p className="text-sm text-muted-foreground mb-1">
              {'Template for document-specific prompts. Use Handlebars: `{{document_content_text}}`, `{{media url=document_media_url}}`, `{{json_schema_text}}`, `{{examples_list}}`.'}
            </p>
            <Textarea id="user-prompt-template" value={localUserPromptTemplate} onChange={(e) => setLocalUserPromptTemplate(e.target.value)} rows={10} disabled={isProcessing} />
          </div>
           <Button onClick={handleApplyPromptsToSession} disabled={isProcessing} variant="outline">
             <UploadCloud className="mr-2 h-4 w-4" /> Apply Editor Content to Current Session
           </Button>
        </CardContent>
      </Card>

      {/* Few-shot Examples Section */}
      <Card>
        <CardHeader>
          <CardTitle>Few-shot Examples (for current editor session)</CardTitle>
          <CardDescription>Provide input/output pairs to guide the LLM. These are part of the currently edited prompt set.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {localExamples.map((example, index) => (
            <Card key={index} className="bg-muted/50">
              <CardHeader className="p-3">
                <div className="flex justify-between items-center">
                  <CardTitle className="text-sm">Example {index + 1}</CardTitle>
                  <Button variant="ghost" size="icon" onClick={() => removeExample(index)} disabled={isProcessing}><Trash2 className="h-4 w-4 text-destructive" /></Button>
                </div>
              </CardHeader>
              <CardContent className="p-3 space-y-2">
                <div><Label htmlFor={`ex-in-${index}`} className="text-xs">Input:</Label><Textarea id={`ex-in-${index}`} value={example.input} readOnly rows={2} className="bg-background text-xs"/></div>
                <div><Label htmlFor={`ex-out-${index}`} className="text-xs">Output (JSON):</Label><Textarea id={`ex-out-${index}`} value={example.output} readOnly rows={2} className="font-mono bg-background text-xs"/></div>
              </CardContent>
            </Card>
          ))}
          <div className="space-y-2 pt-3 border-t">
            <Label className="text-sm font-semibold">Add New Example</Label>
            <div><Label htmlFor="new-ex-in" className="text-xs">Input:</Label><Textarea id="new-ex-in" value={currentExampleInput} onChange={(e) => setCurrentExampleInput(e.target.value)} rows={2} disabled={isProcessing} /></div>
            <div><Label htmlFor="new-ex-out" className="text-xs">Output (JSON):</Label><Textarea id="new-ex-out" value={currentExampleOutput} onChange={(e) => setCurrentExampleOutput(e.target.value)} rows={2} className="font-mono" disabled={isProcessing} /></div>
            <Button onClick={addExample} variant="outline" size="sm" disabled={isProcessing}>Add Example to Editor</Button>
          </div>
        </CardContent>
      </Card>
      
      {/* Saved Prompt Sets Section */}
      <Card>
        <CardHeader>
          <CardTitle>Saved Prompt Sets</CardTitle>
          <CardDescription>Manage your saved prompt configurations (system, user, and examples).</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
           <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-end">
            <div>
              <Label htmlFor="prompt-set-name">Prompt Set Name</Label>
              <Input id="prompt-set-name" placeholder="Name for this prompt set" value={currentPromptSetName} onChange={(e) => setCurrentPromptSetName(e.target.value)} disabled={isProcessing} className="mt-1"/>
            </div>
            <Button onClick={handleSavePromptSetToList} disabled={isProcessing || !currentPromptSetName.trim()}>
              <Save className="mr-2 h-4 w-4" />
              {selectedPromptSetId && savedPromptSets.find(ps => ps.id === selectedPromptSetId && ps.name === currentPromptSetName.trim()) ? 'Update Set in List' : 'Save as New Set to List'}
            </Button>
          </div>
          <div className="space-y-2">
            <Label htmlFor="saved-prompt-sets-select">Load Prompt Set from List</Label>
            <div className="flex gap-2 items-center">
              <Select onValueChange={handleSelectPromptSet} value={selectedPromptSetId} disabled={isProcessing}>
                <SelectTrigger id="saved-prompt-sets-select" className="flex-grow"><SelectValue placeholder="Select a saved prompt set..." /></SelectTrigger>
                <SelectContent>
                  {savedPromptSets.length === 0 && <SelectItem value="no-sets" disabled>No saved prompt sets</SelectItem>}
                  {savedPromptSets.map((ps) => (
                    <SelectItem key={ps.id} value={ps.id}>{ps.name} <span className="text-xs text-muted-foreground ml-2">({new Date(ps.createdAt).toLocaleDateString()})</span></SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button variant="destructive" size="icon" disabled={isProcessing || !selectedPromptSetId} title="Delete selected prompt set"><Trash2 className="h-4 w-4" /></Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Are you sure?</AlertDialogTitle>
                    <AlertDialogDescription>
                      Delete prompt set "{savedPromptSets.find(ps => ps.id === selectedPromptSetId)?.name || 'selected set'}"? This cannot be undone.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel disabled={isProcessing}>Cancel</AlertDialogCancel>
                    <AlertDialogAction onClick={handleDeletePromptSetFromList} disabled={isProcessing} className="bg-destructive hover:bg-destructive/90">
                      {isProcessing ? <Loader2 className="mr-2 h-4 w-4 animate-spin"/> : null}
                      Delete
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </div>
            {selectedPromptSetId && (
                <p className="text-xs text-muted-foreground mt-1">
                    Selected: {currentPromptSetName}. Editing prompts and saving with the same name will update this set.
                </p>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
