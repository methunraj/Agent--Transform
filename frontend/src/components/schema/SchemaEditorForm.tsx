'use client';

import { useState, useTransition, useEffect } from 'react';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useToast } from '@/hooks/use-toast';
import { generateSchema } from '@/ai/flows/schema-definition-ui';
import { Loader2, Sparkles, Save, List, Download, Trash2, UploadCloud } from 'lucide-react';
import { useSchema, type SavedSchema } from '@/contexts/SchemaContext';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
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



export function SchemaEditorForm() {
  const { 
    schemaJson: contextSchemaJson, 
    setSchemaJson: setContextSchemaJson,
    savedSchemas,
    saveSchemaToList,
    loadSchemaFromList,
    deleteSchemaFromList,
    updateSchemaInList
  } = useSchema();
  
  const [localSchemaJson, setLocalSchemaJson] = useState(contextSchemaJson);
  const [intent, setIntent] = useState('');
  const [isGenerating, startGenerationTransition] = useTransition();
  const [isProcessingListAction, startListActionTransition] = useTransition(); // For save/load/delete list ops

  const [selectedSchemaId, setSelectedSchemaId] = useState<string | undefined>(undefined);
  const [currentSchemaName, setCurrentSchemaName] = useState('');

  const { toast } = useToast();

  useEffect(() => {
    if (!selectedSchemaId) {
        setLocalSchemaJson(contextSchemaJson);
    }
  }, [contextSchemaJson, selectedSchemaId]);
  
  useEffect(() => {
    if (selectedSchemaId) {
      const loaded = loadSchemaFromList(selectedSchemaId);
      if (loaded) {
        setLocalSchemaJson(loaded.schemaJson);
        setCurrentSchemaName(loaded.name);
      } else {
        setSelectedSchemaId(undefined);
        setCurrentSchemaName('');
      }
    }
  }, [selectedSchemaId, loadSchemaFromList]);


  const handleGenerateSchema = () => {
    if (!intent.trim()) {
      toast({ title: 'Intent Required', description: 'Please enter your intent for schema generation.', variant: 'destructive' });
      return;
    }
    startGenerationTransition(async () => {
      try {
        const result = await generateSchema({ intent });
        let schemaToSet = result.schema;
        
        // Use a different regex approach without the 's' flag
        const markdownJsonMatch = schemaToSet.match(/```json\n?([\s\S]*?)\n?```/);
        if (markdownJsonMatch && markdownJsonMatch[1]) {
            schemaToSet = markdownJsonMatch[1];
        }
        schemaToSet = schemaToSet.trim();

        setLocalSchemaJson(schemaToSet);
        toast({ title: 'Schema Generated', description: 'AI has successfully generated a schema. Review and apply or save to list.' });
        setSelectedSchemaId(undefined); 
        setCurrentSchemaName(''); 
      } catch (error) {
        console.error('Schema generation error:', error);
        toast({ title: 'Error', description: 'An error occurred during schema generation.', variant: 'destructive' });
      }
    });
  };

  const handleApplySchemaToSession = () => {
    startListActionTransition(() => {
      const cleanedJsonString = localSchemaJson.trim();
      setContextSchemaJson(cleanedJsonString); 
      toast({ title: 'Schema Applied', description: 'The schema in the editor has been applied to the current session.' });
    });
  };

  const handleSchemaEditorChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setLocalSchemaJson(event.target.value);
  };
  
  const handleSaveSchemaToList = () => {
    if (!currentSchemaName.trim()) {
      toast({ title: 'Name Required', description: 'Please enter a name for the schema.', variant: 'destructive'});
      return;
    }
    startListActionTransition(() => {
      const cleanedJsonString = localSchemaJson.trim();

      if (selectedSchemaId && savedSchemas.find(s => s.id === selectedSchemaId && s.name === currentSchemaName.trim())) {
        updateSchemaInList(selectedSchemaId, currentSchemaName.trim(), cleanedJsonString);
        toast({ title: 'Schema Updated', description: `Schema "${currentSchemaName.trim()}" has been updated in your list.` });
      } else {
        saveSchemaToList(currentSchemaName.trim(), cleanedJsonString);
        toast({ title: 'Schema Saved', description: `Schema "${currentSchemaName.trim()}" has been saved to your list.` });
        const newSaved = savedSchemas.find(s => s.name === currentSchemaName.trim() && s.schemaJson === cleanedJsonString); // Re-fetch from potentially updated list
        if (newSaved) setSelectedSchemaId(newSaved.id);
      }
    });
  };

  const handleDeleteSchemaFromList = () => {
    if (!selectedSchemaId) {
      toast({ title: 'No Schema Selected', description: 'Please select a schema from the list to delete.', variant: 'destructive'});
      return;
    }
    startListActionTransition(() => {
      try {
        const schemaToDelete = savedSchemas.find(s => s.id === selectedSchemaId);
        deleteSchemaFromList(selectedSchemaId);
        toast({ title: 'Schema Deleted', description: `Schema "${schemaToDelete?.name}" has been deleted from your list.` });
        setSelectedSchemaId(undefined);
        setCurrentSchemaName('');
        setLocalSchemaJson(contextSchemaJson); 
      } catch (error) {
        const specificError = error instanceof Error ? error.message : String(error);
        toast({ title: 'Delete Failed', description: `Could not delete schema. Error: ${specificError}`, variant: 'destructive'});
      }
    });
  };

  const handleSelectSchema = (schemaId: string) => {
    if (!schemaId) { 
      setSelectedSchemaId(undefined);
      setCurrentSchemaName('');
      setLocalSchemaJson(contextSchemaJson); 
      return;
    }
    setSelectedSchemaId(schemaId);
  };


  const isBusy = isGenerating || isProcessingListAction;

  return (
    <div className="space-y-6">
      {/* Schema Generation Section */}
      <Card>
        <CardHeader>
          <CardTitle>AI Schema Generation</CardTitle>
          <CardDescription>Describe your data needs, and AI will suggest a JSON schema.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <Label htmlFor="intent">Schema Intent</Label>
            <Textarea
              id="intent"
              placeholder="e.g., Extract names, emails, and job titles from resumes. Include company name and duration of employment."
              value={intent}
              onChange={(e) => setIntent(e.target.value)}
              rows={3}
              disabled={isBusy}
            />
          </div>
          <Button onClick={handleGenerateSchema} disabled={isBusy || !intent.trim()} className="mt-3">
            {isGenerating ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Sparkles className="mr-2 h-4 w-4" />}
            Generate with AI
          </Button>
        </CardContent>
      </Card>

      {/* Schema Editor and Management Section */}
      <Card>
        <CardHeader>
          <CardTitle>Schema Editor & Management</CardTitle>
          <CardDescription>Edit the JSON schema directly. You can apply it to the current session or save/load from your list.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <Label htmlFor="schema-json">JSON Schema Editor</Label>
            <Textarea
              id="schema-json"
              value={localSchemaJson}
              onChange={handleSchemaEditorChange}
              rows={15}
              placeholder="Enter or generate your JSON schema here..."
              className="font-mono text-sm rounded-md shadow-sm"
              disabled={isBusy}
            />
          </div>
          <Button onClick={handleApplySchemaToSession} disabled={isBusy} variant="outline">
            {isProcessingListAction && !isGenerating ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <UploadCloud className="mr-2 h-4 w-4" />}
            Apply Editor Content to Current Session
          </Button>
        </CardContent>
      </Card>
      
      {/* Saved Schemas Section */}
      <Card>
        <CardHeader>
          <CardTitle>Saved Schemas</CardTitle>
          <CardDescription>Manage your saved schema configurations.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
           <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-end">
            <div>
              <Label htmlFor="schema-name">Schema Name</Label>
              <Input
                id="schema-name"
                placeholder="Enter a name for this schema"
                value={currentSchemaName}
                onChange={(e) => setCurrentSchemaName(e.target.value)}
                disabled={isBusy}
                className="mt-1"
              />
            </div>
            <Button onClick={handleSaveSchemaToList} disabled={isBusy || !currentSchemaName.trim() || !localSchemaJson.trim()}>
              <Save className="mr-2 h-4 w-4" />
              {selectedSchemaId && savedSchemas.find(s => s.id === selectedSchemaId && s.name === currentSchemaName.trim()) ? 'Update Schema in List' : 'Save as New to List'}
            </Button>
          </div>

          <div className="space-y-2">
            <Label htmlFor="saved-schemas-select">Load Schema from List</Label>
            <div className="flex gap-2 items-center">
              <Select onValueChange={handleSelectSchema} value={selectedSchemaId} disabled={isBusy}>
                <SelectTrigger id="saved-schemas-select" className="flex-grow">
                  <SelectValue placeholder="Select a saved schema..." />
                </SelectTrigger>
                <SelectContent>
                  {savedSchemas.length === 0 && <SelectItem value="no-schemas" disabled>No saved schemas yet</SelectItem>}
                  {savedSchemas.map((s) => (
                    <SelectItem key={s.id} value={s.id}>{s.name} <span className="text-xs text-muted-foreground ml-2">({new Date(s.createdAt).toLocaleDateString()})</span></SelectItem>
                  ))}
                </SelectContent>
              </Select>
               <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button variant="destructive" size="icon" disabled={isBusy || !selectedSchemaId} title="Delete selected schema from list">
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Are you sure?</AlertDialogTitle>
                    <AlertDialogDescription>
                      This action will permanently delete the schema
                      "{savedSchemas.find(s => s.id === selectedSchemaId)?.name || 'selected schema'}"
                      from your list. This cannot be undone.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel disabled={isBusy}>Cancel</AlertDialogCancel>
                    <AlertDialogAction onClick={handleDeleteSchemaFromList} disabled={isBusy} className="bg-destructive hover:bg-destructive/90">
                      {isProcessingListAction ? <Loader2 className="mr-2 h-4 w-4 animate-spin"/> : null}
                      Delete
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </div>
             {selectedSchemaId && (
                <p className="text-xs text-muted-foreground mt-1">
                    Selected: {currentSchemaName}. Editing this schema and saving with the same name will update it. Change the name to save as a new schema.
                </p>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

