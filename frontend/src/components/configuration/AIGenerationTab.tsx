'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { useToast } from '@/hooks/use-toast';
import { 
  BrainCircuit, 
  Loader2, 
  Sparkles, 
  FileJson2, 
  MessageSquareText, 
  Lightbulb,
  History,
  Settings2,
  Save,
  ChevronRight,
  ChevronLeft,
  RotateCcw,
  CheckCircle,
  PanelLeft,
  Code
} from 'lucide-react';
import { useConfiguration, type GenerationInput } from '@/contexts/ConfigurationContext';

export function AIGenerationTab() {
  const {
    isGenerating,
    generationHistory,
    generateFromPrompt,
    schemaJson,
    setSchemaJson,
    systemPrompt,
    setSystemPrompt,
    userPromptTemplate,
    setUserPromptTemplate,
    examples,
    setExamples,
    isSchemaGenerated,
    arePromptsGenerated,
    savedSchemas,
    savedPromptSets,
    loadSchema,
    loadPromptSet,
    completeConfigurations,
    loadCompleteConfiguration,
    saveCompleteConfiguration,
    resetConfiguration,
  } = useConfiguration();

  const { toast } = useToast();

  // Generation form state
  const [userIntent, setUserIntent] = useState('');
  const [exampleCount, setExampleCount] = useState(2);
  const [includeValidation, setIncludeValidation] = useState(true);
  const [includeComprehensiveExamples, setIncludeComprehensiveExamples] = useState(true);
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);

  // UI state
  const [currentStep, setCurrentStep] = useState<1 | 2 | 3 | 4>(1);
  const [saveConfigName, setSaveConfigName] = useState('');
  const [showSavedConfigs, setShowSavedConfigs] = useState(false);

  const handleGenerate = async () => {
    if (!userIntent.trim()) {
      toast({ title: "Missing Input", description: "Please describe what you want to extract.", variant: "destructive" });
      return;
    }

    try {
      const generationInput: GenerationInput = {
        userIntent: userIntent.trim(),
        exampleCount,
        includeValidation,
        includeComprehensiveExamples,
      };

      await generateFromPrompt(generationInput);
      
      toast({ 
        title: "Generation Complete", 
        description: "AI configuration has been generated successfully!" 
      });

      // Move to next step
      setCurrentStep(2);
    } catch (error) {
      console.error('Generation failed:', error);
      toast({ 
        title: "Generation Failed", 
        description: error instanceof Error ? error.message : "An error occurred during generation.",
        variant: "destructive" 
      });
    }
  };

  const handleSaveCompleteConfig = () => {
    if (!saveConfigName.trim()) {
      toast({ title: "Missing Name", description: "Please enter a name for the configuration.", variant: "destructive" });
      return;
    }

    try {
      saveCompleteConfiguration(saveConfigName.trim());
      toast({ title: "Configuration Saved", description: `Configuration "${saveConfigName}" saved successfully.` });
      setSaveConfigName('');
    } catch (error) {
      toast({ 
        title: "Save Failed", 
        description: error instanceof Error ? error.message : "Failed to save configuration.",
        variant: "destructive" 
      });
    }
  };

  const handleReset = () => {
    resetConfiguration();
    setUserIntent('');
    setCurrentStep(1);
    toast({ title: "Configuration Reset", description: "All configuration has been reset to defaults." });
  };

  // Functions to handle loading with notifications
  const handleLoadSchema = (id: string) => {
    loadSchema(id);
    toast({ 
      title: "Schema Loaded", 
      description: "Schema loaded successfully.",
      variant: "default"
    });
  };

  const handleLoadPromptSet = (id: string) => {
    loadPromptSet(id);
    toast({ 
      title: "Prompts Loaded", 
      description: "Prompt set loaded successfully.",
      variant: "default"
    });
  };

  const handleLoadCompleteConfig = (id: string) => {
    loadCompleteConfiguration(id);
    setShowSavedConfigs(false);
    setCurrentStep(4); // Set to review step after loading
    toast({ 
      title: "Configuration Loaded", 
      description: "Complete configuration loaded successfully. Please review it before proceeding.",
      variant: "default"
    });
  };

  const isConfigComplete = schemaJson && systemPrompt && userPromptTemplate;

  return (
    <div className="space-y-6">
      {/* Progress Indicator */}
      <div className="w-full bg-muted rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className={`w-8 h-8 rounded-full flex items-center justify-center ${currentStep === 1 ? 'bg-primary text-primary-foreground' : (isSchemaGenerated || schemaJson ? 'bg-green-500 text-white' : 'bg-muted-foreground/20 text-muted-foreground')}`}>
              {isSchemaGenerated || (schemaJson && currentStep !== 1) ? <CheckCircle className="h-4 w-4" /> : 1}
            </div>
            <span className={currentStep === 1 ? 'font-medium' : 'text-muted-foreground'}>Define</span>
          </div>
          <div className="flex-1 h-1 mx-2 bg-border">
            <div className={`h-full bg-primary ${currentStep >= 2 ? 'w-full' : 'w-0'} transition-all duration-300`}></div>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-8 h-8 rounded-full flex items-center justify-center ${currentStep === 2 ? 'bg-primary text-primary-foreground' : (arePromptsGenerated || (systemPrompt && userPromptTemplate) ? 'bg-green-500 text-white' : 'bg-muted-foreground/20 text-muted-foreground')}`}>
              {arePromptsGenerated || (systemPrompt && userPromptTemplate && currentStep !== 2) ? <CheckCircle className="h-4 w-4" /> : 2}
            </div>
            <span className={currentStep === 2 ? 'font-medium' : 'text-muted-foreground'}>Schema</span>
          </div>
          <div className="flex-1 h-1 mx-2 bg-border">
            <div className={`h-full bg-primary ${currentStep >= 3 ? 'w-full' : 'w-0'} transition-all duration-300`}></div>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-8 h-8 rounded-full flex items-center justify-center ${currentStep === 3 ? 'bg-primary text-primary-foreground' : (examples.length > 0 ? 'bg-green-500 text-white' : 'bg-muted-foreground/20 text-muted-foreground')}`}>
              {examples.length > 0 && currentStep !== 3 ? <CheckCircle className="h-4 w-4" /> : 3}
            </div>
            <span className={currentStep === 3 ? 'font-medium' : 'text-muted-foreground'}>Prompts</span>
          </div>
          <div className="flex-1 h-1 mx-2 bg-border">
            <div className={`h-full bg-primary ${currentStep >= 4 ? 'w-full' : 'w-0'} transition-all duration-300`}></div>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-8 h-8 rounded-full flex items-center justify-center ${currentStep === 4 ? 'bg-primary text-primary-foreground' : 'bg-muted-foreground/20 text-muted-foreground'}`}>
              4
            </div>
            <span className={currentStep === 4 ? 'font-medium' : 'text-muted-foreground'}>Review</span>
          </div>
        </div>
      </div>

      {/* Saved Configurations Panel */}
      {showSavedConfigs && (
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle>Saved Configurations</CardTitle>
              <Button variant="ghost" size="sm" onClick={() => setShowSavedConfigs(false)}>
                <PanelLeft className="h-4 w-4" />
                Close
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {completeConfigurations.length > 0 ? (
                completeConfigurations.map((config) => (
                  <div key={config.id} className="flex items-center justify-between p-3 border rounded-md hover:bg-muted/50 transition-colors">
                    <div>
                      <div className="font-medium">{config.name}</div>
                      <div className="text-xs text-muted-foreground">
                        {new Date(config.createdAt).toLocaleDateString()}
                      </div>
                    </div>
                    <Button size="sm" onClick={() => handleLoadCompleteConfig(config.id)}>
                      Load
                    </Button>
                  </div>
                ))
              ) : (
                <div className="text-center py-6 text-muted-foreground">
                  No saved configurations found
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 1: Define Extraction */}
      {currentStep === 1 && (
        <Card className="border-primary/20 shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-primary" />
              Define What to Extract
            </CardTitle>
            <CardDescription>
              Describe what you want to extract and let AI generate the complete configuration
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="user-intent">What do you want to extract? *</Label>
                <Textarea
                  id="user-intent"
                  placeholder="Describe your extraction goal in detail. For example: 'Extract customer information from invoices including company name, invoice number, date, total amount, and line items with descriptions, quantities and prices.'"
                  value={userIntent}
                  onChange={(e) => setUserIntent(e.target.value)}
                  className="min-h-[100px]"
                  disabled={isGenerating}
                />
              </div>

              <div className="flex items-center gap-2">
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
                >
                  {showAdvancedOptions ? 'Hide' : 'Show'} Advanced Options
                </Button>
              </div>

              {showAdvancedOptions && (
                <div className="space-y-4 p-4 border rounded-md bg-muted/30">
                  <div className="grid grid-cols-1 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="example-count">Examples to Generate</Label>
                      <Input
                        id="example-count"
                        type="number"
                        min={1}
                        max={5}
                        value={exampleCount}
                        onChange={(e) => setExampleCount(parseInt(e.target.value) || 2)}
                        disabled={isGenerating}
                      />
                    </div>
                    <div className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        id="include-comprehensive-examples"
                        checked={includeComprehensiveExamples}
                        onChange={(e) => setIncludeComprehensiveExamples(e.target.checked)}
                        disabled={isGenerating}
                        className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
                      />
                      <Label htmlFor="include-comprehensive-examples" className="text-sm font-medium">
                        Include comprehensive examples for various document types
                      </Label>
                    </div>
                  </div>
                </div>
              )}

              <div className="flex items-center justify-between pt-4">
                <Button 
                  variant="outline" 
                  onClick={() => setShowSavedConfigs(true)}
                  disabled={isGenerating}
                >
                  Load Saved Config
                </Button>
                <Button 
                  onClick={handleGenerate} 
                  disabled={isGenerating || !userIntent.trim()}
                  size="lg"
                >
                  {isGenerating ? (
                    <>
                      <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <BrainCircuit className="mr-2 h-5 w-5" />
                      Generate Configuration
                    </>
                  )}
                </Button>
              </div>
            </div>

            {generationHistory.length > 0 && (
              <div className="pt-4">
                <Separator className="my-4" />
                <div className="space-y-2">
                  <Label className="flex items-center gap-2">
                    <History className="h-4 w-4" />
                    Recent Generations
                  </Label>
                  <ScrollArea className="h-32 w-full rounded-md border p-2">
                    {generationHistory.slice(0, 5).map((result) => (
                      <div key={result.id} className="text-xs p-2 border-b last:border-b-0 hover:bg-muted/50 cursor-pointer" onClick={() => setUserIntent(result.input.userIntent)}>
                        <div className="font-medium">{result.input.userIntent.substring(0, 60)}...</div>
                        <div className="text-muted-foreground">
                          {new Date(result.timestamp).toLocaleTimeString()}
                        </div>
                      </div>
                    ))}
                  </ScrollArea>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Step 2: Schema */}
      {currentStep === 2 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span className="flex items-center gap-2">
                <FileJson2 className="h-5 w-5" />
                JSON Schema
              </span>
              {isSchemaGenerated && (
                <Badge variant="outline" className="flex items-center gap-1">
                  <BrainCircuit className="h-3 w-3" />
                  AI Generated
                </Badge>
              )}
            </CardTitle>
            <CardDescription>
              Define the structure of data to extract from documents
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="schema-editor">Schema Definition</Label>
                <div className="flex items-center gap-2">
                  <Select onValueChange={(id) => handleLoadSchema(id)}>
                    <SelectTrigger className="w-[200px]">
                      <SelectValue placeholder="Load saved schema" />
                    </SelectTrigger>
                    <SelectContent>
                      {savedSchemas.map((schema) => (
                        <SelectItem key={schema.id} value={schema.id}>
                          {schema.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <Textarea
                id="schema-editor"
                value={schemaJson}
                onChange={(e) => setSchemaJson(e.target.value)}
                className="font-mono text-sm min-h-[300px]"
                placeholder="JSON schema will appear here after generation..."
              />
            </div>

            <div className="flex items-center justify-between pt-4">
              <div className="flex items-center gap-2">
                <Button variant="outline" onClick={() => setCurrentStep(1)}>
                  <ChevronLeft className="mr-2 h-4 w-4" />
                  Back
                </Button>
                <Button variant="outline" onClick={handleReset}>
                  <RotateCcw className="mr-2 h-4 w-4" />
                  Reset
                </Button>
              </div>
              <Button 
                onClick={() => setCurrentStep(3)} 
                disabled={!schemaJson}
              >
                Continue
                <ChevronRight className="ml-2 h-4 w-4" />
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 3: Prompts */}
      {currentStep === 3 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span className="flex items-center gap-2">
                <MessageSquareText className="h-5 w-5" />
                Prompts & Examples
              </span>
              {arePromptsGenerated && (
                <Badge variant="outline" className="flex items-center gap-1">
                  <BrainCircuit className="h-3 w-3" />
                  AI Generated
                </Badge>
              )}
            </CardTitle>
            <CardDescription>
              Configure the prompts that guide the AI extraction process
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="flex items-center justify-between">
              <Label>Prompt Configuration</Label>
              <Select onValueChange={(id) => handleLoadPromptSet(id)}>
                <SelectTrigger className="w-[200px]">
                  <SelectValue placeholder="Load saved prompts" />
                </SelectTrigger>
                <SelectContent>
                  {savedPromptSets.map((promptSet) => (
                    <SelectItem key={promptSet.id} value={promptSet.id}>
                      {promptSet.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>System Prompt</Label>
              <Textarea
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                className="min-h-[120px]"
                placeholder="System prompt will be generated here..."
              />
            </div>

            <div className="space-y-2">
              <Label>User Prompt Template</Label>
              <Textarea
                value={userPromptTemplate}
                onChange={(e) => setUserPromptTemplate(e.target.value)}
                className="min-h-[120px]"
                placeholder="User prompt template will be generated here..."
              />
            </div>

            <div className="space-y-2">
              <Label>Examples ({examples.length})</Label>
              {examples.length > 0 ? (
                <div className="space-y-4 max-h-[300px] overflow-y-auto pr-2">
                  {examples.map((example, index) => (
                    <div key={index} className="p-3 border rounded-lg space-y-2 bg-muted/30">
                      <div>
                        <Label className="text-xs text-muted-foreground">Input</Label>
                        <div className="text-sm p-2 rounded bg-background">{example.input}</div>
                      </div>
                      <div>
                        <Label className="text-xs text-muted-foreground">Output</Label>
                        <div className="text-sm p-2 rounded bg-background font-mono">{example.output}</div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center text-muted-foreground py-4 border rounded-md">
                  No examples available
                </div>
              )}
            </div>

            <div className="flex items-center justify-between pt-4">
              <Button variant="outline" onClick={() => setCurrentStep(2)}>
                <ChevronLeft className="mr-2 h-4 w-4" />
                Back
              </Button>
              <Button 
                onClick={() => setCurrentStep(4)} 
                disabled={!systemPrompt || !userPromptTemplate}
              >
                Continue
                <ChevronRight className="ml-2 h-4 w-4" />
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 4: Review & Save */}
      {currentStep === 4 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings2 className="h-5 w-5" />
              Review & Save Configuration
            </CardTitle>
            <CardDescription>
              Review your complete configuration and save it for future use
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-4">
                <Accordion type="single" collapsible className="w-full">
                  <AccordionItem value="schema">
                    <AccordionTrigger className="p-4 border rounded-md bg-muted/30 hover:no-underline hover:bg-muted/50">
                      <div className="flex items-center justify-between w-full">
                        <h3 className="font-medium flex items-center gap-2">
                          <FileJson2 className="h-4 w-4" />
                          Schema
                        </h3>
                        <Badge variant={schemaJson ? "default" : "secondary"} className="mr-2">
                          {schemaJson ? "Defined" : "Not Set"}
                        </Badge>
                      </div>
                    </AccordionTrigger>
                    <AccordionContent className="border-x border-b rounded-b-md bg-muted/10 p-4">
                      {schemaJson ? (
                        <div className="max-h-[300px] overflow-y-auto">
                          <pre className="text-xs font-mono p-2 bg-muted/20 rounded whitespace-pre-wrap">
                            {schemaJson}
                          </pre>
                        </div>
                      ) : (
                        <div className="text-center text-muted-foreground py-4">
                          No schema defined
                        </div>
                      )}
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value="prompts" className="mt-4">
                    <AccordionTrigger className="p-4 border rounded-md bg-muted/30 hover:no-underline hover:bg-muted/50">
                      <div className="flex items-center justify-between w-full">
                        <h3 className="font-medium flex items-center gap-2">
                          <MessageSquareText className="h-4 w-4" />
                          Prompts
                        </h3>
                        <Badge variant={systemPrompt && userPromptTemplate ? "default" : "secondary"} className="mr-2">
                          {systemPrompt && userPromptTemplate ? "Configured" : "Not Set"}
                        </Badge>
                      </div>
                    </AccordionTrigger>
                    <AccordionContent className="border-x border-b rounded-b-md bg-muted/10 p-4">
                      {systemPrompt && userPromptTemplate ? (
                        <div className="space-y-4 max-h-[300px] overflow-y-auto">
                          <div className="space-y-2">
                            <Label className="text-xs text-muted-foreground">System Prompt</Label>
                            <div className="text-sm p-2 rounded bg-background whitespace-pre-wrap">{systemPrompt}</div>
                          </div>
                          <div className="space-y-2">
                            <Label className="text-xs text-muted-foreground">User Template</Label>
                            <div className="text-sm p-2 rounded bg-background whitespace-pre-wrap">{userPromptTemplate}</div>
                          </div>
                        </div>
                      ) : (
                        <div className="text-center text-muted-foreground py-4">
                          Prompts not configured
                        </div>
                      )}
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value="examples" className="mt-4">
                    <AccordionTrigger className="p-4 border rounded-md bg-muted/30 hover:no-underline hover:bg-muted/50">
                      <div className="flex items-center justify-between w-full">
                        <h3 className="font-medium flex items-center gap-2">
                          <Lightbulb className="h-4 w-4" />
                          Examples
                        </h3>
                        <Badge variant={examples.length > 0 ? "default" : "secondary"} className="mr-2">
                          {examples.length} examples
                        </Badge>
                      </div>
                    </AccordionTrigger>
                    <AccordionContent className="border-x border-b rounded-b-md bg-muted/10 p-4">
                      {examples.length > 0 ? (
                        <div className="space-y-4 max-h-[300px] overflow-y-auto pr-2">
                          {examples.map((example, index) => (
                            <div key={index} className="p-3 border rounded-lg space-y-2 bg-muted/30">
                              <div>
                                <Label className="text-xs text-muted-foreground">Input</Label>
                                <div className="text-sm p-2 rounded bg-background">{example.input}</div>
                              </div>
                              <div>
                                <Label className="text-xs text-muted-foreground">Output</Label>
                                <div className="text-sm p-2 rounded bg-background font-mono">{example.output}</div>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="text-center text-muted-foreground py-4">
                          No examples available
                        </div>
                      )}
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value="status" className="mt-4">
                    <AccordionTrigger className="p-4 border rounded-md bg-muted/30 hover:no-underline hover:bg-muted/50">
                      <div className="flex items-center justify-between w-full">
                        <h3 className="font-medium flex items-center gap-2">
                          <Settings2 className="h-4 w-4" />
                          Status
                        </h3>
                        <Badge variant={isConfigComplete ? "default" : "destructive"} className="mr-2">
                          {isConfigComplete ? "Ready" : "Incomplete"}
                        </Badge>
                      </div>
                    </AccordionTrigger>
                    <AccordionContent className="border-x border-b rounded-b-md bg-muted/10 p-4">
                      {isConfigComplete ? (
                        <div className="text-sm">
                          <p>Configuration is complete and ready for extraction.</p>
                          <p className="mt-2">You can now save this configuration and use it to process files.</p>
                        </div>
                      ) : (
                        <div className="text-sm">
                          <p>Some required components are missing:</p>
                          <ul className="list-disc list-inside mt-2">
                            {!schemaJson && <li>Schema definition is required</li>}
                            {!systemPrompt && <li>System prompt is required</li>}
                            {!userPromptTemplate && <li>User prompt template is required</li>}
                          </ul>
                        </div>
                      )}
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>
              </div>

              <div className="space-y-4 pt-4">
                <Separator />
                <div className="space-y-2">
                  <Label htmlFor="config-name">Configuration Name</Label>
                  <div className="flex items-center gap-2">
                    <Input
                      id="config-name"
                      placeholder="Enter a name for this configuration"
                      value={saveConfigName}
                      onChange={(e) => setSaveConfigName(e.target.value)}
                    />
                    <Button 
                      onClick={handleSaveCompleteConfig} 
                      disabled={!saveConfigName.trim() || !isConfigComplete}
                    >
                      <Save className="mr-2 h-4 w-4" />
                      Save
                    </Button>
                  </div>
                </div>

                {isConfigComplete && (
                  <Alert className="mt-4">
                    <CheckCircle className="h-4 w-4" />
                    <AlertTitle>Configuration Complete</AlertTitle>
                    <AlertDescription>
                      Your extraction configuration is ready! You can now proceed to upload files and run extractions.
                    </AlertDescription>
                  </Alert>
                )}
              </div>

              <div className="flex items-center justify-between pt-4">
                <div className="flex items-center gap-2">
                  <Button variant="outline" onClick={() => setCurrentStep(3)}>
                    <ChevronLeft className="mr-2 h-4 w-4" />
                    Back
                  </Button>
                  <Button variant="outline" onClick={handleReset}>
                    <RotateCcw className="mr-2 h-4 w-4" />
                    Reset All
                  </Button>
                </div>
                <Button 
                  variant="outline" 
                  onClick={() => setShowSavedConfigs(true)}
                >
                  Manage Saved Configs
                </Button>
              </div>
            </CardContent>
        </Card>
      )}
    </div>
  );
}