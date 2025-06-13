'use client';

import React, { useState, useEffect } from 'react'; 
import * as XLSX from 'xlsx';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { PlayCircle, Settings2, BrainCircuit, Loader2, AlertCircle, CheckCircle, Info, Download, Sigma, Thermometer, FileClock, AlertTriangle, XCircle } from "lucide-react";
import { useConfiguration } from "@/contexts/ConfigurationContext";
import { useFiles } from "@/contexts/FileContext";
import { useJob, type ThinkingDetailLevel } from "@/contexts/JobContext";
import { useToast } from "@/hooks/use-toast";
import type { JobResult } from "@/types";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import ReactMarkdown from 'react-markdown';
import { formatDistanceToNow } from 'date-fns';
import { cn } from '@/lib/utils';


// Helper function to flatten nested objects for Excel export
const flattenObject = (obj: any, prefix: string = ''): Record<string, any> => {
  const flattened: Record<string, any> = {};
  
  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      const newKey = prefix ? `${prefix}.${key}` : key;
      
      if (obj[key] === null || obj[key] === undefined) {
        flattened[newKey] = '';
      } else if (typeof obj[key] === 'object' && !Array.isArray(obj[key])) {
        // Recursively flatten nested objects
        Object.assign(flattened, flattenObject(obj[key], newKey));
      } else if (Array.isArray(obj[key])) {
        // Handle arrays - convert to string or flatten if objects
        if (obj[key].length === 0) {
          flattened[newKey] = '';
        } else if (typeof obj[key][0] === 'object') {
          // Array of objects - create separate columns for each array item
          obj[key].forEach((item: any, index: number) => {
            if (typeof item === 'object') {
              Object.assign(flattened, flattenObject(item, `${newKey}[${index}]`));
            } else {
              flattened[`${newKey}[${index}]`] = item;
            }
          });
        } else {
          // Array of primitives - join as string
          flattened[newKey] = obj[key].join('; ');
        }
      } else {
        // Primitive value
        flattened[newKey] = obj[key];
      }
    }
  }
  
  return flattened;
};

const convertJobResultsToXlsxBuffer = (results: JobResult[]): ArrayBuffer => {
  if (!results || results.length === 0) return new ArrayBuffer(0);

  const allDataObjects: Record<string, any>[] = [];
  
  results.forEach(result => {
    const baseData: Record<string, any> = {
      jobId: result.jobId,
      fileName: result.fileName,
      status: result.status,
      error: result.error || '',
      promptTokens: result.promptTokens ?? '',
      completionTokens: result.completionTokens ?? '',
      totalTokens: result.totalTokens ?? '',
    };
    
    // Calculate thinking tokens if all required values are present
    if (result.promptTokens !== undefined && 
        result.completionTokens !== undefined && 
        result.totalTokens !== undefined &&
        result.totalTokens > (result.promptTokens + result.completionTokens)) {
      baseData.thinkingTokens = result.totalTokens - (result.promptTokens + result.completionTokens);
    } else {
      baseData.thinkingTokens = '';
    }
    
    baseData.thinkingProcess = result.thinkingProcess || '';
    
    if (result.extractedData && result.status === 'success') {
      // Simply use the extracted data as-is without validation
      allDataObjects.push({ 
        ...baseData, 
        extractedData: result.extractedData 
      });
    } else {
      allDataObjects.push(baseData); 
    }
  });

  if (allDataObjects.length === 0) return new ArrayBuffer(0);

  // Create a worksheet with better formatting
  const worksheet = XLSX.utils.json_to_sheet(allDataObjects);
  
  // Auto-size columns (basic implementation)
  const range = XLSX.utils.decode_range(worksheet['!ref'] || 'A1');
  const colWidths: any[] = [];
  
  for (let C = range.s.c; C <= range.e.c; ++C) {
    let maxWidth = 10;
    for (let R = range.s.r; R <= range.e.r; ++R) {
      const cellAddress = XLSX.utils.encode_cell({ r: R, c: C });
      const cell = worksheet[cellAddress];
      if (cell && cell.v) {
        const cellLength = cell.v.toString().length;
        maxWidth = Math.max(maxWidth, Math.min(cellLength, 50)); // Cap at 50 chars
      }
    }
    colWidths.push({ width: maxWidth });
  }
  worksheet['!cols'] = colWidths;
  
  // Create a workbook
  const workbook = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(workbook, worksheet, "Extraction Results");

  // Generate XLSX buffer
  const xlsxBuffer = XLSX.write(workbook, { bookType: 'xlsx', type: 'array' });
  return xlsxBuffer;
};


export default function RunExtractionPage() {
  const {
    schemaJson,
    systemPrompt,
    userPromptTemplate,
    examples,
    llmConfig
  } = useConfiguration();
  const { provider, model, apiKey, thinkingBudget: numericThinkingBudget, temperature } = llmConfig;
  const { files } = useFiles();
  const { 
    jobQueue, jobResults, 
    isProcessingQueue,
    thinkingEnabled, setThinkingEnabled,
    thinkingDetailLevel, setThinkingDetailLevel,
    progress, currentTask,
    processedFilesCount, failedFilesCount, totalFilesToProcess,
    currentFileProcessing, currentThinkingStream,
    startProcessingJobQueue, cancelProcessingJobQueue,
    clearJobResults,
    useAgnoProcessing, setUseAgnoProcessing,
    useCaching, setUseCaching,
    cacheStats,
    cachePricePerMillionTokens, setCachePricePerMillionTokens,
    cachePricing, updateCachePricing,
  } = useJob();
  const { toast } = useToast();

  const [showResultsSection, setShowResultsSection] = useState(false);
  const [maxRetries, setMaxRetries] = useState<number>(2);
  // Remove local useAgnoProcessing state - now using from JobContext 

  useEffect(() => {
    if (jobResults.length > 0 && !isProcessingQueue) {
      const timer = setTimeout(() => setShowResultsSection(true), 100); 
      return () => clearTimeout(timer);
    } else if (jobResults.length === 0 && !isProcessingQueue) { 
      setShowResultsSection(false);
    }
  }, [jobResults, isProcessingQueue]);

  const handleStartExtraction = () => {
    if (files.length === 0) {
      toast({ title: "No Files", description: "Please select files for extraction in File Management.", variant: "destructive" });
      return;
    }
    if (!apiKey && provider !== 'googleAI') { 
       toast({ title: "API Key Missing", description: `Please configure your API key for ${provider} in LLM Configuration.`, variant: "destructive" });
       return;
    }
    if (!schemaJson) {
      toast({ title: "Schema Missing", description: "Please define a schema in Schema Definition.", variant: "destructive" });
      return;
    }
    if (!systemPrompt || !userPromptTemplate) {
      toast({ title: "Prompts Missing", description: "Please configure prompts in Prompt Configuration.", variant: "destructive" });
      return;
    }
    
    // Log token metadata to console for debugging
    console.log("\n=== EXTRACTION JOB STARTED ===");
    console.log("Files to process:", files.length);
    console.log("Provider:", provider);
    console.log("Model:", model);
    console.log("Max Retries:", maxRetries);
    console.log("Thinking Enabled:", thinkingEnabled);
    console.log("Thinking Detail Level:", thinkingDetailLevel);
    console.log("Agno Processing Enabled:", useAgnoProcessing);
    console.log("Schema Length:", schemaJson.length);
    console.log("System Prompt Length:", systemPrompt.length);
    console.log("User Prompt Length:", userPromptTemplate.length);
    console.log("Examples Count:", examples?.length || 0);
    console.log("Caching Enabled:", useCaching);
    if (useCaching) {
      console.log("Cache Input Token Price (per 1M):", cachePricePerMillionTokens);
      const storagePrice = typeof cachePricing.cacheStoragePricePerMillionTokensPerHour === 'number'
        ? cachePricing.cacheStoragePricePerMillionTokensPerHour
        : (cachePricing.cacheStoragePricePerMillionTokensPerHour as Record<string, number>).default;
      console.log("Cache Storage Price (per 1M/hour):", storagePrice);
    }
    console.log("===========================\n");
    
    setShowResultsSection(false); 
    startProcessingJobQueue(
      files, 
      maxRetries, 
      schemaJson, 
      systemPrompt, 
      userPromptTemplate, 
      examples, 
      provider, 
      model, 
      apiKey,
      numericThinkingBudget,
      temperature,
      useAgnoProcessing
    );
  };

  const isRunDisabled = isProcessingQueue || files.length === 0 || !schemaJson || !systemPrompt || !userPromptTemplate || (!apiKey && provider !== 'googleAI');

  const handleDownloadXlsx = () => {
    if (jobResults.length === 0) {
      toast({ title: "No Data", description: "No extraction results available to download.", variant: "default" });
      return;
    }
    
    const successfulResults = jobResults.filter(r => r.status === 'success' && r.extractedData);
    if (successfulResults.length === 0 && jobResults.length > 0) {
      toast({ title: "No Successful Data", description: "No successfully extracted data available to download. XLSX will contain error details.", variant: "default" });
    }

    const xlsxBuffer = convertJobResultsToXlsxBuffer(jobResults); 
    if (!xlsxBuffer || xlsxBuffer.byteLength === 0) {
      toast({ title: "XLSX Generation Failed", description: "Could not generate XLSX data.", variant: "destructive" });
      return;
    }

    const blob = new Blob([xlsxBuffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
    const link = document.createElement("a");
    if (link.download !== undefined) { 
      const url = URL.createObjectURL(blob);
      link.setAttribute("href", url);
      link.setAttribute("download", `extraction_results_${new Date().toISOString().slice(0,10)}.xlsx`);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      toast({ title: "XLSX Downloaded", description: "Extraction results have been downloaded as XLSX." });
    } else {
       toast({ title: "Download Failed", description: "Your browser does not support this download method.", variant: "destructive"});
    }
  };

  const handleClearResults = () => {
    clearJobResults(); 
    setShowResultsSection(false);
  };

  return (
    <div className="flex-1 space-y-6 pb-8">
      <div className="flex items-center justify-between space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Run Extraction Job</h1>
      </div>

      <Card className="shadow-lg hover:shadow-xl transition-shadow duration-300">
        <CardHeader>
          <CardTitle>Initiate Data Extraction</CardTitle>
          <CardDescription>
            Combine your defined schema, prompts, selected files, and LLM settings, then start the process.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-8">
          <section className="space-y-4 p-6 border rounded-lg bg-card shadow-md">
            <h2 className="text-xl font-semibold flex items-center"><Settings2 className="mr-2 h-6 w-6 text-primary" />Configuration Summary</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <p><strong className="text-muted-foreground">Schema:</strong> {schemaJson ? <Badge variant="secondary">Defined</Badge> : <Badge variant="destructive">Not Defined</Badge>}</p>
                <p><strong className="text-muted-foreground">Prompts:</strong> {systemPrompt && userPromptTemplate ? <Badge variant="secondary">Configured</Badge> : <Badge variant="destructive">Not Configured</Badge>}</p>
                <p><strong className="text-muted-foreground">Files:</strong> {files.length > 0 ? <Badge variant="secondary">{files.length} selected</Badge> : <Badge variant="outline">None selected</Badge>}</p>
              </div>
              <div>
                <p><strong className="text-muted-foreground">LLM:</strong>
                  {provider && model ? <Badge variant="secondary">{provider} / {model}</Badge> : <Badge variant="destructive">Not Set</Badge>}
                  {apiKey || provider === 'googleAI' ? <Badge variant="secondary" className="ml-1">Key OK</Badge> : <Badge variant="destructive" className="ml-1">No API Key</Badge>}
                </p>
                {provider === 'googleAI' && numericThinkingBudget !== undefined && (
                  <p><strong className="text-muted-foreground">Thinking Budget (Model):</strong> <Badge variant="outline">{numericThinkingBudget === 0 ? 'Off' : numericThinkingBudget}</Badge></p>
                )}
                <p><strong className="text-muted-foreground">Temperature:</strong> <Badge variant="outline">{temperature.toFixed(2)}</Badge></p>
              </div>
            </div>
          </section>

          <section className="space-y-4 p-6 border rounded-lg bg-card shadow-md">
            <h2 className="text-xl font-semibold flex items-center"><PlayCircle className="mr-2 h-6 w-6 text-primary" />Job Controls</h2>
            <div className="flex flex-col sm:flex-row sm:items-center gap-4">
              <div className="flex items-center space-x-2">
                <Switch id="thinking-enabled" checked={thinkingEnabled} onCheckedChange={setThinkingEnabled} disabled={isProcessingQueue} aria-label="Enable AI Thinking Visualization"/>
                <Label htmlFor="thinking-enabled">Enable AI Thinking Visualization</Label>
              </div>
              <div className="flex items-center space-x-2">
                <Switch id="agno-processing" checked={useAgnoProcessing} onCheckedChange={setUseAgnoProcessing} disabled={isProcessingQueue} aria-label="Enable Agno AI Processing"/>
                <Label htmlFor="agno-processing">Transform Data with Agno AI</Label>
              </div>
              {thinkingEnabled && (
                <div className="w-full sm:w-auto">
                  <Label htmlFor="thinking-detail-level" className="text-sm">Explanation Detail</Label>
                  <Select value={thinkingDetailLevel} onValueChange={(value) => setThinkingDetailLevel(value as ThinkingDetailLevel)} disabled={isProcessingQueue}>
                    <SelectTrigger id="thinking-detail-level" className="w-full sm:w-[180px]"><SelectValue placeholder="Select detail level" /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="brief">Brief</SelectItem>
                      <SelectItem value="standard">Standard</SelectItem>
                      <SelectItem value="detailed">Detailed</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              )}
              <div className="w-full sm:w-auto">
                <Label htmlFor="max-retries" className="text-sm">Max Retries per File</Label>
                <Input id="max-retries" type="number" min="0" max="5" value={maxRetries} onChange={(e) => setMaxRetries(parseInt(e.target.value,10))} className="w-20" disabled={isProcessingQueue} />
              </div>
            </div>
            
            <div className="flex flex-col sm:flex-row sm:items-start gap-4 mt-4 p-3 bg-muted/20 rounded-md">
              <div className="flex flex-col gap-2">
                <div className="flex items-center space-x-2">
                  <Switch id="use-cache" checked={useCaching} onCheckedChange={setUseCaching} disabled={isProcessingQueue} aria-label="Enable Context Caching"/>
                  <Label htmlFor="use-cache">
                    <span className="font-medium text-sm">Enable Context Caching</span>
                    <span className="text-xs block text-muted-foreground">(Uses Gemini's implicit caching to reduce token usage)</span>
                  </Label>
                </div>
                <div className="text-xs text-muted-foreground pl-7">
                  <span className="font-semibold">Note:</span> Using implicit caching mechanism. Put large and common content at the beginning of prompts for best results.
                </div>
              </div>
              {useCaching && (
                <div className="flex flex-col gap-3">
                  <div className="text-sm font-medium">Pricing Configuration</div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div>
                      <Label htmlFor="input-token-price" className="text-xs flex items-center gap-1">
                        <span>Input Token Price (per 1M)</span>
                        <Info className="h-3 w-3 text-muted-foreground" />
                      </Label>
                      <Input 
                        id="input-token-price" 
                        type="number" 
                        min="0.01" 
                        step="0.01" 
                        value={isNaN(cachePricePerMillionTokens) ? "0.15" : cachePricePerMillionTokens} 
                        onChange={(e) => {
                          const value = parseFloat(e.target.value);
                          if (!isNaN(value) && value >= 0) {
                            setCachePricePerMillionTokens(value);
                          }
                        }} 
                        className="w-24 h-8 text-xs" 
                        disabled={isProcessingQueue} 
                      />
                      <p className="text-[10px] text-muted-foreground mt-1">Used for calculating token savings</p>
                    </div>
                    <div>
                      <Label htmlFor="storage-token-price" className="text-xs flex items-center gap-1">
                        <span>Storage Price (per 1M/hour)</span>
                        <Info className="h-3 w-3 text-muted-foreground" />
                      </Label>
                      <Input 
                        id="storage-token-price" 
                        type="number" 
                        min="0.01" 
                        step="0.01" 
                        value={isNaN(cachePricing.cacheStoragePricePerMillionTokensPerHour as number) ? 
                          "1.00" : 
                          typeof cachePricing.cacheStoragePricePerMillionTokensPerHour === 'number' ? 
                            cachePricing.cacheStoragePricePerMillionTokensPerHour : 
                            (cachePricing.cacheStoragePricePerMillionTokensPerHour as Record<string, number>).default || 1.00
                        } 
                        onChange={(e) => {
                          const value = parseFloat(e.target.value);
                          if (!isNaN(value) && value >= 0) {
                            updateCachePricing({
                              cacheStoragePricePerMillionTokensPerHour: value
                            });
                          }
                        }} 
                        className="w-24 h-8 text-xs" 
                        disabled={isProcessingQueue} 
                      />
                      <p className="text-[10px] text-muted-foreground mt-1">Price for cache storage</p>
                    </div>
                  </div>
                  {cacheStats && (
                    <div className="text-xs text-muted-foreground mt-2">
                      <div className="font-medium mb-1">Cache Statistics</div>
                      <div><span className="font-medium">Cache Hits:</span> {cacheStats.cacheHits}</div>
                      <div><span className="font-medium">Tokens Saved:</span> {cacheStats.tokensSaved.toLocaleString()}</div>
                      <div><span className="font-medium">Gross Saved:</span> ${(cacheStats.tokensSaved / 1000000 * cachePricePerMillionTokens).toFixed(6)}</div>
                      <div><span className="font-medium">Storage Cost:</span> ${cacheStats.storageCost.toFixed(6)}</div>
                      <div><span className="font-medium">Net Savings:</span> ${cacheStats.netSavings.toFixed(6)}</div>
                    </div>
                  )}
                </div>
              )}
            </div>
            <div className="flex gap-2 mt-4">
              <Button size="lg" className="flex-grow md:flex-grow-0 shadow-md hover:shadow-lg active:shadow-inner transition-shadow" onClick={handleStartExtraction} disabled={isRunDisabled}>
                {isProcessingQueue ? <Loader2 className="mr-2 h-5 w-5 animate-spin" /> : <PlayCircle className="mr-2 h-5 w-5" />}
                {isProcessingQueue ? 'Processing...' : `Start Extraction (${files.length} file${files.length === 1 ? '' : 's'})`}
              </Button>
              {isProcessingQueue && (
                 <Button size="lg" variant="destructive" onClick={cancelProcessingJobQueue} className="shadow-md hover:shadow-lg active:shadow-inner transition-shadow">
                    <XCircle className="mr-2 h-5 w-5" /> Cancel
                 </Button>
              )}
            </div>
            {isProcessingQueue && (
              <div className="mt-4 space-y-2">
                <Label>{currentTask}</Label>
                <Progress value={progress} className="w-full [&>div]:bg-gradient-to-r [&>div]:from-accent [&>div]:to-primary" />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Total: {totalFilesToProcess}</span>
                  <span className="text-green-600">Succeeded: {processedFilesCount}</span>
                  <span className="text-destructive">Failed: {failedFilesCount}</span>
                  <span>Pending: {jobQueue.length > 0 ? jobQueue.length : (totalFilesToProcess - processedFilesCount - failedFilesCount)}</span>
                </div>
              </div>
            )}
          </section>
          
          <div
            className={cn(
              "transition-all duration-700 ease-in-out overflow-hidden",
              thinkingEnabled && currentFileProcessing
                ? "opacity-100 max-h-[1000px] mt-6" 
                : "opacity-0 max-h-0 mt-0 pointer-events-none"
            )}
          >
            <Card className={cn("border-accent shadow-accent/20 shadow-lg")}>
              <CardHeader>
                <CardTitle className="flex items-center text-accent">
                  <BrainCircuit className="mr-2 h-6 w-6" />
                  AI Thinking Process for: {currentFileProcessing || '...'} (Live)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-72 w-full rounded-md border bg-muted/20 p-3 shadow-inner">
                  {currentThinkingStream ? (
                    <ReactMarkdown className="prose prose-sm dark:prose-invert max-w-none text-xs break-words">
                      {currentThinkingStream}
                    </ReactMarkdown>
                  ) : (
                    <p className="text-muted-foreground text-sm animate-pulse">Waiting for AI thinking explanation stream for {currentFileProcessing}...</p>
                  )}
                </ScrollArea>
              </CardContent>
            </Card>
          </div>
          
          {jobResults.length > 0 && !isProcessingQueue && (
             <section
              className={cn(
                "space-y-4 mt-6 transition-opacity duration-1000 ease-in-out",
                showResultsSection ? "opacity-100 animate-fadeIn" : "opacity-0"
              )}
            >
              <div className="flex justify-between items-center">
                <h2 className="text-xl font-semibold">Extraction Results ({jobResults.length})</h2>
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" onClick={handleDownloadXlsx} disabled={jobResults.length === 0} className="shadow-sm hover:shadow-md active:shadow-inner transition-shadow">
                    <Download className="mr-2 h-4 w-4" /> Download XLSX
                  </Button>
                  <Button variant="outline" size="sm" onClick={handleClearResults} className="shadow-sm hover:shadow-md active:shadow-inner transition-shadow">Clear Results</Button>
                </div>
              </div>
              <ScrollArea className="h-[600px] w-full rounded-md border p-4 bg-card shadow-md">
              {jobResults.slice().reverse().map((result) => ( 
                <Card key={result.jobId} className="mb-4 bg-secondary/20 shadow-sm hover:shadow-md transition-shadow duration-200">
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between text-base">
                      <div className="flex items-center">
                        {result.status === 'failed' ? <AlertTriangle className="mr-2 h-5 w-5 text-destructive animate-pulse" /> : 
                         result.status === 'retrying' ? <FileClock className="mr-2 h-5 w-5 text-amber-500 animate-spin" /> :
                         <CheckCircle className="mr-2 h-5 w-5 text-green-500" />}
                        {result.fileName}
                      </div>
                      <Badge variant={
                        result.status === 'failed' ? "destructive" :
                        result.status === 'retrying' ? "secondary" : 
                        "default" 
                        }
                      >
                        {result.status.charAt(0).toUpperCase() + result.status.slice(1)}
                      </Badge>
                    </CardTitle>
                     <p className="text-xs text-muted-foreground font-normal">
                        Job ID: {result.jobId} - {formatDistanceToNow(result.timestamp, { addSuffix: true })}
                      </p>
                    {result.error && (
                       <Alert variant="destructive" className="mt-2 shadow-sm">
                          <AlertTriangle className="h-4 w-4" />
                          <AlertTitle>Error</AlertTitle>
                          <AlertDescription>{result.error}</AlertDescription>
                       </Alert>
                    )}
                  </CardHeader>
                  {(result.extractedData || result.promptTokens !== undefined) && (
                    <CardContent className="space-y-3">
                      {result.extractedData && result.status === 'success' && (
                        <div>
                          <Label className="text-sm font-semibold">Extracted Data (JSON)</Label>
                          <ScrollArea className="h-48 mt-1 w-full rounded-md border bg-background p-2 shadow-inner">
                            <pre className="whitespace-pre-wrap text-xs font-mono break-words">
                              {result.extractedData}
                            </pre>
                          </ScrollArea>
                        </div>
                      )}
                      {(result.promptTokens !== undefined || result.completionTokens !== undefined || result.totalTokens !== undefined) && (
                        <div>
                          <Label className="text-sm font-semibold flex items-center"><Sigma className="mr-2 h-4 w-4 text-primary" />Token Usage</Label>
                          <div className="text-xs mt-1 text-muted-foreground space-y-0.5 bg-background p-2 rounded-md border shadow-inner">
                            {result.promptTokens !== undefined && <p>Input Tokens: {result.promptTokens}</p>}
                            {result.completionTokens !== undefined && <p>Output Tokens: {result.completionTokens}</p>}
                            {result.totalTokens !== undefined && <p className="font-medium">Total Tokens: {result.totalTokens}</p>}
                          </div>
                        </div>
                      )}
                    </CardContent>
                  )}
                  {result.thinkingProcess && ( 
                    <CardContent>
                       <Label className="text-sm font-semibold flex items-center"><BrainCircuit className="mr-2 h-4 w-4 text-primary" />Completed AI Thinking Explanation</Label>
                       <ScrollArea className="h-64 mt-1 w-full rounded-md border bg-background p-2 shadow-inner">
                         <ReactMarkdown className="prose prose-sm dark:prose-invert max-w-none text-xs break-words">
                           {result.thinkingProcess}
                         </ReactMarkdown>
                       </ScrollArea>
                    </CardContent>
                  )}
                   {!result.extractedData && !result.thinkingProcess && !result.error && !result.totalTokens && result.status !== 'retrying' && (
                     <CardContent>
                       <Alert variant="default" className="bg-background shadow-sm">
                         <Info className="h-4 w-4" />
                         <AlertTitle>Processing Information</AlertTitle>
                         <AlertDescription>No data, thinking process, or token usage was recorded for this file, or an issue occurred that was not caught as an error.</AlertDescription>
                       </Alert>
                     </CardContent>
                   )}
                </Card>
              ))}
              </ScrollArea>
            </section>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

