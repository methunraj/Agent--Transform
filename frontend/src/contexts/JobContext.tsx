'use client';
import type { Dispatch, SetStateAction } from 'react';
import { createContext, useContext, useState, useMemo, useCallback, useEffect } from 'react';
import type { JobResult, AppFile, AppFileWithRetry } from '@/types';
import { useToast } from '@/hooks/use-toast';
import { extractData, type ExtractDataInput } from '@/ai/flows/extract-data-flow';
import { visualizeThinking, type VisualizeThinkingInput, type VisualizeThinkingStreamChunk, type VisualizeThinkingOutput } from '@/ai/flows/visualize-thinking';
import * as CachingService from '@/ai/caching-service';
import { v4 as uuidv4 } from 'uuid';
import type { CachePricing } from '@/ai/caching-service';

export type ThinkingDetailLevel = 'brief' | 'standard' | 'detailed'; 

interface JobContextType {
  jobQueue: AppFileWithRetry[];
  jobResults: JobResult[];
  clearJobResults: () => void;
  
  isProcessingQueue: boolean;
  
  thinkingEnabled: boolean;
  setThinkingEnabled: Dispatch<SetStateAction<boolean>>;
  thinkingDetailLevel: ThinkingDetailLevel;
  setThinkingDetailLevel: Dispatch<SetStateAction<ThinkingDetailLevel>>;

  // Agno AI processing
  useAgnoProcessing: boolean;
  setUseAgnoProcessing: Dispatch<SetStateAction<boolean>>;

  progress: number; 
  currentTask: string;
  
  processedFilesCount: number;
  failedFilesCount: number;
  totalFilesToProcess: number;
  
  currentFileProcessing: string | null; 
  currentThinkingStream: string; 

  // Caching related properties
  useCaching: boolean;
  setUseCaching: Dispatch<SetStateAction<boolean>>;
  cacheStats: CachingService.CacheUsageStats | null;
  cachePricePerMillionTokens: number;
  setCachePricePerMillionTokens: Dispatch<SetStateAction<number>>;
  cachePricing: CachePricing;
  updateCachePricing: (pricing: Partial<CachePricing>) => Promise<void>;

  startProcessingJobQueue: (
    filesToProcess: AppFile[], 
    maxRetriesPerFile: number,
    schemaJson: string,
    systemPrompt: string,
    userPromptTemplate: string,
    examples: any[], 
    llmProvider: string,
    model: string,
    llmApiKey: string, 
    llmNumericThinkingBudget: number | undefined,
    llmTemperature: number,
    useAgnoProcessing?: boolean
  ) => void;
  cancelProcessingJobQueue: () => void;
}

const JobContext = createContext<JobContextType | undefined>(undefined);

const getInitialThinkingDetailLevel = (): ThinkingDetailLevel => {
  if (typeof process === 'undefined' || typeof process.env === 'undefined') return 'standard';
  const envBudget = process.env.NEXT_PUBLIC_DEFAULT_THINKING_BUDGET;
  if (envBudget === 'brief' || envBudget === 'standard' || envBudget === 'detailed') {
    return envBudget;
  }
  return 'standard'; 
};

const mapDetailLevelToNumericBudget = (level: ThinkingDetailLevel): number => {
  switch (level) {
    case 'brief': return 1024; 
    case 'standard': return 4096;
    case 'detailed': return 8192;
    default: return 4096; 
  }
};

export function JobProvider({ children }: { children: React.ReactNode }) {
  const [jobQueue, setJobQueue] = useState<AppFileWithRetry[]>([]);
  const [jobResults, setJobResults] = useState<JobResult[]>([]);
  const [isProcessingQueue, setIsProcessingQueue] = useState<boolean>(false);
  const [thinkingEnabled, setThinkingEnabled] = useState<boolean>(false);
  const [thinkingDetailLevel, setThinkingDetailLevel] = useState<ThinkingDetailLevel>(getInitialThinkingDetailLevel());
  const [useAgnoProcessing, setUseAgnoProcessing] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);
  const [currentTask, setCurrentTask] = useState<string>('');
  
  const [processedFilesCount, setProcessedFilesCount] = useState<number>(0);
  const [failedFilesCount, setFailedFilesCount] = useState<number>(0);
  const [totalFilesToProcess, setTotalFilesToProcess] = useState<number>(0);

  const [currentFileProcessing, setCurrentFileProcessing] = useState<string | null>(null);
  const [currentThinkingStream, setCurrentThinkingStream] = useState<string>('');
  
  const [isCancelled, setIsCancelled] = useState(false);
  
  // Add caching state
  const [useCaching, setUseCaching] = useState<boolean>(false);
  const [cachePricePerMillionTokens, setCachePricePerMillionTokens] = useState<number>(0.15); // Default price
  const [currentCacheId, setCurrentCacheId] = useState<string | undefined>(undefined);
  const [loadedCacheStats, setLoadedCacheStats] = useState<CachingService.CacheUsageStats | null>(null);

  // Add cache pricing state
  const [cachePricing, setCachePricing] = useState<CachePricing>({
    cacheStoragePricePerMillionTokensPerHour: { default: 1.00 },
    cacheCreationPricePerMillionTokens: { default: 0.025 },
    inputTokenPricePerMillionTokens: { default: 0.15 }
  });

  const { toast } = useToast();

  const addJobResult = useCallback((result: JobResult) => {
    setJobResults(prevResults => {
      const existingIndex = prevResults.findIndex(r => r.jobId === result.jobId);
      if (existingIndex > -1) {
        const updatedResults = [...prevResults];
        updatedResults[existingIndex] = result;
        return updatedResults;
      }
      return [...prevResults, result];
    });
  }, []);
  
  const clearJobResults = useCallback(() => {
    setJobResults([]);
    setProcessedFilesCount(0);
    setFailedFilesCount(0);
    setTotalFilesToProcess(0);
    setProgress(0);
    setCurrentTask('');
    setCurrentFileProcessing(null);
    setCurrentThinkingStream('');
  }, []);

  const cancelProcessingJobQueue = useCallback(() => {
    setIsCancelled(true);
    setCurrentTask("Processing cancelled by user.");
    setIsProcessingQueue(false); 
    setJobQueue([]); 
  }, []);

  // Load cache stats when needed
  useEffect(() => {
    let isMounted = true;
    
    const loadStats = async () => {
      try {
        const stats = await CachingService.getCacheStats();
        if (isMounted) {
          setLoadedCacheStats(stats);
        }
      } catch (err) {
        console.error("Failed to load cache stats:", err);
      }
    };
    
    loadStats();
    
    return () => {
      isMounted = false;
    };
  }, [jobResults.length, isProcessingQueue]);
  
  // Use the loaded stats for the UI
  const cacheStats = loadedCacheStats;

  // Load cache pricing when needed
  useEffect(() => {
    let isMounted = true;
    
    const loadPricing = async () => {
      try {
        // Get the current pricing using the async getter function
        const pricing = await CachingService.getCachePricing();
        if (isMounted && pricing) {
          // Set the cache pricing state
          setCachePricing(pricing);
          
          // Update the local price state to match the global configuration
          const price = typeof pricing.inputTokenPricePerMillionTokens === 'number' 
            ? pricing.inputTokenPricePerMillionTokens 
            : (pricing.inputTokenPricePerMillionTokens as Record<string, number>).default || 0.15;
          setCachePricePerMillionTokens(price);
        }
      } catch (err) {
        console.error("Failed to load cache pricing:", err);
      }
    };
    
    loadPricing();
    
    return () => {
      isMounted = false;
    };
  }, []);

  // Function to update cache pricing
  const updateCachePricing = useCallback(async (pricing: Partial<CachePricing>) => {
    try {
      const updatedPricing = await CachingService.updateCachePricing(pricing);
      setCachePricing(updatedPricing);
      
      // If input token price is being updated, update our local state too
      if (pricing.inputTokenPricePerMillionTokens !== undefined) {
        const price = typeof pricing.inputTokenPricePerMillionTokens === 'number' 
          ? pricing.inputTokenPricePerMillionTokens 
          : (pricing.inputTokenPricePerMillionTokens as Record<string, number>).default || 0.15;
        setCachePricePerMillionTokens(price);
      }
      
      console.log("[Job Context] Updated cache pricing configuration");
    } catch (err) {
      console.error("Error updating cache pricing:", err);
      toast({ 
        title: "Cache Pricing Update Failed", 
        description: "Could not update cache pricing configuration. See console for details.", 
        variant: "destructive" 
      });
    }
  }, [toast]);

  const startProcessingJobQueue = useCallback(async (
    filesToProcess: AppFile[], 
    maxRetriesPerFile: number,
    schemaJson: string,
    systemPrompt: string,
    userPromptTemplate: string,
    examples: any[],
    llmProvider: string,
    model: string,
    llmApiKey: string,
    llmNumericThinkingBudget: number | undefined,
    llmTemperature: number,
    useAgnoProcessing: boolean = false
  ) => {
    console.log("Provider:", llmProvider);
    console.log("Model:", model);
    console.log("Max Retries:", maxRetriesPerFile);
    console.log("Thinking Enabled:", thinkingEnabled);
    console.log("Thinking Detail Level:", thinkingDetailLevel);
    console.log("Agno Processing Enabled:", useAgnoProcessing);
    console.log("Caching Enabled:", useCaching);
    console.log("Cache Price Per Million Tokens:", `$${cachePricePerMillionTokens.toFixed(2)}`);
    console.log("Schema Length:", schemaJson.length);
    console.log("System Prompt Length:", systemPrompt.length);
    console.log("User Prompt Length:", userPromptTemplate.length);
    console.log("Examples Count:", examples?.length || 0);
    console.log("===========================\n");
    
    setIsProcessingQueue(true);
    setIsCancelled(false);
    clearJobResults();
    
    // Reset current cache ID when starting a new job only if we're not reusing an existing cache
    if (!useCaching) {
      setCurrentCacheId(undefined);
      console.log("[Cache] Caching disabled, resetting cache ID");
    } else {
      console.log(`[Cache] Caching enabled, ${currentCacheId ? 'reusing cache ID: ' + currentCacheId : 'will create new cache'}`);
    }
    
    const initialQueue: AppFileWithRetry[] = filesToProcess.map(file => ({ ...file, retryCount: 0, jobId: uuidv4() }));
    setJobQueue(initialQueue);
    setTotalFilesToProcess(filesToProcess.length);
    
    let filesSuccessfullyProcessedCount = 0;
    let filesFailedPermanentlyCount = 0;
    let distinctFilesAttemptedOrCompleted = 0;

    let currentProcessingQueue = [...initialQueue];

    while (currentProcessingQueue.length > 0) {
      if (isCancelled) break;

      const fileJob = currentProcessingQueue.shift(); 
      if (!fileJob) break;

      if (fileJob.retryCount === 0) {
        distinctFilesAttemptedOrCompleted++;
      }
      
      setCurrentFileProcessing(fileJob.name);
      setCurrentThinkingStream('');
      setCurrentTask(`Processing ${fileJob.name} (Attempt ${fileJob.retryCount + 1})...`);
      setProgress(Math.round((distinctFilesAttemptedOrCompleted / filesToProcess.length) * 100));
      
      let extractedDataJson: string | null = null;
      let finalThinkingProcess: string | null = null; 
      let errorMsg: string | undefined;
      let tokens: { promptTokens?: number; completionTokens?: number; totalTokens?: number; estimatedTokens?: number; tokenBreakdown?: Record<string, number> } = {};
      let jobStatus: JobResult['status'] = 'retrying';

      try {
        const extractionInput: ExtractDataInput = {
          documentFile: {
            name: fileJob.name,
            type: fileJob.type,
            dataUri: fileJob.dataUri,
            textContent: fileJob.textContent,
          },
          schemaDefinition: schemaJson,
          systemPrompt: systemPrompt,
          userTaskDescription: userPromptTemplate,
          examples: examples,
          llmProvider: llmProvider,
          modelName: model,
          numericThinkingBudget: llmNumericThinkingBudget,
          temperature: llmTemperature,
          // Add cache parameters
          useCache: useCaching,
          cacheId: currentCacheId, // Use existing cache ID if available
          cachePricePerMillionTokens: cachePricePerMillionTokens, // Add price for token calculation
        };
        
        setCurrentTask(`Extracting data from ${fileJob.name}...`);
        const extractionOutput = await extractData(extractionInput);
        
        console.log(`\n=== EXTRACTION OUTPUT DEBUG ===`);
        console.log(`File: ${fileJob.name}`);
        console.log(`extractionOutput exists: ${!!extractionOutput}`);
        console.log(`extractedJson exists: ${!!extractionOutput?.extractedJson}`);
        console.log(`extractedJson type: ${typeof extractionOutput?.extractedJson}`);
        console.log(`extractedJson length: ${extractionOutput?.extractedJson?.length || 0}`);
        if (extractionOutput?.extractedJson) {
          console.log(`First 100 chars: ${extractionOutput.extractedJson.substring(0, 100)}...`);
        }
        console.log(`==============================\n`);
        
        extractedDataJson = extractionOutput.extractedJson;
        tokens = {
          promptTokens: extractionOutput.promptTokens,
          completionTokens: extractionOutput.completionTokens,
          totalTokens: extractionOutput.totalTokens,
          estimatedTokens: extractionOutput.estimatedTokens,
          tokenBreakdown: extractionOutput.breakdownByType,
        };
        
        // If using cache and extraction was successful, capture the cache ID for next file
        if (useCaching && extractionOutput.cacheSavingsInfo?.cacheId) {
          setCurrentCacheId(extractionOutput.cacheSavingsInfo.cacheId);
          
          const cacheHit = extractionOutput.cacheSavingsInfo.cacheHit;
          const tokensSaved = extractionOutput.cacheSavingsInfo.tokensSaved;
          const cachedTokens = extractionOutput.cacheSavingsInfo.cachedTokens;
          
          console.log(`\n=== CACHE RESULTS ===`);
          console.log(`File: ${fileJob.name}`);
          console.log(`Cache Hit: ${cacheHit ? 'Yes' : 'No'}`);
          console.log(`Cache ID: ${extractionOutput.cacheSavingsInfo.cacheId}`);
          
          if (cacheHit && tokensSaved) {
            const savingsAmount = (tokensSaved / 1000000) * cachePricePerMillionTokens;
            console.log(`Tokens Saved: ${tokensSaved.toLocaleString()}`);
            console.log(`Estimated Savings: $${savingsAmount.toFixed(6)}`);
          }
          
          if (cachedTokens) {
            console.log(`Cached Tokens: ${cachedTokens.toLocaleString()}`);
          }
          
          console.log(`Expiration: ${extractionOutput.cacheSavingsInfo.expireTime}`);
          console.log(`Using for next extraction: ${extractionOutput.cacheSavingsInfo.cacheId}`);
          console.log(`=====================\n`);
        }

        if (thinkingEnabled) {
          setCurrentTask(`Visualizing thinking for ${fileJob.name}...`);
          let accumulatedThinkingForFile = "";
          
          const thinkingQuery = `Task: ${userPromptTemplate.substring(0,150)}...\nSchema: ${schemaJson.substring(0, 150)}...\nDocument: ${fileJob.name} (${fileJob.type})`;
          const explanationBudgetForViz = mapDetailLevelToNumericBudget(thinkingDetailLevel);
          
          const visualizeInput: VisualizeThinkingInput = { 
            query: thinkingQuery, 
            explanationDetailBudget: explanationBudgetForViz,
            llmProvider: llmProvider, 
            modelName: model, 
            temperature: llmTemperature,
          };

          try {
            // Skip streaming callback for now to avoid type errors
            const flowResultPromise = visualizeThinking(visualizeInput);
            
            // Handle the final result only
            const finalVisualizationOutput = await flowResultPromise;
            finalThinkingProcess = finalVisualizationOutput.thinkingProcess || "";

            // Set the thinking stream once at the end
            setCurrentThinkingStream(finalVisualizationOutput.thinkingProcess || "");
            accumulatedThinkingForFile = finalVisualizationOutput.thinkingProcess || "";

          } catch (visError) {
            console.error(`Thinking visualization error for ${fileJob.name}:`, visError);
            const visErrorMsg = visError instanceof Error ? visError.message : "Unknown visualization error.";
            const errorText = `\n\n--- Error during thinking visualization: ${visErrorMsg} ---`;
            setCurrentThinkingStream(prev => prev + errorText);
            finalThinkingProcess = accumulatedThinkingForFile + errorText;
          }
        }
        
        // Agno AI Processing (if enabled and extraction was successful)
        console.log(`\n=== AGNO PROCESSING CHECK ===`);
        console.log(`useAgnoProcessing: ${useAgnoProcessing}`);
        console.log(`extractedDataJson: ${extractedDataJson ? 'Has data' : 'NULL/UNDEFINED'}`);
        console.log(`extractedDataJson length: ${extractedDataJson?.length || 0}`);
        console.log(`============================\n`);
        
        if (useAgnoProcessing && extractedDataJson) {
          try {
            setCurrentTask(`Enhancing data with Agno AI for ${fileJob.name}...`);
            
            // Create AbortController for 20 minute timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 20 * 60 * 1000); // 20 minutes
            
            const agnoResponse = await fetch('/api/agno-process', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                extractedData: extractedDataJson,
                fileName: fileJob.name,
                llmProvider: llmProvider,
                model: model,
                apiKey: llmApiKey,
                temperature: llmTemperature
              }),
              signal: controller.signal,
            });
            
            // Clear timeout if request completes before timeout
            clearTimeout(timeoutId);
            
            if (agnoResponse.ok) {
              const agnoResult = await agnoResponse.json();
              if (agnoResult.success && agnoResult.download_url) {
                // Add Agno token usage to existing tokens
                if (agnoResult.token_usage) {
                  tokens.agnoTokens = agnoResult.token_usage;
                  tokens.agnoProcessingCost = agnoResult.processingCost;
                }
                
                // Trigger automatic download of the enhanced XLSX file
                const link = document.createElement('a');
                link.href = agnoResult.download_url;
                link.download = agnoResult.file_name || `enhanced_${fileJob.name}.xlsx`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                
                console.log(`\n=== AGNO PROCESSING RESULTS ===`);
                console.log(`File: ${fileJob.name}`);
                console.log(`Enhancement: Successful - XLSX file downloaded`);
                console.log(`Download URL: ${agnoResult.download_url}`);
                console.log(`File Name: ${agnoResult.file_name}`);
                if (agnoResult.token_usage) {
                  console.log(`Agno Input Tokens: ${agnoResult.token_usage.input_tokens || 'N/A'}`);
                  console.log(`Agno Output Tokens: ${agnoResult.token_usage.output_tokens || 'N/A'}`);
                  console.log(`Agno Total Tokens: ${agnoResult.token_usage.total_tokens || 'N/A'}`);
                  console.log(`Agno Processing Cost: $${agnoResult.processingCost || 'N/A'}`);
                }
                console.log(`===============================\n`);
                
                // Note: The Python backend automatically cleans up files after 1 hour
                // No manual cleanup needed
              }
            } else {
              console.warn(`Agno processing failed for ${fileJob.name}:`, await agnoResponse.text());
            }
          } catch (agnoError) {
            console.error(`Agno processing error for ${fileJob.name}:`, agnoError);
            // Continue with original extracted data if Agno processing fails
          }
        }
        
        jobStatus = 'success';
        filesSuccessfullyProcessedCount++;
        setProcessedFilesCount(prev => prev +1);
        toast({ title: "Extraction Successful", description: `Successfully processed ${fileJob.name}.` });
      } catch (err) {
        console.error(`Error processing ${fileJob.name}:`, err);
        errorMsg = err instanceof Error ? err.message : "An unknown error occurred during extraction.";
        
        // JSON validation error handling removed
        
        if (fileJob.retryCount < maxRetriesPerFile && !isCancelled) {
          jobStatus = 'retrying';
          toast({ title: "Extraction Error (Retrying)", description: `Failed to process ${fileJob.name}: ${errorMsg}. Retry ${fileJob.retryCount + 1}/${maxRetriesPerFile}.`, variant: "default" });
          currentProcessingQueue.push({ ...fileJob, retryCount: fileJob.retryCount + 1 }); 
        } else {
          jobStatus = 'failed';
          filesFailedPermanentlyCount++;
          setFailedFilesCount(prev => prev + 1);
          toast({ title: `Extraction Failed ${isCancelled ? '(Cancelled)' : '(Max Retries)'}`, description: `Failed to process ${fileJob.name}${isCancelled ? '' : ` after ${maxRetriesPerFile + 1} attempts`}: ${errorMsg}`, variant: "destructive" });
        }
      }

      addJobResult({
        jobId: fileJob.jobId,
        fileName: fileJob.name,
        extractedData: extractedDataJson,
        thinkingProcess: finalThinkingProcess, 
        error: errorMsg,
        timestamp: Date.now(),
        promptTokens: tokens.promptTokens,
        completionTokens: tokens.completionTokens,
        totalTokens: tokens.totalTokens,
        estimatedTokens: tokens.estimatedTokens,
        tokenBreakdown: tokens.tokenBreakdown,
        agnoTokens: tokens.agnoTokens,
        agnoProcessingCost: tokens.agnoProcessingCost,
        status: isCancelled && jobStatus !== 'success' ? 'failed' : jobStatus,
      });
      
      // Log token usage for this job
      console.log('\n=== JOB TOKEN USAGE SUMMARY ===');
      console.log(`File: ${fileJob.name}`);
      console.log(`Status: ${jobStatus}`);
      console.log(`Prompt Tokens: ${tokens.promptTokens || 'N/A'}`);
      console.log(`Completion Tokens: ${tokens.completionTokens || 'N/A'}`);
      
      // Calculate and log thinking tokens
      const promptTokens = tokens.promptTokens || 0;
      const completionTokens = tokens.completionTokens || 0;
      const totalTokens = tokens.totalTokens || 0;
      const thinkingTokens = totalTokens - (promptTokens + completionTokens);
      
      if (tokens.promptTokens !== undefined && tokens.completionTokens !== undefined && tokens.totalTokens !== undefined) {
        console.log(`Thinking Tokens: ${thinkingTokens > 0 ? thinkingTokens : 0} (billed as output tokens)`);
      }
      
      console.log(`Total Tokens: ${tokens.totalTokens || 'N/A'}`);
      console.log(`Estimated Tokens: ${tokens.estimatedTokens || 'N/A'}`);
      
      if (tokens.tokenBreakdown) {
        console.log('Token Breakdown:');
        console.log(JSON.stringify(tokens.tokenBreakdown, null, 2));
      }
      console.log('==============================\n');

      setProgress(Math.round((distinctFilesAttemptedOrCompleted / filesToProcess.length) * 100));

    }

    if (isCancelled) {
        setCurrentTask("Processing cancelled by user.");
    } else {
        setCurrentTask(filesToProcess.length > 0 ? "All files processed." : "No files to process.");
        setProgress(100);
    }
    setCurrentFileProcessing(null);
    setIsProcessingQueue(false);
    setJobQueue([]);
  }, [
      clearJobResults, thinkingEnabled, thinkingDetailLevel, addJobResult, toast, useCaching,
    ]);


  const value = useMemo(() => ({
    jobQueue, jobResults, clearJobResults,
    isProcessingQueue,
    thinkingEnabled, setThinkingEnabled,
    thinkingDetailLevel, setThinkingDetailLevel,
    useAgnoProcessing, setUseAgnoProcessing,
    progress, currentTask,
    processedFilesCount, failedFilesCount, totalFilesToProcess,
    currentFileProcessing, currentThinkingStream,
    startProcessingJobQueue, cancelProcessingJobQueue,
    useCaching, setUseCaching,
    cacheStats,
    cachePricePerMillionTokens, setCachePricePerMillionTokens,
    cachePricing, updateCachePricing,
  }), [
    jobQueue, jobResults, clearJobResults,
    isProcessingQueue,
    thinkingEnabled, 
    thinkingDetailLevel,
    useAgnoProcessing,
    progress, currentTask,
    processedFilesCount, failedFilesCount, totalFilesToProcess,
    currentFileProcessing, currentThinkingStream,
    startProcessingJobQueue, cancelProcessingJobQueue,
    useCaching, setUseCaching, 
    cacheStats,
    cachePricePerMillionTokens, setCachePricePerMillionTokens,
    cachePricing, updateCachePricing,
  ]);

  return <JobContext.Provider value={value}>{children}</JobContext.Provider>;
}

export function useJob() {
  const context = useContext(JobContext);
  if (context === undefined) {
    throw new Error('useJob must be used within a JobProvider');
  }
  return context;
}
