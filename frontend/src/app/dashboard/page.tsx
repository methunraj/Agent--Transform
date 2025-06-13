'use client';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Activity, Sigma, BarChart3, CheckCircle, AlertCircle, DollarSign, AlertTriangle, FileClock } from "lucide-react";
import { useJob } from "@/contexts/JobContext";
import { useConfiguration } from "@/contexts/ConfigurationContext";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import type { JobResult } from "@/types";
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";

export default function DashboardPage() {
  const {
    isProcessingQueue,
    jobResults,
    totalFilesToProcess,
    processedFilesCount,
    failedFilesCount,
    jobQueue,
    currentTask: jobContextCurrentTask, // Renamed to avoid conflict
    useCaching, // Check if caching is enabled
    cachePricePerMillionTokens, // Get the current price per million tokens
    cacheStats, // Get cache stats directly
    cachePricing, // Get cache pricing configuration
  } = useJob();
  const { llmConfig } = useConfiguration();
  const { pricePerMillionInputTokens, pricePerMillionOutputTokens } = llmConfig;
  const { toast } = useToast();

  const successfulExtractionsCount = jobResults.filter(job => job.status === 'success').length;
  const successPercentage = totalFilesToProcess > 0 ? Math.round((successfulExtractionsCount / totalFilesToProcess) * 100) : 0;

  // Calculate token sums from all job results
  const totalPromptTokens = jobResults.reduce((sum, job) => sum + (job.promptTokens || 0), 0);
  const totalCompletionTokens = jobResults.reduce((sum, job) => sum + (job.completionTokens || 0), 0);
  const totalEstimatedTokens = jobResults.reduce((sum, job) => sum + (job.estimatedTokens || 0), 0);
  
  // Calculate thinking tokens (total - (input + output))
  const calculateThinkingTokens = (job: JobResult): number => {
    if (job.totalTokens === undefined || 
        job.promptTokens === undefined || 
        job.completionTokens === undefined) {
      return 0;
    }
    
    const calculatedSum = job.promptTokens + job.completionTokens;
    return job.totalTokens > calculatedSum ? job.totalTokens - calculatedSum : 0;
  };
  
  const totalThinkingTokens = jobResults.reduce((sum, job) => sum + calculateThinkingTokens(job), 0);

  // Calculate token breakdown totals
  const tokenBreakdownTotals = jobResults.reduce((totals, job) => {
    if (!job.tokenBreakdown) return totals;
    
    return {
      documentTokens: (totals.documentTokens || 0) + (job.tokenBreakdown.documentTokens || 0),
      schemaTokens: (totals.schemaTokens || 0) + (job.tokenBreakdown.schemaTokens || 0),
      systemPromptTokens: (totals.systemPromptTokens || 0) + (job.tokenBreakdown.systemPromptTokens || 0),
      examplesTokens: (totals.examplesTokens || 0) + (job.tokenBreakdown.examplesTokens || 0),
      mediaTokens: (totals.mediaTokens || 0) + (job.tokenBreakdown.mediaTokens || 0),
    };
  }, { documentTokens: 0, schemaTokens: 0, systemPromptTokens: 0, examplesTokens: 0, mediaTokens: 0 });

  // For display purposes, calculate grand total based on available data
  // This gives a single number to display as "Total Tokens"
  const displayableGrandTotalTokens = jobResults.reduce((sum, job) => {
    if (job.totalTokens !== undefined) {
      // If we have the total tokens directly from the API, use that
      return sum + job.totalTokens;
    } else if (job.promptTokens !== undefined && job.completionTokens !== undefined) {
      // Otherwise, sum the parts
      return sum + job.promptTokens + job.completionTokens;
    }
    return sum + 0;
  }, 0);

  // Calculate cost with cache savings included
  let estimatedCost: number | null = null;
  let costDisplay: string;
  let costCalculationMessage = "Based on current settings & usage";
  
  // Get cache-related values with defaults
  const cachingEnabled = useCaching || false;
  const cacheHits = cacheStats?.cacheHits || 0;
  const tokensSaved = cacheStats?.tokensSaved || 0;
  const cacheStorageCost = cacheStats?.storageCost || 0;
  const cacheNetSavings = cacheStats?.netSavings || 0;

  // Add state for active caches and their expiration times
  const [activeCaches, setActiveCaches] = useState<Array<{name: string, expireTime: string, inputTokens: number}>>([]);
  const [timeRemaining, setTimeRemaining] = useState<Record<string, string>>({});

  // Calculate time remaining for each cache
  useEffect(() => {
    // Only run if caching is enabled
    if (!useCaching) return;
    
    // Fetch active caches
    const fetchCaches = async () => {
      try {
        // This would typically be replaced by an actual API call to list caches
        const caches = await fetch('/api/caches').then(res => res.json()).catch(() => []);
        setActiveCaches(caches || []);
      } catch (error) {
        console.error('Failed to fetch cache info:', error);
        setActiveCaches([]);
      }
    };
    
    fetchCaches();
    
    // Set up timer to update the countdown
    const timer = setInterval(() => {
      const now = new Date();
      const newTimeRemaining: Record<string, string> = {};
      
      activeCaches.forEach(cache => {
        const expireTime = new Date(cache.expireTime);
        const diffMs = expireTime.getTime() - now.getTime();
        
        if (diffMs <= 0) {
          newTimeRemaining[cache.name] = 'Expired';
        } else {
          const diffMins = Math.floor(diffMs / 60000);
          const diffSecs = Math.floor((diffMs % 60000) / 1000);
          newTimeRemaining[cache.name] = `${diffMins}m ${diffSecs}s`;
        }
      });
      
      setTimeRemaining(newTimeRemaining);
    }, 1000);
    
    return () => clearInterval(timer);
  }, [useCaching, activeCaches]);

  // Add function to extend cache TTL
  const extendCacheTTL = async (hours: number = 1) => {
    try {
      const response = await fetch('/api/caches', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ action: 'extendAll', ttlHours: hours }),
      });
      
      const result = await response.json();
      
      if (result.success) {
        toast({
          title: "TTL Extended",
          description: result.message,
        });
        
        // Refresh cache list
        const caches = await fetch('/api/caches').then(res => res.json()).catch(() => []);
        setActiveCaches(caches || []);
      } else {
        toast({
          title: "Error",
          description: result.error || "Failed to extend TTL",
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error('Failed to extend TTL:', error);
      toast({
        title: "Error",
        description: "Failed to extend TTL. See console for details.",
        variant: "destructive",
      });
    }
  };
  
  // Add function to clear all caches
  const clearAllCaches = async () => {
    try {
      const response = await fetch('/api/caches', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ action: 'clearAll' }),
      });
      
      const result = await response.json();
      
      if (result.success) {
        toast({
          title: "Caches Cleared",
          description: result.message,
        });
        
        // Clear the local list
        setActiveCaches([]);
      } else {
        toast({
          title: "Error",
          description: result.error || "Failed to clear caches",
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error('Failed to clear caches:', error);
      toast({
        title: "Error",
        description: "Failed to clear caches. See console for details.",
        variant: "destructive",
      });
    }
  };

  if (totalPromptTokens === 0 && totalCompletionTokens === 0 && jobResults.every(job => !job.totalTokens || job.totalTokens === 0)) {
    // No tokens used at all across all jobs
    costDisplay = "$0.0000";
  } else if (pricePerMillionInputTokens !== undefined && pricePerMillionOutputTokens !== undefined) {
    // If we have prices, calculate cost using available prompt and completion tokens
    const rawCost = (totalPromptTokens / 1000000 * pricePerMillionInputTokens) +
                   (totalCompletionTokens / 1000000 * pricePerMillionOutputTokens) +
                   (totalThinkingTokens / 1000000 * pricePerMillionOutputTokens); // Thinking tokens use output token pricing
    
    // Include cache benefits if caching is enabled
    if (cachingEnabled && cacheNetSavings !== 0) {
      estimatedCost = rawCost - cacheNetSavings; // Subtract net savings (which accounts for storage costs)
      costDisplay = `$${estimatedCost.toFixed(4)}`;
      costCalculationMessage = "Includes cache savings & storage costs";
    } else {
      estimatedCost = rawCost;
      costDisplay = `$${estimatedCost.toFixed(4)}`;
    }

    // Check if breakdown was missing for any job that did report some total tokens
    const someJobsMissingBreakdown = jobResults.some(job =>
        (job.totalTokens && job.totalTokens > 0) &&
        (job.promptTokens === undefined || job.completionTokens === undefined)
    );

    if (someJobsMissingBreakdown && (totalPromptTokens === 0 && totalCompletionTokens === 0 && displayableGrandTotalTokens > 0)) {
         costCalculationMessage = "Input/output token breakdown missing for all jobs with tokens. Cost cannot be accurately calculated unless input/output prices are identical and total tokens are available.";
         if (pricePerMillionInputTokens === pricePerMillionOutputTokens && displayableGrandTotalTokens > 0) {
            estimatedCost = (displayableGrandTotalTokens / 1000000 * pricePerMillionInputTokens);
            costDisplay = `$${estimatedCost.toFixed(4)}`;
            costCalculationMessage = "Est. using total tokens (input/output prices are identical; breakdown unavailable)";
         } else if (displayableGrandTotalTokens > 0) {
            costDisplay = "N/A";
         }
    } else if (someJobsMissingBreakdown) {
        costCalculationMessage = "Input/output token breakdown missing for some jobs; cost is based on available data.";
    }

  } else {
    costDisplay = "Not configured";
    costCalculationMessage = "Configure pricing in LLM settings";
  }

  // For displaying token counts:
  let displayPromptTokensStr = totalPromptTokens > 0 ? totalPromptTokens.toLocaleString() : "0";
  let displayCompletionTokensStr = totalCompletionTokens > 0 ? totalCompletionTokens.toLocaleString() : "0";
  
  const hasJobsWithTokensButNoBreakdown = jobResults.some(job => (job.totalTokens && job.totalTokens > 0) && (job.promptTokens === undefined || job.completionTokens === undefined));

  if (totalPromptTokens === 0 && hasJobsWithTokensButNoBreakdown) {
    displayPromptTokensStr = "N/A*";
  }
  if (totalCompletionTokens === 0 && hasJobsWithTokensButNoBreakdown) {
    displayCompletionTokensStr = "N/A*";
  }

  if (displayPromptTokensStr.includes("N/A*") || displayCompletionTokensStr.includes("N/A*")) {
    costCalculationMessage = costCalculationMessage.includes("(*Breakdown unavailable for some files)") ? costCalculationMessage : costCalculationMessage + " (*Breakdown unavailable for some files)";
  }


  const pendingInQueueCount = jobQueue.length;
  const currentUiTask = jobContextCurrentTask; // Use the renamed variable

  return (
    <div className="flex-1 space-y-8">
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-foreground to-muted-foreground bg-clip-text text-transparent">
            Dashboard
          </h1>
          <p className="text-muted-foreground">
            Monitor your data extraction performance and metrics
          </p>
        </div>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <div className="w-2 h-2 bg-success rounded-full animate-pulse-subtle" />
          <span>Live Updates</span>
        </div>
      </div>
      
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        <Card className="hover-lift overflow-hidden border-border/40 animate-fade-in" style={{animationDelay: '100ms'}}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Files Processed</CardTitle>
            <div className="rounded-full bg-primary/10 p-1.5">
              <BarChart3 className="h-4 w-4 text-primary" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold tracking-tight">{processedFilesCount} <span className="text-muted-foreground text-lg font-normal">/ {totalFilesToProcess}</span></div>
            <div className="mt-2 flex items-center text-sm text-muted-foreground">
              <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
                <div 
                  className="h-full bg-primary rounded-full transition-all duration-1000 ease-in-out" 
                  style={{ width: `${totalFilesToProcess > 0 ? (processedFilesCount / totalFilesToProcess) * 100 : 0}%` }}
                />
              </div>
            </div>
            <p className="mt-2 text-xs text-muted-foreground">
              Successfully processed files
            </p>
          </CardContent>
        </Card>
        
        <Card className="hover-lift overflow-hidden border-border/40 animate-fade-in" style={{animationDelay: '150ms'}}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
            <div className="rounded-full bg-success/10 p-1.5">
              <CheckCircle className="h-4 w-4 text-success" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="flex items-end gap-1">
              <div className="text-3xl font-bold tracking-tight">{successPercentage}%</div>
              <div className="text-sm text-muted-foreground mb-1">
                {successPercentage >= 90 ? '(Excellent)' : successPercentage >= 70 ? '(Good)' : successPercentage >= 50 ? '(Average)' : '(Needs attention)'}
              </div>
            </div>
            <div className="mt-2 flex items-center text-sm text-muted-foreground">
              <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
                <div 
                  className={`h-full rounded-full transition-all duration-1000 ease-in-out ${successPercentage >= 90 ? 'bg-success' : successPercentage >= 70 ? 'bg-success/80' : successPercentage >= 50 ? 'bg-warning' : 'bg-destructive'}`}
                  style={{ width: `${successPercentage}%` }}
                />
              </div>
            </div>
            <p className="mt-2 text-xs text-muted-foreground">
              Based on {totalFilesToProcess} initial files
            </p>
          </CardContent>
        </Card>
        
        <Card className="hover-lift overflow-hidden border-border/40 animate-fade-in" style={{animationDelay: '200ms'}}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Files Failed</CardTitle>
            <div className="rounded-full bg-destructive/10 p-1.5">
              <AlertTriangle className="h-4 w-4 text-destructive" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold tracking-tight">{failedFilesCount}</div>
            <div className="mt-2 flex items-center text-sm text-muted-foreground">
              <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
                <div 
                  className="h-full bg-destructive rounded-full transition-all duration-1000 ease-in-out" 
                  style={{ width: `${totalFilesToProcess > 0 ? (failedFilesCount / totalFilesToProcess) * 100 : 0}%` }}
                />
              </div>
            </div>
            <p className="mt-2 text-xs text-muted-foreground">
              After all retry attempts
            </p>
          </CardContent>
        </Card>
        
        <Card className="hover-lift overflow-hidden border-border/40 animate-fade-in" style={{animationDelay: '250ms'}}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active LLM Tasks</CardTitle>
            <div className="rounded-full bg-info/10 p-1.5">
              <Activity className="h-4 w-4 text-info" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold tracking-tight">
              {isProcessingQueue && currentUiTask && currentUiTask !== "All files processed." && currentUiTask !== "Processing cancelled by user." ? '1+' : '0'}
            </div>
            <div className="mt-2 flex items-center gap-2 text-sm text-muted-foreground">
              {isProcessingQueue && currentUiTask && currentUiTask !== "All files processed." && currentUiTask !== "Processing cancelled by user." ? (
                <>
                  <div className="w-2 h-2 bg-info rounded-full animate-pulse-subtle" />
                  <span>Processing queue... ({pendingInQueueCount} pending)</span>
                </>
              ) : (
                <>
                  <div className="w-2 h-2 bg-muted-foreground rounded-full" />
                  <span>Idle</span>
                </>
              )}
            </div>
            <p className="mt-2 text-xs text-muted-foreground">
              {isProcessingQueue ? 'LLM is actively processing data' : 'Ready to process'}
            </p>
          </CardContent>
        </Card>
      </div>
      
      <div className="grid gap-6 md:grid-cols-2">
        <Card className="hover-lift overflow-hidden border-border/40 animate-fade-in" style={{animationDelay: '300ms'}}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
              <div>
                <CardTitle className="text-lg font-semibold">Token Usage</CardTitle>
                <CardDescription className="text-sm text-muted-foreground">Detailed breakdown of token consumption</CardDescription>
              </div>
              <div className="rounded-full bg-accent/10 p-2">
                <Sigma className="h-5 w-5 text-accent" />
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-3xl font-bold tracking-tight">{displayableGrandTotalTokens.toLocaleString()}</div>
                  <p className="text-sm text-muted-foreground">Total tokens consumed</p>
                </div>
                <div className="text-right space-y-1">
                  <div className="flex items-center gap-2 text-sm">
                    <div className="w-2 h-2 bg-primary rounded-full" />
                    <span className="text-muted-foreground">Input:</span>
                    <span className="font-medium">{displayPromptTokensStr}</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <div className="w-2 h-2 bg-accent rounded-full" />
                    <span className="text-muted-foreground">Output:</span>
                    <span className="font-medium">{displayCompletionTokensStr}</span>
                  </div>
                </div>
              </div>
              {totalThinkingTokens > 0 && (
                <div className="flex items-center gap-2 text-sm">
                  <div className="w-2 h-2 bg-warning rounded-full" />
                  <span className="text-muted-foreground">Thinking:</span>
                  <span className="font-medium">{totalThinkingTokens.toLocaleString()}</span>
                </div>
              )}
              {totalThinkingTokens > 0 && (
                <div className="mt-1 pt-1 border-t border-muted">
                  <p className="text-xs font-semibold text-muted-foreground">
                    Combined I/O only: {(totalPromptTokens + totalCompletionTokens).toLocaleString()}
                  </p>
                  <p className="text-xs font-semibold text-muted-foreground">
                    Grand Total with Thinking: {displayableGrandTotalTokens.toLocaleString()}
                  </p>
                </div>
              )}
              {totalEstimatedTokens > 0 && (
                <div className="mt-2">
                  <p className="text-xs font-semibold text-muted-foreground">
                    Estimated: {totalEstimatedTokens.toLocaleString()}
                  </p>
                  {Object.entries(tokenBreakdownTotals).map(([key, value]) => 
                    value > 0 ? (
                      <p key={key} className="text-xs text-muted-foreground ml-2">
                        {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase()).replace('Tokens', '')}: {value.toLocaleString()}
                      </p>
                    ) : null
                  )}
                </div>
              )}
              {cachingEnabled && cacheHits > 0 && (
                <div className="mt-2 pt-1 border-t border-muted">
                  <p className="text-xs font-semibold text-muted-foreground">
                    Cache Performance
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Cache Hits: {cacheHits}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Tokens Saved: {tokensSaved.toLocaleString()}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
          <Card className="hover-lift overflow-hidden border-border/40 animate-fade-in" style={{animationDelay: '400ms'}}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
              <div>
                <CardTitle className="text-lg font-semibold">Est. Cost</CardTitle>
                <CardDescription className="text-sm text-muted-foreground">Real-time cost estimation</CardDescription>
              </div>
              <div className="rounded-full bg-success/10 p-2">
                <DollarSign className="h-5 w-5 text-success" />
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-3xl font-bold tracking-tight text-success">{costDisplay}</div>
                  <p className="text-sm text-muted-foreground">{costCalculationMessage}</p>
                </div>
              </div>
              
              {cachingEnabled && cacheHits > 0 && (
                <div className="mt-4 pt-4 border-t border-border/40">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="text-sm font-medium text-foreground">Cache Savings</h4>
                    <Badge variant="outline" className="text-xs border-success/20 text-success">
                      Active
                    </Badge>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Token savings:</span>
                      <span className="font-medium text-success">${(((tokensSaved || 0) / 1000000) * (isNaN(cachePricePerMillionTokens) ? 0.15 : cachePricePerMillionTokens)).toFixed(6)}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Storage cost:</span>
                      <span className="font-medium">${(cacheStorageCost || 0).toFixed(6)}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm pt-2 border-t border-border/20">
                      <span className="text-muted-foreground font-medium">Net savings:</span>
                      <span className="font-semibold text-success">${(cacheNetSavings || 0).toFixed(6)}</span>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
      </div>

      <Card className="hover-lift overflow-hidden border-border/40 animate-fade-in" style={{animationDelay: '500ms'}}>
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-xl font-semibold">Recent Extraction Job Log</CardTitle>
              <CardDescription className="text-sm text-muted-foreground mt-1">
                {jobResults.length > 0
                  ? `Showing the latest ${Math.min(10, jobResults.length)} of ${jobResults.length} job events (includes retries).`
                  : "An overview of your recent data extraction activities will appear here once you run jobs."}
              </CardDescription>
            </div>
            {jobResults.length > 0 && (
              <Badge variant="secondary" className="text-xs">
                {jobResults.length} total
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent className={jobResults.length === 0 ? "flex items-center justify-center h-64" : ""}>
          {jobResults.length === 0 ? (
            <div className="text-center space-y-3">
              <div className="w-16 h-16 mx-auto rounded-full bg-muted/30 flex items-center justify-center">
                <FileClock className="h-8 w-8 text-muted-foreground" />
              </div>
              <p className="text-muted-foreground font-medium">No job history yet</p>
              <p className="text-sm text-muted-foreground">Run an extraction to see results here</p>
            </div>
          ) : (
            <ul className="space-y-3">
              {jobResults.slice(-10).reverse().map((job) => (
                <li key={job.jobId} className="group flex items-center justify-between p-4 rounded-lg bg-card/50 border border-border/40 hover:bg-accent/50 hover:border-border transition-all duration-200 hover-lift">
                  <div className="flex items-center gap-4">
                    <div className="flex-shrink-0">
                      {job.status === 'failed' ? (
                        <div className="rounded-full bg-destructive/10 p-2">
                          <AlertTriangle className="h-4 w-4 text-destructive" />
                        </div>
                      ) : job.status === 'retrying' ? (
                        <div className="rounded-full bg-warning/10 p-2">
                          <FileClock className="h-4 w-4 text-warning animate-pulse" />
                        </div>
                      ) : job.error ? (
                        <div className="rounded-full bg-destructive/10 p-2">
                          <AlertCircle className="h-4 w-4 text-destructive" />
                        </div>
                      ) : (
                        <div className="rounded-full bg-success/10 p-2">
                          <CheckCircle className="h-4 w-4 text-success" />
                        </div>
                      )}
                    </div>
                    <div className="flex flex-col space-y-1">
                        <span className="font-medium text-foreground group-hover:text-accent-foreground transition-colors">{job.fileName}</span>
                        <div className="flex flex-wrap gap-2 mt-1">
                          {job.totalTokens !== undefined && (
                            <Badge variant="secondary" className="text-xs px-2 py-0.5">
                              {job.totalTokens.toLocaleString()} tokens
                            </Badge>
                          )}
                          {(job.promptTokens !== undefined && job.completionTokens !== undefined && (job.promptTokens > 0 || job.completionTokens > 0) && (job.promptTokens + job.completionTokens !== job.totalTokens && job.totalTokens !== undefined && (job.promptTokens + job.completionTokens) > 0 )) && (
                            <Badge variant="outline" className="text-xs px-2 py-0.5">
                              I/O: {job.promptTokens}/{job.completionTokens}
                            </Badge>
                          )}
                          {job.promptTokens !== undefined && job.completionTokens !== undefined && job.totalTokens !== undefined && 
                            job.totalTokens > (job.promptTokens + job.completionTokens) && (
                              <Badge variant="outline" className="text-xs px-2 py-0.5 border-warning/20 text-warning">
                                Thinking: {(job.totalTokens - (job.promptTokens + job.completionTokens)).toLocaleString()}
                              </Badge>
                          )}
                          {job.estimatedTokens !== undefined && job.estimatedTokens > 0 && (
                            <Badge variant="outline" className="text-xs px-2 py-0.5 border-info/20 text-info">
                              Est: {job.estimatedTokens.toLocaleString()}
                            </Badge>
                          )}
                        </div>
                        {job.tokenBreakdown && Object.values(job.tokenBreakdown).some(value => value && value > 0) && (
                            <button 
                                onClick={(e) => {
                                    e.stopPropagation();
                                    const breakdownMsg = Object.entries(job.tokenBreakdown || {})
                                        .filter(([_, value]) => value && value > 0)
                                        .map(([key, value]) => `${key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase()).replace('Tokens', '')}: ${value}`)
                                        .join('\n');
                                        
                                    // Add thinking tokens info if available
                                    const thinkingTokens = job.promptTokens !== undefined && 
                                                          job.completionTokens !== undefined && 
                                                          job.totalTokens !== undefined && 
                                                          job.totalTokens > (job.promptTokens + job.completionTokens) 
                                            ? job.totalTokens - (job.promptTokens + job.completionTokens)
                                            : null;
                                            
                                    const fullBreakdown = thinkingTokens 
                                        ? `${breakdownMsg}\nThinking: ${thinkingTokens}\n\nNote: Thinking tokens are billed at the same rate as output tokens.`
                                        : breakdownMsg;
                                        
                                    toast({ 
                                        title: `Token Breakdown for ${job.fileName}`, 
                                        description: fullBreakdown 
                                    });
                                }}
                                className="text-xs text-primary hover:text-primary/80 hover:underline transition-colors font-medium mt-1"
                            >
                                View detailed breakdown →
                            </button>
                        )}
                        {job.error && (
                          <div className="mt-2 p-2 rounded-md bg-destructive/10 border border-destructive/20">
                            <span className="text-xs text-destructive font-medium" title={job.error}>
                              Error: {job.error.length > 80 ? `${job.error.substring(0,80)}...` : job.error}
                            </span>
                          </div>
                        )}
                    </div>
                  </div>
                  <div className="flex flex-col items-end gap-2">
                    <Badge 
                      variant={
                        job.status === 'failed' ? "destructive" :
                        job.status === 'retrying' ? "outline" :
                        "default"
                      }
                      className={
                        job.status === 'retrying' ? "border-warning/20 text-warning animate-pulse" :
                        job.status === 'failed' ? "" :
                        "border-success/20 text-success"
                      }
                    >
                      {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                    </Badge>
                    {job.totalTokens !== undefined && (
                      <span className="text-xs text-muted-foreground font-mono">
                        {new Date(job.timestamp || Date.now()).toLocaleTimeString()}
                      </span>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      {cachingEnabled && (
        <Card className="hover-lift overflow-hidden border-border/40 animate-fade-in" style={{animationDelay: '600ms'}}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
            <div>
              <CardTitle className="text-xl font-semibold">Caching Metrics</CardTitle>
              <CardDescription className="text-sm text-muted-foreground mt-1">
                Real-time cache performance and management
              </CardDescription>
            </div>
            <div className="rounded-full bg-info/10 p-2">
              <FileClock className="h-5 w-5 text-info" />
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-semibold text-foreground">Performance</h4>
                  <Badge variant="secondary" className="text-xs">
                    Active
                  </Badge>
                </div>
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 rounded-lg bg-accent/20 border border-border/40">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-success rounded-full" />
                      <span className="text-sm text-muted-foreground">Cache Hits:</span>
                    </div>
                    <Badge variant="outline" className="border-success/20 text-success font-medium">
                      {cacheHits}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between p-3 rounded-lg bg-accent/20 border border-border/40">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-primary rounded-full" />
                      <span className="text-sm text-muted-foreground">Tokens Saved:</span>
                    </div>
                    <Badge variant="outline" className="border-primary/20 text-primary font-medium">
                      {tokensSaved.toLocaleString()}
                    </Badge>
                  </div>
                </div>
                
                {/* Add TTL Information */}
                {activeCaches.length > 0 && (
                  <div className="mt-6 space-y-4">
                    <div className="flex items-center justify-between">
                      <h4 className="text-sm font-semibold text-foreground">Cache TTL</h4>
                      <Badge variant="outline" className="text-xs">
                        {activeCaches.length} active
                      </Badge>
                    </div>
                    <div className="space-y-2 max-h-40 overflow-y-auto">
                      {activeCaches.map(cache => (
                        <div key={cache.name} className="flex items-center justify-between p-2 rounded-md bg-card/50 border border-border/40">
                          <span className="text-xs text-muted-foreground truncate max-w-[140px] font-mono" title={cache.name}>
                            {cache.name.substring(0, 14)}...
                          </span>
                          <Badge variant={
                            timeRemaining[cache.name] === 'Expired' ? 'destructive' :
                            timeRemaining[cache.name]?.startsWith('0m') ? 'outline' : 'secondary'
                          }
                          className={
                            timeRemaining[cache.name] === 'Expired' ? '' :
                            timeRemaining[cache.name]?.startsWith('0m') ? 'border-warning/20 text-warning' : 'border-success/20 text-success'
                          }>
                            {timeRemaining[cache.name] || 'Loading...'}
                          </Badge>
                        </div>
                      ))}
                    </div>
                    <div className="p-2 rounded-md bg-info/10 border border-info/20">
                      <p className="text-xs text-info font-medium">
                        Storage: ${typeof cachePricing?.cacheStoragePricePerMillionTokensPerHour === 'number' 
                            ? cachePricing.cacheStoragePricePerMillionTokensPerHour.toFixed(2) 
                            : cachePricing?.cacheStoragePricePerMillionTokensPerHour?.default?.toFixed(2) || "1.00"}/M tokens/hour
                      </p>
                    </div>
                  </div>
                )}
              </div>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-semibold text-foreground">Cost Analysis</h4>
                  <Badge variant="secondary" className="text-xs">
                    Real-time
                  </Badge>
                </div>
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 rounded-lg bg-accent/20 border border-border/40">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-success rounded-full" />
                      <span className="text-sm text-muted-foreground">Token Savings:</span>
                    </div>
                    <Badge variant="outline" className="border-success/20 text-success font-medium">
                      ${(((tokensSaved || 0) / 1000000) * (isNaN(cachePricePerMillionTokens) ? 0.15 : cachePricePerMillionTokens)).toFixed(6)}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between p-3 rounded-lg bg-accent/20 border border-border/40">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-warning rounded-full" />
                      <span className="text-sm text-muted-foreground">Storage Cost:</span>
                    </div>
                    <Badge variant="outline" className="border-warning/20 text-warning font-medium">
                      ${(cacheStorageCost || 0).toFixed(6)}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between p-3 rounded-lg bg-gradient-to-r from-success/10 to-success/5 border border-success/20">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-success rounded-full animate-pulse" />
                      <span className="text-sm font-medium text-foreground">Net Savings:</span>
                    </div>
                    <Badge variant={(cacheNetSavings || 0) > 0 ? "default" : "destructive"} 
                           className={(cacheNetSavings || 0) > 0 ? "border-success/20 text-success font-semibold" : ""}>
                      ${(cacheNetSavings || 0).toFixed(6)}
                    </Badge>
                  </div>
                </div>
                
                {/* Add Cache Refresh Controls */}
                <div className="mt-6 space-y-4">
                  <div className="flex items-center justify-between">
                    <h4 className="text-sm font-semibold text-foreground">TTL Management</h4>
                    <Badge variant="outline" className="text-xs">
                      Quick Actions
                    </Badge>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <Button 
                      size="sm" 
                      variant="outline" 
                      className="hover:bg-success/10 hover:border-success/30 hover:text-success transition-all duration-200 group"
                      onClick={() => extendCacheTTL(1)}
                    >
                      Extend All (1hr)
                    </Button>
                    <Button 
                      size="sm" 
                      variant="destructive" 
                      className="hover:bg-destructive/90 transition-all duration-200 group"
                      onClick={() => clearAllCaches()}
                    >
                      Clear All
                    </Button>
                  </div>
                  <div className="mt-4 p-3 rounded-lg bg-muted/30 border border-border/40">
                    <div className="flex items-start gap-2">
                      <Info className="w-3 h-3 text-muted-foreground mt-0.5 flex-shrink-0" />
                      <div className="space-y-1">
                        <p className="text-xs text-muted-foreground leading-relaxed">
                          <span className="font-medium">Extend TTL:</span> Recommended for high-frequency extraction jobs
                        </p>
                        <p className="text-xs text-muted-foreground leading-relaxed">
                          <span className="font-medium">Clear All:</span> Use when done to stop storage charges
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
