'use server';

/**
 * This module provides tracking for Gemini's implicit caching.
 * 
 * Note: We are not using explicit caching via the Gemini API as it's not fully
 * supported in the current SDK integration. Instead, we track content that would 
 * likely benefit from Gemini's implicit caching and monitor potential savings.
 */

import { ai } from './genkit';

/**
 * Represents a cache entry with metadata
 */
export interface CacheEntry {
  name: string;
  model: string;
  displayName?: string;
  createTime: string;
  updateTime: string;
  expireTime: string;
  inputTokens?: number;
}

/**
 * Structure to hold cached content for reuse
 */
export interface CachedContent {
  name: string;
  model: string;
  inputTokens: number;
  expireTime: string;
  systemPrompt?: string;
  content?: string;
}

/**
 * Track cache hits to calculate savings
 */
export interface CacheUsageStats {
  cacheHits: number;
  tokensSaved: number;
  estimatedCostSaved: number;
  storageCost: number; // Cost of storing the cached tokens
  netSavings: number;  // Net savings (token savings - storage cost)
}

// Cache pricing configuration - these can be updated to match actual pricing
export interface CachePricing {
  cacheStoragePricePerMillionTokensPerHour: number | Record<string, number>;  // Price per million tokens per hour for storage (can be model-specific)
  cacheCreationPricePerMillionTokens: number | Record<string, number>;        // Price per million tokens for initial caching (can be model-specific)
  inputTokenPricePerMillionTokens: number | Record<string, number>;           // Price per million tokens for input tokens (can be model-specific)
}

// Default pricing based on examples - these should be updated to match actual pricing
// This is now a private variable, not exported directly
let _cachePricing: CachePricing = {
  // Default values with model-specific overrides
  cacheStoragePricePerMillionTokensPerHour: {
    default: 1.00,
    'gemini-1.5-pro': 1.20,
    'gemini-1.5-flash': 0.80,
    'gemini-1.0-pro': 0.90
  },
  cacheCreationPricePerMillionTokens: {
    default: 0.025,
    'gemini-1.5-pro': 0.030,
    'gemini-1.5-flash': 0.020
  },
  inputTokenPricePerMillionTokens: {
    default: 0.15,
    'gemini-1.5-pro': 0.25,
    'gemini-1.5-flash': 0.10
  }
};

/**
 * Getter function to retrieve the current cache pricing configuration
 * @returns The current cache pricing configuration
 */
export async function getCachePricing(): Promise<CachePricing> {
  return { ..._cachePricing };
}

/**
 * Helper function to get the appropriate price based on model
 * @param priceMap Price map or single price
 * @param model Model name
 * @returns The appropriate price value
 */
export async function getPriceForModel(
  priceMap: number | Record<string, number>, 
  model: string
): Promise<number> {
  if (typeof priceMap === 'number') {
    return priceMap;
  }
  
  // Extract the base model name (remove version specifics if needed)
  const baseModelName = model.replace('googleai/', '').split('-').slice(0, 3).join('-');
  
  // Try exact match
  if (priceMap[model]) {
    return priceMap[model];
  }
  
  // Try base model name
  if (priceMap[baseModelName]) {
    return priceMap[baseModelName];
  }
  
  // Fall back to default
  return priceMap.default || 0;
}

// In-memory cache store for the demo
const cachedContents: Map<string, CachedContent> = new Map();

// Global cache usage statistics
const cacheStats: CacheUsageStats = {
  cacheHits: 0,
  tokensSaved: 0,
  estimatedCostSaved: 0,
  storageCost: 0,
  netSavings: 0
};

/**
 * Helper function to create a deterministic hash from content
 * @param str The string to hash
 * @returns A simple hash string
 */
function hashString(str: string): string {
  let hash = 0;
  if (str.length === 0) return hash.toString(36);
  
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  
  return Math.abs(hash).toString(36);
}

/**
 * Creates a tracking entry for content that may benefit from Gemini's implicit caching.
 * This doesn't actually create a cache in the Gemini API, but helps us track content
 * that should be benefiting from Gemini's implicit caching.
 * 
 * @param model The model name to use for caching
 * @param content The content to cache (text)
 * @param systemPrompt Optional system instruction to cache
 * @param ttlHours Time to live in hours (defaults to 1 hour)
 * @returns The cache entry information
 */
export async function createCache(
  model: string,
  content: string,
  systemPrompt?: string,
  ttlHours = 1
): Promise<CachedContent> {
  // Format the model name for GoogleAI if needed
  const formattedModel = model.startsWith('googleai/') ? model : `googleai/${model}`;
  const modelBaseName = formattedModel.replace('googleai/', '');
  
  // Check minimum token requirement based on model type (as per Gemini docs)
  // The minimum input token count for implicit caching is 1,024 for 2.5 Flash and 2,048 for 2.5 Pro
  const contentTokens = Math.ceil((content?.length || 0) / 4);
  const systemPromptTokens = Math.ceil((systemPrompt?.length || 0) / 4);
  const totalInputTokens = contentTokens + systemPromptTokens;
  
  const minimumTokens = modelBaseName.includes('pro') ? 2048 : 1024;
  
  if (totalInputTokens < minimumTokens) {
    console.log(`[Cache Service] WARNING: Content is too small for implicit caching (${totalInputTokens} tokens)`);
    console.log(`[Cache Service] Minimum tokens for effective implicit caching: ${minimumTokens} for ${modelBaseName}`);
    console.log(`[Cache Service] Implicit caching will not be effective until this threshold is met`);
  } else {
    console.log(`[Cache Service] Content size (${totalInputTokens} tokens) meets the minimum threshold for implicit caching`);
    console.log(`[Cache Service] Similar requests with this content should benefit from Gemini's implicit caching`);
  }
  
  // Generate a deterministic cache ID based on content
  const combinedContent = `${systemPrompt || ''}${content}${formattedModel}`;
  const contentHash = hashString(combinedContent);
  const cacheId = `cache_${contentHash}`;
  
  // Check if this content hash already exists
  const existingCache = await getCache(cacheId);
  if (existingCache) {
    console.log(`[Cache Service] Found existing cache with same content: ${cacheId}`);
    // Update TTL and return existing cache
    return updateCacheTTL(cacheId, ttlHours) as Promise<CachedContent>;
  }
  
  // Calculate expiration time
  const expireTime = new Date();
  expireTime.setHours(expireTime.getHours() + ttlHours);
  
  const newCache: CachedContent = {
    name: cacheId,
    model: formattedModel,
    inputTokens: totalInputTokens,
    expireTime: expireTime.toISOString(),
    systemPrompt,
    content
  };
  
  // Store in our in-memory cache
  cachedContents.set(cacheId, newCache);
  
  // Calculate storage cost
  const cachingCreationCost = (totalInputTokens / 1000000) * await getPriceForModel(_cachePricing.cacheCreationPricePerMillionTokens, model);
  const storageHourlyCost = (totalInputTokens / 1000000) * await getPriceForModel(_cachePricing.cacheStoragePricePerMillionTokensPerHour, model);
  const storageTotalCost = storageHourlyCost * ttlHours;
  
  // Update global storage cost
  cacheStats.storageCost += storageTotalCost;
  cacheStats.netSavings = cacheStats.estimatedCostSaved - cacheStats.storageCost;
  
  console.log(`[Cache Service] Created cache ${cacheId}`);
  console.log(`[Cache Service] Cached approximately ${totalInputTokens} tokens`);
  console.log(`[Cache Service] TTL: ${ttlHours} hours (expires at ${expireTime.toISOString()})`);
  console.log(`[Cache Service] Cache creation cost: $${cachingCreationCost.toFixed(6)}`);
  console.log(`[Cache Service] Storage cost: $${storageTotalCost.toFixed(6)} ($${storageHourlyCost.toFixed(6)}/hour for ${ttlHours} hours)`);
  
  return newCache;
}

/**
 * Lists all available caches
 * @returns Array of cached content entries
 */
export async function listCaches(): Promise<CachedContent[]> {
  // Clean expired caches first
  const now = new Date();
  for (const [key, cache] of cachedContents.entries()) {
    if (new Date(cache.expireTime) < now) {
      cachedContents.delete(key);
    }
  }
  
  return Array.from(cachedContents.values());
}

/**
 * Gets a specific cache by ID
 * @param cacheId The ID of the cache to retrieve
 * @returns The cached content or null if not found
 */
export async function getCache(cacheId: string): Promise<CachedContent | null> {
  // Check if cache exists and isn't expired
  const cache = cachedContents.get(cacheId);
  if (!cache) {
    return null;
  }
  
  if (new Date(cache.expireTime) < new Date()) {
    cachedContents.delete(cacheId);
    return null;
  }
  
  return cache;
}

/**
 * Updates a cache's TTL
 * @param cacheId The ID of the cache to update
 * @param ttlHours New time to live in hours
 * @returns The updated cache or null if not found
 */
export async function updateCacheTTL(cacheId: string, ttlHours: number): Promise<CachedContent | null> {
  const cache = cachedContents.get(cacheId);
  if (!cache) {
    return null;
  }
  
  // Update expiration time
  const expireTime = new Date();
  expireTime.setHours(expireTime.getHours() + ttlHours);
  cache.expireTime = expireTime.toISOString();
  
  console.log(`[Cache Service] Updated cache ${cacheId} TTL to ${ttlHours} hours`);
  console.log(`[Cache Service] New expiration: ${expireTime.toISOString()}`);
  
  return cache;
}

/**
 * Deletes a cache
 * @param cacheId The ID of the cache to delete
 * @returns True if deletion was successful
 */
export async function deleteCache(cacheId: string): Promise<boolean> {
  return cachedContents.delete(cacheId);
}

/**
 * Record a cache hit and update savings statistics
 * @param inputTokens Number of input tokens saved
 * @param pricePerMillionInputTokens Optional override price per million input tokens
 * @param model Optional model name for pricing (defaults to 'default')
 */
export async function recordCacheHit(inputTokens: number, pricePerMillionInputTokens?: number, model: string = 'default'): Promise<void> {
  cacheStats.cacheHits += 1;
  cacheStats.tokensSaved += inputTokens;
  
  // Use provided price or fall back to the configured global price
  const actualPrice = pricePerMillionInputTokens || await getPriceForModel(_cachePricing.inputTokenPricePerMillionTokens, model);
  
  // Calculate cost savings (input tokens only)
  const costSaved = (inputTokens / 1000000) * actualPrice;
  cacheStats.estimatedCostSaved += costSaved;
  
  // Recalculate net savings
  cacheStats.netSavings = cacheStats.estimatedCostSaved - cacheStats.storageCost;
  
  console.log(`\n[Cache Service] CACHE HIT RECORDED!`);
  console.log(`[Cache Service] Cache hit #${cacheStats.cacheHits}`);
  console.log(`[Cache Service] Tokens saved this request: ${inputTokens.toLocaleString()}`);
  console.log(`[Cache Service] Price per million tokens: $${actualPrice}`);
  console.log(`[Cache Service] Money saved this request: $${costSaved.toFixed(6)}`);
  console.log(`[Cache Service] Running total tokens saved: ${cacheStats.tokensSaved.toLocaleString()}`);
  console.log(`[Cache Service] Running total money saved: $${cacheStats.estimatedCostSaved.toFixed(6)}`);
  console.log(`[Cache Service] Total storage cost: $${cacheStats.storageCost.toFixed(6)}`);
  console.log(`[Cache Service] Net savings: $${cacheStats.netSavings.toFixed(6)}\n`);
}

/**
 * Get current cache usage statistics
 * @returns The current cache statistics
 */
export async function getCacheStats(): Promise<CacheUsageStats> {
  // Log current cache status
  const now = new Date();
  const activeCaches = Array.from(cachedContents.entries())
    .filter(([_, cache]) => new Date(cache.expireTime) > now)
    .length;

  console.log("\n=== CACHE SERVICE STATUS ===");
  console.log(`Active Caches: ${activeCaches}`);
  console.log(`Total Cache Hits: ${cacheStats.cacheHits}`);
  console.log(`Total Tokens Saved: ${cacheStats.tokensSaved.toLocaleString()}`);
  console.log(`Gross Cost Saved: $${cacheStats.estimatedCostSaved.toFixed(6)}`);
  console.log(`Storage Cost: $${cacheStats.storageCost.toFixed(6)}`);
  console.log(`Net Savings: $${cacheStats.netSavings.toFixed(6)}`);
  console.log("===========================\n");
  
  // Clean expired caches while we're here
  for (const [key, cache] of cachedContents.entries()) {
    if (new Date(cache.expireTime) < now) {
      console.log(`[Cache Service] Cleaning expired cache: ${key}`);
      cachedContents.delete(key);
    }
  }
  
  return { ...cacheStats };
}

/**
 * Update the cache pricing configuration
 * @param newPricing The new pricing configuration
 * @param defaultModel Optional default model name for logging (defaults to 'default')
 */
export async function updateCachePricing(newPricing: Partial<CachePricing>, defaultModel: string = 'default'): Promise<CachePricing> {
  _cachePricing = {
    ..._cachePricing,
    ...newPricing
  };
  
  console.log("[Cache Service] Updated cache pricing:");
  console.log(`[Cache Service] Storage: $${await getPriceForModel(_cachePricing.cacheStoragePricePerMillionTokensPerHour, defaultModel)} per million tokens per hour`);
  console.log(`[Cache Service] Creation: $${await getPriceForModel(_cachePricing.cacheCreationPricePerMillionTokens, defaultModel)} per million tokens`);
  console.log(`[Cache Service] Input tokens: $${await getPriceForModel(_cachePricing.inputTokenPricePerMillionTokens, defaultModel)} per million tokens`);
  
  // Recalculate net savings with new pricing
  cacheStats.netSavings = cacheStats.estimatedCostSaved - cacheStats.storageCost;
  
  return _cachePricing;
} 