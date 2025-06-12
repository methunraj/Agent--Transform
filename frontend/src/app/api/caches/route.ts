import { NextResponse } from 'next/server';
import * as CachingService from '@/ai/caching-service';

export async function GET() {
  try {
    // Get all caches from the caching service
    const caches = await CachingService.listCaches();
    
    // Return only active caches (not expired)
    const now = new Date();
    const activeCaches = caches.filter(cache => new Date(cache.expireTime) > now);
    
    return NextResponse.json(activeCaches);
  } catch (error) {
    console.error('Error fetching caches:', error);
    return NextResponse.json({ error: 'Failed to fetch caches' }, { status: 500 });
  }
}

// Endpoint to extend the TTL of all caches
export async function POST(request: Request) {
  try {
    const { action, ttlHours = 1 } = await request.json();
    
    if (action === 'extendAll') {
      // Get all caches
      const caches = await CachingService.listCaches();
      
      // Extend TTL for each active cache
      const now = new Date();
      const activeCaches = caches.filter(cache => new Date(cache.expireTime) > now);
      
      const updatedCaches = await Promise.all(
        activeCaches.map(cache => CachingService.updateCacheTTL(cache.name, ttlHours))
      );
      
      return NextResponse.json({ 
        success: true, 
        message: `Extended TTL for ${updatedCaches.filter(Boolean).length} caches by ${ttlHours} hour(s)`,
        caches: updatedCaches.filter(Boolean)
      });
    } 
    else if (action === 'clearAll') {
      // Get all caches
      const caches = await CachingService.listCaches();
      
      // Delete each cache
      const results = await Promise.all(
        caches.map(cache => CachingService.deleteCache(cache.name))
      );
      
      const deletedCount = results.filter(Boolean).length;
      
      return NextResponse.json({ 
        success: true, 
        message: `Deleted ${deletedCount} caches`,
        deletedCount
      });
    }
    
    return NextResponse.json({ error: 'Invalid action' }, { status: 400 });
  } catch (error) {
    console.error('Error managing caches:', error);
    return NextResponse.json({ error: 'Failed to manage caches' }, { status: 500 });
  }
} 