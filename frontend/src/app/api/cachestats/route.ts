import { NextResponse } from 'next/server';
import * as CachingService from '@/ai/caching-service';

export async function GET() {
  try {
    // Get cache statistics
    const stats = await CachingService.getCacheStats();
    
    return NextResponse.json(stats);
  } catch (error) {
    console.error('Error fetching cache statistics:', error);
    return NextResponse.json({ error: 'Failed to fetch cache statistics' }, { status: 500 });
  }
} 