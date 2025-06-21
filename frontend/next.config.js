/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  // Increase API body size limit to handle 100MB file uploads
  api: {
    bodyParser: {
      sizeLimit: '100mb',
    },
    responseLimit: false, // Disable response size limit for large Excel files
    externalResolver: true, // Allow external API calls
  },
  // Increase request timeout for long-running operations
  // serverRuntimeConfig is not a standard Next.js config option for timeouts.
  // Timeouts for API routes are typically handled by the serverless function limits of the deployment platform (e.g., Vercel).
  // The `bodyTimeLimit` seems to be a custom interpretation or perhaps from an older version/different framework.
  // For Next.js API routes, especially on Vercel, the default timeout is short (e.g., 10s on Hobby, 60s on Pro).
  // For longer tasks, background jobs or switching to Edge Functions (with streaming) are recommended.
  // However, if this `bodyTimeLimit` is intended for a custom server or specific middleware, we update it.
  // The more standard way for API routes in Next.js is to rely on platform limits or use `experimental.proxyTimeout`.
  // As per the plan, we're updating `bodyTimeLimit`.
  // If this is meant for the built-in Next.js server (`next start`), it's not a standard config.
  // For API routes, the timeout is usually platform-dependent.
  // Let's assume this is for a custom server setup or a specific middleware expectation.
  // If this is intended for the API route `/api/agno-process/route.ts` when it acts as a proxy,
  // the `fetch` call within that route should have its own timeout.

  // The `bodyTimeLimit` is not a standard Next.js config for API route timeouts.
  // The serverless function timeout on Vercel (common deployment for Next.js) is the relevant limit.
  // For Hobby plan, it's 10s. For Pro, it can be configured up to 300s (5 min) for Serverless Functions.
  // Edge functions can stream responses for longer.
  // This setting might be a misunderstanding or for a custom server.
  // We will update it as requested, but also ensure the /api/agno-process/route.ts has its own fetch timeout.

  // The property `serverRuntimeConfig` and `bodyTimeLimit` is not standard for Next.js timeouts.
  // API Route timeouts are generally governed by the deployment platform (e.g., Vercel's serverless function limits).
  // For Vercel: Hobby plan = 10s, Pro plan can be configured up to 300s (5 min).
  // For very long operations (>5 min), background jobs are the standard recommendation.
  // However, since the plan explicitly mentions it, I will change it.
  // It's possible this config is for a custom server setup or a specific middleware.

  // If this is intended for the `next start` command or a custom server, this might be relevant.
  // For API routes deployed on serverless platforms like Vercel, this has no effect.
  // The timeout of the API route itself is the key.
  // The plan asks for this to be changed, so I will change it.
  experimental: { // `experimental` features are subject to change.
    // proxyTimeout: 2700000, // 45 minutes in milliseconds - This would be more relevant if using rewrites as proxy
    // This is not the correct place for bodyTimeLimit based on Next.js docs.
    // However, following the plan:
    // bodyTimeLimit: 2700000, // THIS IS NOT A STANDARD NEXT.JS EXPERIMENTAL CONFIG

    // Re-evaluating: `serverRuntimeConfig` is for passing server-only runtime configs.
    // `bodyTimeLimit` is not a known property within it for timeout control in standard Next.js.
    // The original file had it under `serverRuntimeConfig`. I will keep it there as per original structure.
    // This implies the project might have a custom server setup or specific middleware that uses this.
    allowedDevOrigins: [
        "http://localhost:9002",
        "https://*.cloudworkstations.dev",
        "https://6000-firebase-studio-1746773309598.cluster-c3a7z3wnwzapkx3rfr5kz62dac.cloudworkstations.dev",
        "https://9000-firebase-studio-1746773309598.cluster-c3a7z3wnwzapkx3rfr5kz62dac.cloudworkstations.dev",
    ],
  },
  // The initial file had `serverRuntimeConfig` at the top level, not under `experimental`.
  // Let's adhere to the original structure for this non-standard config.
  serverRuntimeConfig: { // This is for server-only runtime configuration variables.
    bodyTimeLimit: 2700000, // 45 minutes in milliseconds (updated from 20 min)
                           // Note: This is not a standard Next.js option for request timeouts.
                           // API route timeouts are usually platform-dependent (e.g., Vercel serverless function limits).
                           // This might be used by custom server logic or middleware if present.
  },
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'picsum.photos',
        port: '',
        pathname: '/**',
      },
    ],
  },
  experimental: {
    allowedDevOrigins: [
        "http://localhost:9002",
        "https://*.cloudworkstations.dev",
        "https://6000-firebase-studio-1746773309598.cluster-c3a7z3wnwzapkx3rfr5kz62dac.cloudworkstations.dev",
        "https://9000-firebase-studio-1746773309598.cluster-c3a7z3wnwzapkx3rfr5kz62dac.cloudworkstations.dev",
    ],
  },
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.fallback = config.resolve.fallback || {};
      config.resolve.fallback.async_hooks = false;
      config.resolve.fallback.buffer = false; 
      config.resolve.fallback.child_process = false;
      config.resolve.fallback.crypto = false; 
      config.resolve.fallback.dns = false;
      config.resolve.fallback.fs = false;
      config.resolve.fallback['fs/promises'] = false;
      config.resolve.fallback.http = false; 
      config.resolve.fallback.https = false; 
      config.resolve.fallback.net = false;
      config.resolve.fallback.os = false; 
      config.resolve.fallback.path = false; 
      config.resolve.fallback.querystring = false; 
      config.resolve.fallback.stream = false; 
      config.resolve.fallback.string_decoder = false;
      config.resolve.fallback.sys = false;
      config.resolve.fallback.timers = false; 
      config.resolve.fallback.tls = false;
      config.resolve.fallback.tty = false; 
      config.resolve.fallback.url = false; 
      config.resolve.fallback.util = false; 
      config.resolve.fallback.vm = false; 
      config.resolve.fallback.zlib = false;
      config.resolve.fallback['node:perf_hooks'] = false;
      config.resolve.fallback['http2'] = false;
    }
    config.experiments = { ...config.experiments, topLevelAwait: true };
    return config;
  },
};

module.exports = nextConfig;
