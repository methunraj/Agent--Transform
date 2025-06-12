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
  serverRuntimeConfig: {
    bodyTimeLimit: 1200000, // 20 minutes in milliseconds
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
