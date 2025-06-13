export default function LLMConfigurationPage() {
  return (
    <div className="flex-1 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold tracking-tight">LLM Configuration</h1>
      </div>
      <div className="max-w-2xl mx-auto text-center py-12">
        <h2 className="text-2xl font-semibold mb-4">Page Moved</h2>
        <p className="text-muted-foreground mb-6">
          LLM Configuration has been moved to the unified Configuration page for a better user experience.
        </p>
        <a 
          href="/configuration" 
          className="inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none ring-offset-background bg-primary text-primary-foreground hover:bg-primary/90 h-10 py-2 px-4"
        >
          Go to Configuration
        </a>
      </div>
    </div>
  );
}