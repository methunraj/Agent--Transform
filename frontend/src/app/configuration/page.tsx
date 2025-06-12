'use client';

import { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Settings, BrainCircuit, CheckCircle } from 'lucide-react';
import { LLMSettingsTab } from '@/components/configuration/LLMSettingsTab';
import { AIGenerationTab } from '@/components/configuration/AIGenerationTab';
import { useConfiguration } from '@/contexts/ConfigurationContext';
import { Badge } from '@/components/ui/badge';

export default function ConfigurationPage() {
  const [activeTab, setActiveTab] = useState<'llm' | 'generation'>('llm');
  const { llmConfig, isConfigurationComplete } = useConfiguration();

  return (
    <div className="flex-1 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Configuration</h1>
          <p className="text-muted-foreground">
            Set up your AI extraction configuration in two simple steps
          </p>
        </div>
        {isConfigurationComplete && (
          <Badge variant="secondary" className="flex items-center gap-2">
            <CheckCircle className="h-4 w-4" />
            Configuration Complete
          </Badge>
        )}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Setup Process</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4 text-sm">
            <div className="flex items-center gap-2">
              <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                llmConfig.isConfigured ? 'bg-green-500 text-white' : 'bg-muted text-muted-foreground'
              }`}>
                1
              </div>
              <span className={llmConfig.isConfigured ? 'text-green-600 font-medium' : 'text-muted-foreground'}>
                Configure LLM Settings
              </span>
            </div>
            <div className="flex-1 h-px bg-border" />
            <div className="flex items-center gap-2">
              <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                llmConfig.isConfigured ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'
              }`}>
                2
              </div>
              <span className={llmConfig.isConfigured ? 'text-foreground' : 'text-muted-foreground'}>
                Generate AI Configuration
              </span>
            </div>
          </div>
        </CardContent>
      </Card>

      <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as 'llm' | 'generation')}>
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="llm" className="flex items-center gap-2">
            <Settings className="h-4 w-4" />
            LLM Settings
            {llmConfig.isConfigured && <CheckCircle className="h-3 w-3 text-green-500" />}
          </TabsTrigger>
          <TabsTrigger 
            value="generation" 
            disabled={!llmConfig.isConfigured}
            className="flex items-center gap-2"
          >
            <BrainCircuit className="h-4 w-4" />
            AI Generation & Setup
            {!llmConfig.isConfigured && (
              <span className="text-xs text-muted-foreground">(Configure LLM first)</span>
            )}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="llm" className="mt-6">
          <LLMSettingsTab />
        </TabsContent>

        <TabsContent value="generation" className="mt-6">
          {llmConfig.isConfigured ? (
            <AIGenerationTab />
          ) : (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Settings className="h-12 w-12 text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold text-muted-foreground mb-2">
                  Configure LLM Settings First
                </h3>
                <p className="text-center text-sm text-muted-foreground mb-4">
                  You need to set up your LLM configuration before you can generate AI configurations.
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}