'use client';

import { useState, useTransition } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { useToast } from '@/hooks/use-toast';
import { KeyRound, CheckCircle, XCircle, Save, Loader2, Info, Brain, DollarSign, Thermometer, TestTube } from 'lucide-react';
import { useConfiguration } from '@/contexts/ConfigurationContext';

const availableModels = {
  googleAI: [
    'gemini-2.5-flash-preview-05-20',
    'gemini-2.5-pro-preview-05-06',
    'gemini-2.0-flash',
    'gemini-2.0-flash-lite',
  ],
};

const modelsSupportingThinkingBudget = [
  'gemini-2.5-flash-preview-05-20',
  'gemini-2.5-pro-preview-05-06',
  'gemini-2.0-flash',
  'gemini-2.0-flash-lite',
];

export function LLMSettingsTab() {
  const { llmConfig, updateLLMConfig, validateLLMConnection } = useConfiguration();
  const [isValidating, setIsValidating] = useState(false);
  const [isSaving, startSavingTransition] = useTransition();
  const { toast } = useToast();

  // Local state for form fields
  const [localApiKey, setLocalApiKey] = useState(llmConfig.apiKey);
  const [localModel, setLocalModel] = useState(llmConfig.model);
  const [localTemperature, setLocalTemperature] = useState(llmConfig.temperature);
  const [localThinkingBudget, setLocalThinkingBudget] = useState(llmConfig.thinkingBudget);
  const [localInputPrice, setLocalInputPrice] = useState(llmConfig.pricePerMillionInputTokens);
  const [localOutputPrice, setLocalOutputPrice] = useState(llmConfig.pricePerMillionOutputTokens);

  const handleValidateConnection = async () => {
    setIsValidating(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate API call
      const isValid = await validateLLMConnection();
      if (isValid) {
        toast({ title: "Connection Valid", description: "LLM configuration is working correctly." });
      } else {
        toast({ title: "Connection Invalid", description: "Please check your configuration.", variant: "destructive" });
      }
    } catch (error) {
      toast({ title: "Validation Failed", description: "Could not validate connection.", variant: "destructive" });
    } finally {
      setIsValidating(false);
    }
  };

  const handleSaveConfiguration = () => {
    startSavingTransition(() => {
      updateLLMConfig({
        apiKey: localApiKey,
        model: localModel,
        temperature: localTemperature,
        thinkingBudget: localThinkingBudget,
        pricePerMillionInputTokens: localInputPrice,
        pricePerMillionOutputTokens: localOutputPrice,
      });

      toast({
        title: "LLM Configuration Saved",
        description: `Settings for ${localModel} have been updated successfully.`,
      });
    });
  };

  const showThinkingBudgetConfig = modelsSupportingThinkingBudget.includes(localModel);
  const isBusy = isValidating || isSaving;

  return (
    <div className="space-y-6">
      <Alert>
        <Info className="h-4 w-4" />
        <AlertTitle>API Key Management</AlertTitle>
        <AlertDescription>
          For Google AI (Gemini), you can provide an API key here for client-side operations. 
          For server-side extraction, the system will use your <code>GOOGLE_API_KEY</code> environment variable or Application Default Credentials.
        </AlertDescription>
      </Alert>

      <Card>
        <CardHeader>
          <CardTitle>Provider & Model Selection</CardTitle>
          <CardDescription>Choose your AI provider and model</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="provider">Provider</Label>
            <Select value={llmConfig.provider} disabled>
              <SelectTrigger id="provider">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="googleAI">Google AI (Gemini)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="model">Model</Label>
            <Select value={localModel} onValueChange={setLocalModel} disabled={isBusy}>
              <SelectTrigger id="model">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {availableModels.googleAI.map((model) => (
                  <SelectItem key={model} value={model}>
                    {model}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>API Configuration</CardTitle>
          <CardDescription>Configure your API access</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="api-key" className="flex items-center gap-2">
              <KeyRound className="h-4 w-4" />
              API Key (Optional for Server Operations)
            </Label>
            <div className="flex items-center gap-2">
              <Input
                id="api-key"
                type="password"
                placeholder="Enter your Google AI API key (optional)"
                value={localApiKey}
                onChange={(e) => setLocalApiKey(e.target.value)}
                disabled={isBusy}
              />
              {llmConfig.isValid === true && <CheckCircle className="h-5 w-5 text-green-500" />}
              {llmConfig.isValid === false && <XCircle className="h-5 w-5 text-destructive" />}
            </div>
            <p className="text-xs text-muted-foreground">
              Leave empty to use environment variables or Application Default Credentials
            </p>
          </div>

          <Button 
            onClick={handleValidateConnection} 
            variant="outline" 
            size="sm" 
            disabled={isBusy}
            className="flex items-center gap-2"
          >
            {isValidating ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <TestTube className="h-4 w-4" />
            )}
            {isValidating ? 'Testing...' : 'Test Connection'}
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Model Parameters</CardTitle>
          <CardDescription>Fine-tune model behavior</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-4">
            <Label htmlFor="temperature-slider" className="flex items-center gap-2">
              <Thermometer className="h-4 w-4" />
              Temperature: {localTemperature.toFixed(2)}
            </Label>
            <div className="flex items-center gap-4">
              <Slider
                id="temperature-slider"
                min={0.0}
                max={2.0}
                step={0.05}
                value={[localTemperature]}
                onValueChange={(value) => setLocalTemperature(value[0])}
                className="flex-grow"
                disabled={isBusy}
              />
              <Input
                type="number"
                min={0.0}
                max={2.0}
                step={0.05}
                value={localTemperature.toFixed(2)}
                onChange={(e) => {
                  const val = parseFloat(e.target.value);
                  if (!isNaN(val) && val >= 0.0 && val <= 2.0) {
                    setLocalTemperature(val);
                  }
                }}
                className="w-20"
                disabled={isBusy}
              />
            </div>
            <p className="text-xs text-muted-foreground">
              Lower values (0.2) for more deterministic output, higher (0.8) for more creative responses
            </p>
          </div>

          {showThinkingBudgetConfig && (
            <div className="space-y-4">
              <Label htmlFor="thinking-budget-slider" className="flex items-center gap-2">
                <Brain className="h-4 w-4" />
                Thinking Budget: {localThinkingBudget ?? 'Default'}
              </Label>
              <div className="flex items-center gap-4">
                <Slider
                  id="thinking-budget-slider"
                  min={0}
                  max={24576}
                  step={256}
                  value={[localThinkingBudget ?? 0]}
                  onValueChange={(value) => setLocalThinkingBudget(value[0] || undefined)}
                  className="flex-grow"
                  disabled={isBusy}
                />
                <Input
                  type="number"
                  min={0}
                  max={24576}
                  value={localThinkingBudget ?? ''}
                  onChange={(e) => {
                    const val = e.target.value;
                    if (val === '') {
                      setLocalThinkingBudget(undefined);
                    } else {
                      const numVal = parseInt(val, 10);
                      if (!isNaN(numVal) && numVal >= 0 && numVal <= 24576) {
                        setLocalThinkingBudget(numVal);
                      }
                    }
                  }}
                  className="w-28"
                  disabled={isBusy}
                />
              </div>
              <p className="text-xs text-muted-foreground">
                Controls the model's reasoning depth. Set to 0 to disable thinking, higher values allow more detailed reasoning.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Pricing Configuration</CardTitle>
          <CardDescription>Set up cost estimation for token usage</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="input-price" className="flex items-center gap-2">
                <DollarSign className="h-4 w-4" />
                Input Tokens ($ per million)
              </Label>
              <Input
                id="input-price"
                type="number"
                step="any"
                placeholder="e.g., 0.15"
                value={localInputPrice ?? ''}
                onChange={(e) => {
                  const val = e.target.value;
                  setLocalInputPrice(val === '' ? undefined : parseFloat(val));
                }}
                disabled={isBusy}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="output-price" className="flex items-center gap-2">
                <DollarSign className="h-4 w-4" />
                Output Tokens ($ per million)
              </Label>
              <Input
                id="output-price"
                type="number"
                step="any"
                placeholder="e.g., 0.60"
                value={localOutputPrice ?? ''}
                onChange={(e) => {
                  const val = e.target.value;
                  setLocalOutputPrice(val === '' ? undefined : parseFloat(val));
                }}
                disabled={isBusy}
              />
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            Configure pricing for accurate cost estimation in the dashboard
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Configuration Status</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">Status:</span>
              <Badge variant={llmConfig.isConfigured ? "default" : "secondary"}>
                {llmConfig.isConfigured ? "Configured" : "Incomplete"}
              </Badge>
              {llmConfig.isValid === true && (
                <Badge variant="secondary" className="text-green-600">
                  <CheckCircle className="h-3 w-3 mr-1" />
                  Validated
                </Badge>
              )}
            </div>
            <Button 
              onClick={handleSaveConfiguration} 
              disabled={isBusy}
              className="flex items-center gap-2"
            >
              {isSaving ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Save className="h-4 w-4" />
              )}
              Save Configuration
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}