'use client';

import { useState, useTransition, useEffect, useCallback } from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { useToast } from '@/hooks/use-toast';
import { KeyRound, CheckCircle, XCircle, Save, Loader2, Info, Brain, DollarSign, Thermometer } from 'lucide-react';
import { useLLMConfig } from '@/contexts/LLMContext';
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

const providerDisplayNames: Record<string, string> = {
  googleAI: 'Google AI (Gemini)',
};

const modelsSupportingThinkingBudget = [
  'gemini-2.5-flash-preview-05-20',
  'gemini-2.5-pro-preview-05-06',
  'gemini-2.0-flash',
  'gemini-2.0-flash-lite',
];

// Helper function (can be moved to a util file if used elsewhere)
const getInitialTemperatureGlobal = (): number => {
  if (typeof process === 'undefined' || typeof process.env === 'undefined') return 0.3;
  const envTemp = process.env.NEXT_PUBLIC_DEFAULT_LLM_TEMPERATURE;
  if (envTemp) {
    const num = parseFloat(envTemp);
    if (!isNaN(num) && num >= 0.0 && num <= 2.0) return num;
  }
  return 0.3;
};

export function LLMConfigForm() {
  const {
    provider: contextProvider,
    // setProvider: setContextProvider, // Not used directly if provider is fixed
    apiKey: contextApiKey, setApiKey: setContextApiKey,
    model: contextModel, setModel: setContextModel,
    isKeyValid: contextIsKeyValid, setIsKeyValid: setContextIsKeyValid,
    availableModels,
    // The following context values are for the *active* context model
    // numericThinkingBudget: contextNumericThinkingBudget, 
    setNumericThinkingBudget: setContextNumericThinkingBudget,
    // pricePerMillionInputTokens: contextPricePerMillionInputTokens,
    setPricePerMillionInputTokens: setContextPricePerMillionInputTokens,
    // pricePerMillionOutputTokens: contextPricePerMillionOutputTokens,
    setPricePerMillionOutputTokens: setContextPricePerMillionOutputTokens,
    // temperature: contextTemperature,
    setTemperature: setContextTemperature,
  } = useLLMConfig();

  // Local state for form fields
  const [localProvider, setLocalProvider] = useState(contextProvider); // Currently fixed to 'googleAI'
  const [localApiKey, setLocalApiKey] = useState(contextApiKey);
  const [localModel, setLocalModel] = useState(contextModel); // For the dropdown selection
  const [localIsKeyValid, setLocalIsKeyValid] = useState<boolean | null>(contextIsKeyValid);
  
  // These local states will hold settings for the model selected in `localModel` (dropdown)
  const [localNumericThinkingBudget, setLocalNumericThinkingBudget] = useState<number | undefined>(undefined);
  const [localPricePerMillionInputTokens, setLocalPricePerMillionInputTokens] = useState<number | undefined>(undefined);
  const [localPricePerMillionOutputTokens, setLocalPricePerMillionOutputTokens] = useState<number | undefined>(undefined);
  const [localTemperature, setLocalTemperature] = useState<number>(getInitialTemperatureGlobal());
  
  const [isValidating, setIsValidating] = useState(false);
  const [isSaving, startSavingTransition] = useTransition();
  const { toast } = useToast();

  // Effect 1: Initialize local form state (API key, selected model in dropdown) from global context ONCE or when context itself changes these specific values.
  useEffect(() => {
    setLocalProvider(contextProvider); // Though provider is fixed
    setLocalApiKey(contextApiKey);
    setLocalModel(contextModel); // Sets the dropdown to the globally active model
    setLocalIsKeyValid(contextIsKeyValid);
  }, [contextProvider, contextApiKey, contextModel, contextIsKeyValid]);

  // Effect 2: When user CHANGES THE MODEL IN THE DROPDOWN (localModel), load specific settings for THAT model from localStorage into form fields.
  useEffect(() => {
    if (!localModel || typeof window === 'undefined') return; // Don't run if localModel is not set yet or on server

    const providerId = 'googleAI'; // Fixed provider

    const storedInputPrice = localStorage.getItem(`intelliextract_priceInput_${providerId}_${localModel}`);
    setLocalPricePerMillionInputTokens(storedInputPrice ? (parseFloat(storedInputPrice) || undefined) : undefined);

    const storedOutputPrice = localStorage.getItem(`intelliextract_priceOutput_${providerId}_${localModel}`);
    setLocalPricePerMillionOutputTokens(storedOutputPrice ? (parseFloat(storedOutputPrice) || undefined) : undefined);
    
    const storedTemperature = localStorage.getItem(`intelliextract_temperature_${providerId}_${localModel}`);
    setLocalTemperature(storedTemperature ? (parseFloat(storedTemperature) ?? getInitialTemperatureGlobal()) : getInitialTemperatureGlobal());

    const storedNumericBudget = localStorage.getItem(`intelliextract_numericThinkingBudget_${providerId}_${localModel}`);
    const budgetFromStorage = storedNumericBudget ? parseInt(storedNumericBudget, 10) : undefined;
    setLocalNumericThinkingBudget(budgetFromStorage !== undefined && !isNaN(budgetFromStorage) ? budgetFromStorage : undefined);

  }, [localModel]); // Only re-run when the model selected in the dropdown (localModel) changes.

  const handleLocalModelChange = (newModel: string) => {
    setLocalModel(newModel); // This will trigger Effect 2 to load settings for newModel
  };

  const validateApiKey = async () => {
    if (!localApiKey.trim() && localProvider !== 'googleAI') {
      toast({ title: "API Key Required", description: "Please enter an API key to validate.", variant: "destructive" });
      return;
    }
    setIsValidating(true);
    await new Promise(resolve => setTimeout(resolve, 700)); 
    const isValid = localApiKey.length > 15 || (localProvider === 'googleAI' && localApiKey.trim() === '');
    setLocalIsKeyValid(isValid);
    setIsValidating(false);
    if (isValid) {
      toast({ title: "API Key Format Check", description: "API key format seems plausible or is not strictly required in UI for Google AI (uses .env/ADC). Actual validity tested during extraction." });
    } else {
      toast({ title: "API Key Format Check", description: "The API key seems short or invalid. Please double-check.", variant: "destructive" });
    }
  };

  const handleSaveConfiguration = () => {
    if (!localModel) {
        toast({ title: "Model Not Selected", description: "Please select a model before saving.", variant: "destructive" });
        return;
    }
    startSavingTransition(() => {
      setContextApiKey(localApiKey); 
      setContextModel(localModel); // This updates the global active model. LLMContext's useEffect will load its settings.
      setContextIsKeyValid(localIsKeyValid);
      
      // Explicitly set and save the settings for the `localModel`
      setContextPricePerMillionInputTokens(localPricePerMillionInputTokens, localModel);
      setContextPricePerMillionOutputTokens(localPricePerMillionOutputTokens, localModel);
      setContextTemperature(localTemperature, localModel);
      
      if (localProvider === 'googleAI' && modelsSupportingThinkingBudget.includes(localModel)) {
        setContextNumericThinkingBudget(localNumericThinkingBudget, localModel);
      } else {
        // If model doesn't support it, or provider changes, ensure it's undefined for this model
        setContextNumericThinkingBudget(undefined, localModel); 
      }

      toast({
        title: "LLM Configuration Updated",
        description: `Settings for ${providerDisplayNames[localProvider] || localProvider} (Model: ${localModel}) have been updated. For server operations, ensure GOOGLE_API_KEY is set in your .env file if not using Application Default Credentials.`,
      });
    });
  };

  const showThinkingBudgetConfig = localProvider === 'googleAI' && localModel && modelsSupportingThinkingBudget.includes(localModel);
  const isBusy = isValidating || isSaving;

  return (
    <div className="space-y-6 max-w-2xl mx-auto">
      <Alert>
        <Info className="h-4 w-4" />
        <AlertTitle>Important: API Key Management</AlertTitle>
        <AlertDescription>
          For Google AI (Gemini), if you provide an API key here, it will be used for client-side calls (if any) and stored locally.
          For server-side data extraction, Genkit will primarily use Application Default Credentials (ADC) or the <code>GOOGLE_API_KEY</code>
          <strong className="text-foreground"> set in your project&apos;s <code>.env</code> file</strong>.
          Entering a key here is optional if your server is configured via <code>.env</code> or ADC.
        </AlertDescription>
      </Alert>

      <div className="space-y-2">
        <Label htmlFor="llm-provider" className="text-base font-semibold">LLM Provider</Label>
        <Select value={localProvider} onValueChange={() => {}} disabled={true}> {/* Provider is fixed */}
          <SelectTrigger id="llm-provider" className="w-full rounded-md shadow-sm">
            <SelectValue placeholder="Select a provider" />
          </SelectTrigger>
          <SelectContent>
            {Object.keys(availableModels).map(provKey => (
                 <SelectItem key={provKey} value={provKey}>
                    {providerDisplayNames[provKey] || provKey}
                 </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <Label htmlFor="api-key" className="text-base font-semibold">
          API Key for {providerDisplayNames[localProvider] || localProvider} (Optional for Server)
        </Label>
        <div className="flex items-center gap-2">
          <KeyRound className="h-5 w-5 text-muted-foreground" />
          <Input
            id="api-key"
            type="password"
            placeholder={`Enter API key (optional, uses .env/ADC if empty)`}
            value={localApiKey}
            onChange={(e) => {
              setLocalApiKey(e.target.value);
              setLocalIsKeyValid(null); 
            }}
            className="flex-grow rounded-md shadow-sm"
            disabled={isBusy}
          />
          {localIsKeyValid === true && <CheckCircle className="h-5 w-5 text-green-500" />}
          {localIsKeyValid === false && <XCircle className="h-5 w-5 text-destructive" />}
        </div>
         <Button onClick={validateApiKey} variant="outline" size="sm" className="mt-2" disabled={isBusy || localProvider !== 'googleAI' && !localApiKey.trim()}>
            {isValidating ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
            {isValidating ? 'Validating...' : 'Validate Key Format'}
        </Button>
        <p className="text-xs text-muted-foreground">
            Key is stored locally. Server uses <code>.env</code> or ADC.
        </p>
      </div>

      {availableModels[localProvider] && availableModels[localProvider].length > 0 && (
        <div className="space-y-2">
          <Label htmlFor="llm-model" className="text-base font-semibold">Model</Label>
          <Select value={localModel} onValueChange={handleLocalModelChange} disabled={isBusy}>
            <SelectTrigger id="llm-model" className="w-full rounded-md shadow-sm">
              <SelectValue placeholder="Select a model" />
            </SelectTrigger>
            <SelectContent>
              {availableModels[localProvider].map((modelName) => (
                <SelectItem key={modelName} value={modelName}>
                  {modelName}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
           <p className="text-xs text-muted-foreground mt-1">
             This selection updates the active model for the UI and informs server-side flows. Model-specific settings like pricing and temperature will adapt.
           </p>
        </div>
      )}
      
      {showThinkingBudgetConfig && (
        <div className="space-y-4 p-4 border rounded-lg bg-card shadow-sm">
          <Label htmlFor="thinking-budget-slider" className="text-base font-semibold flex items-center">
            <Brain className="mr-2 h-5 w-5 text-primary" />
            Thinking Budget (for {localModel})
          </Label>
          <div className="flex items-center gap-4">
            <Slider
              id="thinking-budget-slider"
              min={0}
              max={24576}
              step={256} 
              value={[localNumericThinkingBudget ?? 0]} 
              onValueChange={(value) => setLocalNumericThinkingBudget(value[0])}
              className="flex-grow"
              disabled={isBusy}
            />
            <Input
              id="thinking-budget-input"
              type="number"
              min={0}
              max={24576}
              value={localNumericThinkingBudget ?? ''} 
              onChange={(e) => {
                const val = e.target.value;
                if (val === '') {
                    setLocalNumericThinkingBudget(undefined);
                } else {
                    const numVal = parseInt(val, 10);
                    if (!isNaN(numVal) && numVal >= 0 && numVal <= 24576) {
                        setLocalNumericThinkingBudget(numVal);
                    }
                }
              }}
              className="w-28"
              disabled={isBusy}
            />
          </div>
          <p className="text-xs text-muted-foreground">
            Range: 0-24576. Set to 0 to disable thinking. Higher values allow more detailed reasoning.
            If left blank or not set, the model uses its default behavior.
          </p>
        </div>
      )}

      <div className="space-y-4 p-4 border rounded-lg bg-card shadow-sm">
        <Label htmlFor="temperature-slider" className="text-base font-semibold flex items-center">
            <Thermometer className="mr-2 h-5 w-5 text-primary" />
            Model Temperature (for {localModel || 'selected model'})
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
              disabled={isBusy || !localModel}
            />
            <Input
              id="temperature-input"
              type="number"
              min={0.0}
              max={2.0}
              step={0.05}
              value={localTemperature.toFixed(2)}
              onChange={(e) => {
                  const val = parseFloat(e.target.value);
                  if (!isNaN(val) && val >= 0.0 && val <= 2.0) {
                      setLocalTemperature(val);
                  } else if (e.target.value === '') {
                     setLocalTemperature(getInitialTemperatureGlobal()); 
                  }
              }}
              className="w-24"
              disabled={isBusy || !localModel}
            />
        </div>
         <p className="text-xs text-muted-foreground">
            Controls randomness. Lower values (e.g., 0.2) for more deterministic output, higher (e.g., 0.8) for more creative. Range: 0.0 - 2.0.
          </p>
      </div>

      <div className="space-y-4 p-4 border rounded-lg bg-card shadow-sm">
        <Label className="text-base font-semibold flex items-center">
          <DollarSign className="mr-2 h-5 w-5 text-primary" />
          Model Pricing (for {localModel || 'selected model'})
        </Label>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="space-y-1">
            <Label htmlFor="price-input-tokens" className="text-sm">Price per Million Input Tokens ($)</Label>
            <Input
              id="price-input-tokens"
              type="number"
              step="any"
              placeholder="e.g., 0.50"
              value={localPricePerMillionInputTokens ?? ''}
              onChange={(e) => {
                const val = e.target.value;
                setLocalPricePerMillionInputTokens(val === '' ? undefined : parseFloat(val));
              }}
              className="rounded-md shadow-sm"
              disabled={isBusy || !localModel}
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="price-output-tokens" className="text-sm">Price per Million Output Tokens ($)</Label>
            <Input
              id="price-output-tokens"
              type="number"
              step="any"
              placeholder="e.g., 1.50"
              value={localPricePerMillionOutputTokens ?? ''}
              onChange={(e) => {
                const val = e.target.value;
                setLocalPricePerMillionOutputTokens(val === '' ? undefined : parseFloat(val));
              }}
              className="rounded-md shadow-sm"
              disabled={isBusy || !localModel}
            />
          </div>
        </div>
        <p className="text-xs text-muted-foreground">
          Enter the pricing for the <span className="font-medium">{localModel || 'selected model'}</span> to enable cost estimation on the Dashboard. These values are stored locally.
        </p>
      </div>
      
      <Button onClick={handleSaveConfiguration} className="w-full" disabled={isBusy || !localModel}>
        {isSaving ? (
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        ) : (
          <Save className="mr-2 h-4 w-4" />
        )}
        Save Configuration
      </Button>
    </div>
  );
}
