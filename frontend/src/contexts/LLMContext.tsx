'use client';
import type { Dispatch, SetStateAction } from 'react';
import { createContext, useContext, useState, useMemo, useEffect, useCallback } from 'react';

// Helper to get initial numeric thinking budget from environment variable or a default
const getInitialNumericThinkingBudgetForModel = (modelId?: string): number | undefined => {
  if (typeof window !== 'undefined' && modelId) {
    const storedBudget = localStorage.getItem(`intelliextract_numericThinkingBudget_googleAI_${modelId}`);
    if (storedBudget) {
      const num = parseInt(storedBudget, 10);
      if (!isNaN(num) && num >= 0 && num <= 24576) return num;
    }
  }
  // Fallback to global env/default if nothing specific for model
  if (typeof process === 'undefined' || typeof process.env === 'undefined') return undefined;
  const envBudget = process.env.NEXT_PUBLIC_DEFAULT_LLM_NUMERIC_THINKING_BUDGET;
  if (envBudget) {
    const num = parseInt(envBudget, 10);
    if (!isNaN(num) && num >= 0 && num <= 24576) return num;
  }
  return undefined;
};

// Helper to get initial temperature from environment variable or a default
const getInitialTemperatureForModel = (modelId?: string): number => {
   if (typeof window !== 'undefined' && modelId) {
    const storedTemp = localStorage.getItem(`intelliextract_temperature_googleAI_${modelId}`);
    if (storedTemp) {
      const num = parseFloat(storedTemp);
      if (!isNaN(num) && num >= 0.0 && num <= 2.0) return num;
    }
  }
  // Fallback to global env/default
  if (typeof process === 'undefined' || typeof process.env === 'undefined') return 0.3; // Default if no env var
  const envTemp = process.env.NEXT_PUBLIC_DEFAULT_LLM_TEMPERATURE;
  if (envTemp) {
    const num = parseFloat(envTemp);
    if (!isNaN(num) && num >= 0.0 && num <= 2.0) return num;
  }
  return 0.3; // Final fallback default
};


interface LLMContextType {
  provider: string;
  setProvider: (newProvider: string) => void;
  apiKey: string;
  setApiKey: (newApiKey: string) => void;
  model: string;
  setModel: (newModel: string) => void;
  isKeyValid: boolean | null;
  setIsKeyValid: Dispatch<SetStateAction<boolean | null>>;
  availableModels: Record<string, string[]>;
  numericThinkingBudget?: number;
  setNumericThinkingBudget: (newBudget: number | undefined, forWhichModel: string) => void;
  pricePerMillionInputTokens?: number;
  setPricePerMillionInputTokens: (newPrice: number | undefined, forWhichModel: string) => void;
  pricePerMillionOutputTokens?: number;
  setPricePerMillionOutputTokens: (newPrice: number | undefined, forWhichModel: string) => void;
  temperature: number;
  setTemperature: (newTemperature: number, forWhichModel: string) => void;
}

const defaultAvailableModels: Record<string, string[]> = {
  googleAI: [
    'gemini-2.5-flash-preview-05-20',
    'gemini-2.5-pro-preview-05-06',
    'gemini-2.0-flash',
    'gemini-2.0-flash-lite',
  ],
};

const LLMContext = createContext<LLMContextType | undefined>(undefined);

export function LLMProvider({ children }: { children: React.ReactNode }) {
  const [_provider, _setInternalProvider] = useState('googleAI'); 
  const [apiKey, _setInternalApiKey] = useState('');
  const [model, _setInternalModel] = useState('');
  const [isKeyValid, setIsKeyValid] = useState<boolean | null>(null);
  const [availableModels, setAvailableModels] = useState<Record<string, string[]>>(defaultAvailableModels);
  const [modelDetails, setModelDetails] = useState<any[]>([]); // To store full model info from backend
  const [isLoadingModels, setIsLoadingModels] = useState<boolean>(true);
  const [modelError, setModelError] = useState<string | null>(null);
  
  const [numericThinkingBudget, _setInternalNumericThinkingBudget] = useState<number | undefined>(undefined);
  const [pricePerMillionInputTokens, _setInternalPricePerMillionInputTokens] = useState<number | undefined>(undefined);
  const [pricePerMillionOutputTokens, _setInternalPricePerMillionOutputTokens] = useState<number | undefined>(undefined);
  const [temperature, _setInternalTemperature] = useState<number>(0.3);

  // Fetch available models from the backend
  useEffect(() => {
    const fetchModels = async () => {
      setIsLoadingModels(true);
      setModelError(null);
      try {
        const response = await fetch('/api/ai-models'); // Calls our Next.js API proxy
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || `Failed to fetch models: ${response.status}`);
        }
        const data = await response.json();

        // Assuming data is an array of ModelInfo objects from backend schema
        // We need to transform it into Record<string, string[]> for existing availableModels structure
        // and also store the full details.
        const googleModels = data.filter((m: any) => m.provider === 'Google').map((m: any) => m.id);

        if (googleModels.length > 0) {
          setAvailableModels({ googleAI: googleModels });
          setModelDetails(data); // Store full details

          // Initialize model if not already set or if current model is not in new list
          const currentStoredModel = localStorage.getItem('intelliextract_model_googleAI');
          if (!currentStoredModel || !googleModels.includes(currentStoredModel)) {
            const newInitialModel = googleModels[0];
            _setInternalModel(newInitialModel);
            localStorage.setItem('intelliextract_model_googleAI', newInitialModel);
          } else {
            _setInternalModel(currentStoredModel); // Keep existing valid model
          }
        } else {
          // Fallback to default hardcoded if API returns empty or no Google models
          setAvailableModels(defaultAvailableModels);
          setModelDetails([]); // Or transform defaultAvailableModels to ModelInfo structure
           const newInitialModel = defaultAvailableModels['googleAI']?.[0] || '';
          _setInternalModel(newInitialModel);
          localStorage.setItem('intelliextract_model_googleAI', newInitialModel);
        }

      } catch (error) {
        console.error("Error fetching models:", error);
        setModelError(error instanceof Error ? error.message : String(error));
        setAvailableModels(defaultAvailableModels); // Fallback on error
        setModelDetails([]);
        const fallbackModel = defaultAvailableModels['googleAI']?.[0] || '';
        _setInternalModel(fallbackModel); // Set to a fallback model
        localStorage.setItem('intelliextract_model_googleAI', fallbackModel);
      } finally {
        setIsLoadingModels(false);
      }
    };

    fetchModels();
  }, []);


  // Effect for initializing the provider and API key from localStorage ONCE on mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      _setInternalApiKey(localStorage.getItem('intelliextract_apiKey_googleAI') || '');
      // Model initialization is now handled by the fetchModels effect
    }
  }, []);

  // Effect for loading model-specific settings (prices, temp, budget) when 'model' state changes
  useEffect(() => {
    if (typeof window !== 'undefined' && model) {
      // const providerId = 'googleAI'; // Assuming fixed provider for now
      // const modelInfo = modelDetails.find(m => m.id === model && m.provider.toLowerCase() === providerId.toLowerCase());

      // Prices are not directly available from model object; use stored or default.
      // Pricing URL is available in modelInfo.pricing_details_url

      // const storedInputPrice = localStorage.getItem(`intelliextract_priceInput_${providerId}_${model}`);
      // _setInternalPricePerMillionInputTokens(storedInputPrice ? (parseFloat(storedInputPrice) || undefined) : undefined);
      // const storedOutputPrice = localStorage.getItem(`intelliextract_priceOutput_${providerId}_${model}`);
      // _setInternalPricePerMillionOutputTokens(storedOutputPrice ? (parseFloat(storedOutputPrice) || undefined) : undefined);
      // For now, pricing is not set dynamically per model from API, so this part can be simplified or removed
      // if pricing is managed elsewhere or displayed via the pricing_details_url.
      
      _setInternalTemperature(getInitialTemperatureForModel(model));
      _setInternalNumericThinkingBudget(getInitialNumericThinkingBudgetForModel(model));
    }
  }, [model, modelDetails]);


  const setProvider = useCallback((newProvider: string) => {
    if (newProvider === 'googleAI') {
      _setInternalProvider(newProvider);
      const defaultModelForNewProvider = availableModels[newProvider]?.[0] || defaultAvailableModels[newProvider]?.[0] || '';
      _setInternalModel(defaultModelForNewProvider);
      localStorage.setItem('intelliextract_model_googleAI', defaultModelForNewProvider);
    }
  }, [availableModels]);

  const setApiKey = useCallback((newApiKey: string) => {
    _setInternalApiKey(newApiKey);
    if (typeof window !== 'undefined') {
      localStorage.setItem(`intelliextract_apiKey_googleAI`, newApiKey);
    }
  }, []);

  const setModel = useCallback((newModel: string) => {
    // Check against the dynamically fetched models for 'googleAI' provider
    if (availableModels['googleAI']?.includes(newModel)) {
      _setInternalModel(newModel);
      if (typeof window !== 'undefined') {
        localStorage.setItem(`intelliextract_model_googleAI`, newModel);
      }
    } else {
      console.warn(`Attempted to set model "${newModel}" which is not in the available list for GoogleAI.`);
    }
  }, [availableModels]);

  const setNumericThinkingBudget = useCallback((newBudget: number | undefined, forWhichModel: string) => {
    if (typeof window !== 'undefined') {
      const key = `intelliextract_numericThinkingBudget_googleAI_${forWhichModel}`;
      if (newBudget !== undefined) {
        localStorage.setItem(key, String(newBudget));
      } else {
        localStorage.removeItem(key);
      }
    }
    // If the budget being set is for the currently active model, update context state
    if (forWhichModel === model) {
      _setInternalNumericThinkingBudget(newBudget);
    }
  }, [model]); 

  const setPricePerMillionInputTokens = useCallback((newPrice: number | undefined, forWhichModel: string) => {
    if (typeof window !== 'undefined') {
      const key = `intelliextract_priceInput_googleAI_${forWhichModel}`;
      if (newPrice !== undefined) {
        localStorage.setItem(key, String(newPrice));
      } else {
        localStorage.removeItem(key);
      }
    }
    if (forWhichModel === model) {
      _setInternalPricePerMillionInputTokens(newPrice);
    }
  }, [model]);

  const setPricePerMillionOutputTokens = useCallback((newPrice: number | undefined, forWhichModel: string) => {
     if (typeof window !== 'undefined') {
      const key = `intelliextract_priceOutput_googleAI_${forWhichModel}`;
      if (newPrice !== undefined) {
        localStorage.setItem(key, String(newPrice));
      } else {
        localStorage.removeItem(key);
      }
    }
    if (forWhichModel === model) {
      _setInternalPricePerMillionOutputTokens(newPrice);
    }
  }, [model]);

  const setTemperature = useCallback((newTemperature: number, forWhichModel: string) => {
    if (typeof window !== 'undefined') {
      const key = `intelliextract_temperature_googleAI_${forWhichModel}`;
      localStorage.setItem(key, String(newTemperature));
    }
    if (forWhichModel === model) {
      _setInternalTemperature(newTemperature);
    }
  }, [model]);

  const value = useMemo(() => ({
    provider: _provider, setProvider,
    apiKey, setApiKey,
    model, setModel,
    isKeyValid, setIsKeyValid,
    availableModels: defaultAvailableModels,
    numericThinkingBudget, setNumericThinkingBudget,
    pricePerMillionInputTokens, setPricePerMillionInputTokens,
    pricePerMillionOutputTokens, setPricePerMillionOutputTokens,
    temperature, setTemperature,
  }), [
    _provider, setProvider,
    apiKey, setApiKey,
    model, setModel,
    isKeyValid, // setIsKeyValid is from useState, stable
    numericThinkingBudget, setNumericThinkingBudget,
    pricePerMillionInputTokens, setPricePerMillionInputTokens,
    pricePerMillionOutputTokens, setPricePerMillionOutputTokens,
    temperature, setTemperature,
  ]);

  return <LLMContext.Provider value={value}>{children}</LLMContext.Provider>;
}

export function useLLMConfig() {
  const context = useContext(LLMContext);
  if (context === undefined) {
    throw new Error('useLLMConfig must be used within an LLMProvider');
  }
  return context;
}
