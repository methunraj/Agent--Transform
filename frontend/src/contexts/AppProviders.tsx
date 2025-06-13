'use client';

import { ConfigurationProvider } from './ConfigurationContext';
import { FileProvider } from './FileContext';
import { JobProvider } from './JobContext';

export function AppProviders({ children }: { children: React.ReactNode }) {
  return (
    <ConfigurationProvider>
      <FileProvider>
        <JobProvider>
          {children}
        </JobProvider>
      </FileProvider>
    </ConfigurationProvider>
  );
}