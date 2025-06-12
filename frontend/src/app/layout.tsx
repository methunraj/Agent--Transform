import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';
import { SidebarProvider, Sidebar, SidebarInset, SidebarTrigger, SidebarHeader, SidebarFooter } from '@/components/ui/sidebar';
import { AppSidebar } from '@/components/layout/AppSidebar';
import { Toaster } from "@/components/ui/toaster";
import { AppLogo } from '@/components/layout/AppLogo';
import { Button } from '@/components/ui/button';
import { PanelLeft } from 'lucide-react';
import { AppProviders } from '@/contexts/AppProviders';


const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: 'IntelliExtract UI',
  description: 'Extract structured data from documents using LLMs',
  icons: {
    icon: '/favicon.svg',
    shortcut: '/favicon.svg',
    apple: '/favicon.svg',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}>
        <AppProviders>
          <SidebarProvider defaultOpen>
            <Sidebar 
              variant="sidebar" 
              collapsible="icon" 
              className="border-r border-sidebar-border bg-sidebar-background backdrop-blur-xl"
            >
              <AppSidebar />
            </Sidebar>
            <SidebarInset className="flex flex-col min-h-screen">
              <header className="sticky top-0 z-50 flex h-16 items-center justify-between gap-4 border-b border-border bg-background/80 backdrop-blur-xl px-4 sm:static sm:h-auto sm:border-0 sm:bg-transparent sm:px-8 sm:py-6">
                <div className="flex items-center gap-3">
                  <SidebarTrigger className="md:hidden hover-scale focus-ring-modern p-2 rounded-lg transition-all-smooth">
                    <PanelLeft className="h-5 w-5" />
                    <span className="sr-only">Toggle Menu</span>
                  </SidebarTrigger>
                  <div className="md:hidden">
                    <AppLogo />
                  </div>
                </div>
                <div className="hidden md:flex items-center gap-4">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <div className="w-2 h-2 bg-success rounded-full animate-pulse-subtle" />
                    <span>Connected</span>
                  </div>
                </div>
              </header>
              <main className="flex-1 p-6 sm:px-8 sm:py-8 animate-fade-in">
                <div className="mx-auto max-w-7xl">
                  {children}
                </div>
              </main>
            </SidebarInset>
          </SidebarProvider>
          <Toaster />
        </AppProviders>
      </body>
    </html>
  );
}
