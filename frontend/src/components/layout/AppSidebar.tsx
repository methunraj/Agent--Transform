'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  SidebarHeader,
  SidebarContent,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarFooter,
} from '@/components/ui/sidebar';
import { AppLogo } from './AppLogo';
import { Separator } from '@/components/ui/separator';
import { Button } from '@/components/ui/button';
import {
  LayoutDashboard,
  Settings,
  FolderOpen,
  PlayCircle,
  LogOut,
  BotMessageSquare,
  Sparkles
} from 'lucide-react';
import { cn } from '@/lib/utils';

const navItems = [
  { 
    href: '/dashboard', 
    icon: LayoutDashboard, 
    label: 'Dashboard',
    description: 'View analytics and metrics'
  },
  { 
    href: '/configuration', 
    icon: Settings, 
    label: 'Configuration',
    description: 'Setup LLM and AI settings'
  },
  { 
    href: '/file-management', 
    icon: FolderOpen, 
    label: 'Data Sources',
    description: 'Manage your files'
  },
  { 
    href: '/run-extraction', 
    icon: PlayCircle, 
    label: 'Run Extraction',
    description: 'Process and extract data'
  },
];

export function AppSidebar() {
  const pathname = usePathname();

  return (
    <>
      <SidebarHeader className="hidden md:flex border-b border-sidebar-border h-16 items-center justify-start px-4 bg-gradient-to-r from-sidebar-background to-sidebar-background/95">
        <AppLogo />
      </SidebarHeader>
      
      <SidebarContent className="flex-1 overflow-y-auto p-3 space-y-1">
        <div className="mb-4">
          <div className="flex items-center gap-2 px-3 py-2 text-xs font-medium text-muted-foreground uppercase tracking-wider">
            <Sparkles className="h-3 w-3" />
            <span className="group-data-[collapsible=icon]:hidden">Navigation</span>
          </div>
        </div>
        
        <SidebarMenu className="space-y-1">
          {navItems.map((item, index) => {
            const isActive = pathname === item.href;
            return (
              <SidebarMenuItem key={item.href} className="animate-slide-in-left" style={{ animationDelay: `${index * 50}ms` }}>
                <Link href={item.href} legacyBehavior passHref>
                  <SidebarMenuButton
                    asChild
                    variant="default"
                    className={cn(
                      'group relative justify-start w-full h-12 px-3 rounded-xl transition-all-smooth hover-lift',
                      'border border-transparent',
                      isActive 
                        ? 'bg-gradient-to-r from-primary/10 to-accent/10 text-primary border-primary/20 shadow-lg shadow-primary/10' 
                        : 'hover:bg-sidebar-accent/50 hover:text-sidebar-accent-foreground hover:border-sidebar-border'
                    )}
                    tooltip={{
                      children: (
                        <div className="text-left">
                          <div className="font-medium">{item.label}</div>
                          <div className="text-xs text-muted-foreground">{item.description}</div>
                        </div>
                      ), 
                      className: "bg-card text-card-foreground border-border shadow-lg"
                    }}
                  >
                    <a className="flex items-center gap-3 w-full">
                      <div className={cn(
                        'relative p-1.5 rounded-lg transition-all-smooth',
                        isActive 
                          ? 'bg-primary/20 text-primary' 
                          : 'text-muted-foreground group-hover:text-foreground group-hover:bg-accent/20'
                      )}>
                        <item.icon className="h-4 w-4" />
                        {isActive && (
                          <div className="absolute inset-0 bg-primary/20 rounded-lg blur-sm -z-10" />
                        )}
                      </div>
                      <div className="group-data-[collapsible=icon]:hidden flex-1">
                        <div className={cn(
                          'font-medium text-sm transition-colors-smooth',
                          isActive ? 'text-primary' : 'text-foreground'
                        )}>
                          {item.label}
                        </div>
                        <div className="text-xs text-muted-foreground group-hover:text-muted-foreground/80 transition-colors-smooth">
                          {item.description}
                        </div>
                      </div>
                      {isActive && (
                        <div className="w-1 h-6 bg-gradient-to-b from-primary to-accent rounded-full" />
                      )}
                    </a>
                  </SidebarMenuButton>
                </Link>
              </SidebarMenuItem>
            );
          })}
        </SidebarMenu>
      </SidebarContent>
      
      <SidebarFooter className="p-4 border-t border-sidebar-border bg-gradient-to-r from-sidebar-background to-sidebar-background/95">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <div className="w-2 h-2 bg-success rounded-full animate-pulse-subtle" />
          <span className="group-data-[collapsible=icon]:hidden">System Ready</span>
        </div>
      </SidebarFooter>
    </>
  );
}