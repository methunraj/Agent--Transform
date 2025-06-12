import { BotMessageSquare } from 'lucide-react';
import Link from 'next/link';

export function AppLogo() {
  return (
    <Link 
      href="/dashboard" 
      className="group flex items-center gap-3 text-lg font-bold text-foreground transition-colors-smooth hover:text-primary focus-ring-modern rounded-lg p-2 -m-2"
    >
      <div className="relative">
        <BotMessageSquare className="h-8 w-8 text-primary transition-transform-smooth group-hover:scale-110 group-hover:rotate-3" />
        <div className="absolute inset-0 bg-primary/20 rounded-lg blur-lg opacity-0 group-hover:opacity-100 transition-all duration-300 -z-10" />
      </div>
      <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent font-extrabold tracking-tight">
        IntelliExtract
      </span>
    </Link>
  );
}
