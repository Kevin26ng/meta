"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Search } from "lucide-react";

export default function Navbar() {
  const pathname = usePathname();

  const navLink = (path: string, label: string) => {
    const isActive = pathname === path || (path !== "/" && pathname?.startsWith(path));
    return (
      <Link 
        href={path} 
        className={`hover:text-chaos-text transition-colors ${
          isActive 
            ? "text-chaos-green border-b-2 border-chaos-green py-5 -mb-[22px]" 
            : "text-chaos-muted"
        }`}
      >
        {label}
      </Link>
    );
  };

  return (
    <nav className="sticky top-0 z-50 w-full border-b border-chaos-border bg-chaos-dark/80 backdrop-blur-md">
      <div className="flex h-16 items-center px-4 gap-6">
        <Link href="/" className="flex items-center gap-2">
          <span className="text-xl font-bold text-chaos-green tracking-tight">Chaos<span className="text-chaos-text">Lab</span></span>
        </Link>
        
        <div className="flex items-center gap-6 flex-1 ml-4 text-sm font-medium overflow-x-auto">
          {navLink("/", "Hub")}
          {navLink("/builder", "Builder")}
          {navLink("/playground", "Playground")}
          {navLink("/arena", "Arena")}
        </div>
        
        <div className="flex items-center gap-4">
          <div className="relative group hidden md:block">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-chaos-muted group-focus-within:text-chaos-green transition-colors" />
            <input 
              type="text" 
              placeholder="Search experiments..." 
              className="bg-chaos-panel border border-chaos-border rounded-md pl-9 pr-4 py-1.5 text-sm outline-none focus:border-chaos-green focus:bg-chaos-panel-hover transition-all w-64 text-chaos-text placeholder:text-chaos-muted"
            />
          </div>
          <button className="bg-chaos-green text-chaos-dark font-bold px-4 py-1.5 text-sm rounded hover:bg-chaos-green/90 transition-colors uppercase tracking-wider">
            Launch Console
          </button>
        </div>
      </div>
    </nav>
  );
}
