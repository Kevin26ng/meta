"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Search, X } from "lucide-react";
import { useState, useEffect, useRef, useCallback } from "react";

export default function Navbar() {
  const pathname = usePathname();
  const router = useRouter();
  const searchParams = useSearchParams();
  const inputRef = useRef<HTMLInputElement>(null);

  // Sync with URL search param when on Hub page
  const [query, setQuery] = useState("");

  useEffect(() => {
    if (pathname === "/") {
      setQuery(searchParams.get("search") || "");
    }
  }, [pathname, searchParams]);

  const handleSearch = useCallback((value: string) => {
    setQuery(value);
    if (pathname === "/") {
      // On Hub page: update URL params in real-time
      const params = new URLSearchParams(searchParams.toString());
      if (value) {
        params.set("search", value);
      } else {
        params.delete("search");
      }
      router.replace(`/?${params.toString()}`, { scroll: false });
    }
  }, [pathname, router, searchParams]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && query.trim()) {
      // If not on Hub, navigate there with search param
      if (pathname !== "/") {
        router.push(`/?search=${encodeURIComponent(query.trim())}`);
      }
    }
    if (e.key === "Escape") {
      handleSearch("");
      inputRef.current?.blur();
    }
  };

  const handleClear = () => {
    handleSearch("");
    inputRef.current?.focus();
  };

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
              ref={inputRef}
              id="navbar-search"
              type="text" 
              placeholder="Search experiments..." 
              value={query}
              onChange={(e) => handleSearch(e.target.value)}
              onKeyDown={handleKeyDown}
              className="bg-chaos-panel border border-chaos-border rounded-md pl-9 pr-8 py-1.5 text-sm outline-none focus:border-chaos-green/50 focus:bg-chaos-panel-hover focus:shadow-[0_0_12px_rgba(57,255,20,0.1)] transition-all w-64 text-chaos-text placeholder:text-chaos-muted"
            />
            {query && (
              <button
                onClick={handleClear}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-chaos-muted hover:text-chaos-text transition-colors"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
