import React from 'react';
import { Link } from '@tanstack/react-router';
import { cn } from "@/lib/utils";
import { Button} from "@/components/ui/button";
import { Input } from "@/components/ui/input";

import { 
  Home, 
  Send, 
  History, 
  FileText, 
  Users, 
  Settings, 
  HelpCircle as Help, 
  CreditCard, 
  Search, 
  Building2,
  LogOut 
} from 'lucide-react';

export default function SidebarContent() {
  const navItems = [
    { icon: Home, label: "Dashboard", path: "/dashboard" },
    { icon: Send, label: "Send Money", path: "/send-money" },
    { icon: History, label: "Transactions", path: "/transactions" },
    { icon: Building2, label: "Connect Bank", path: "/connect-bank" },
    { icon: FileText, label: "Statements", path: "/statements" },
    { icon: Users, label: "Beneficiaries", path: "/beneficiaries" },
    { icon: Settings, label: "Settings", path: "/settings" },
    { icon: Help, label: "Help & Support", path: "/help-support" },
  ];

  return (
    <div className="flex flex-col h-full">
      {/* Logo */}
      <div className="px-6 py-5 border-b border-[var(--border)]">
        <div className="font-mono text-xs tracking-widest uppercase opacity-70">
          Shieldpay <span className="opacity-40">®</span> / AI.FRAUD
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-4">
        {navItems.map((item) => (
          <Link to={item.path} key={item.label}>
            <div className="flex items-center gap-3 px-6 py-3 font-mono text-[11px] uppercase tracking-widest text-[var(--muted)] hover:text-[var(--foreground)] hover:bg-white/[0.03] transition-colors border-l-2 border-transparent hover:border-[var(--accent)]">
              <item.icon className="h-3.5 w-3.5 shrink-0" />
              {item.label}
            </div>
          </Link>
        ))}
      </nav>

      {/* Logout */}
      <div className="px-6 py-5 border-t border-[var(--border)]">
        <button className="flex items-center gap-3 font-mono text-[11px] uppercase tracking-widest text-[var(--muted)] hover:text-[var(--foreground)] transition-colors w-full">
          <LogOut className="h-3.5 w-3.5" />
          Logout
        </button>
      </div>
    </div>
  );
}
