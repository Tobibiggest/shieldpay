import {
 
  Search,
  Bell,
  CreditCard,

  Menu,

} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import SidebarContent from './SidebarContent';


const Header = ({ user, onSignIn }) => {
    return (
        <header className="sticky top-0 z-10 border-b border-[var(--border)] bg-[var(--background)]/80 backdrop-blur-md">
          <div className="flex items-center justify-between px-6 py-4">
            {/* Mobile menu */}
            <div className="flex items-center gap-3">
              <Sheet>
                <SheetTrigger asChild>
                  <Button variant="ghost" size="icon" className="md:hidden text-[var(--muted)] hover:text-[var(--foreground)]">
                    <Menu className="h-4 w-4" />
                  </Button>
                </SheetTrigger>
                <SheetContent side="left" className="w-64 p-0 border-r border-[var(--border)]" style={{background:'var(--background)'}}>
                  <SidebarContent />
                </SheetContent>
              </Sheet>
              <span className="font-mono text-xs tracking-widest uppercase opacity-60 md:hidden">
                Shieldpay <span className="opacity-40">®</span>
              </span>
            </div>

            {/* Right: search + user */}
            <div className="flex items-center gap-4">
              <div className="relative hidden sm:block">
                <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 opacity-40" />
                <input
                  type="text"
                  placeholder="Search..."
                  className="pl-9 pr-4 py-2 w-52 bg-white/5 border border-[var(--border)] font-mono text-[11px] text-[var(--foreground)] placeholder:opacity-40 focus:outline-none focus:border-[var(--accent)] transition-colors"
                />
              </div>
              <button className="relative opacity-50 hover:opacity-100 transition-opacity">
                <Bell className="h-4 w-4" />
              </button>
              <div className="flex items-center gap-2">
                <Avatar className="h-7 w-7">
                  <AvatarImage src={user?.photoURL} alt="User" />
                  <AvatarFallback className="text-[10px] font-mono bg-[var(--accent)] text-white">{user?.displayName?.charAt(0)}</AvatarFallback>
                </Avatar>
                <span className="font-mono text-[10px] uppercase tracking-widest opacity-60 hidden sm:block">{user?.displayName?.split(' ')[0]}</span>
              </div>
            </div>
          </div>
        </header>
    );
};

export default Header;
