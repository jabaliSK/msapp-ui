import React from 'react';
import { 
  LayoutDashboard, 
  Files, 
  Database, 
  Settings, 
  LogOut, 
  Search,
  Server,
  Globe
} from 'lucide-react';
import { ViewState } from '../types';

interface LayoutProps {
  children: React.ReactNode;
  currentView: ViewState;
  setView: (view: ViewState) => void;
  onLogout: () => void;
}

export const Layout: React.FC<LayoutProps> = ({ children, currentView, setView, onLogout }) => {
  return (
    <div className="flex h-screen bg-msheireb-limestone overflow-hidden font-sans">
      {/* Sidebar */}
      <aside className="w-64 bg-white border-r border-msheireb-sand shadow-lg flex flex-col z-20">
        <div className="p-8 flex items-center gap-3">
          <div className="w-8 h-8 bg-msheireb-gold rounded-sm rotate-45 flex items-center justify-center shadow-md">
            <div className="w-4 h-4 bg-white -rotate-45"></div>
          </div>
          <div>
            <h1 className="text-xl font-serif font-bold text-msheireb-charcoal tracking-wide">MSHEIREB</h1>
            <p className="text-[10px] uppercase tracking-[0.2em] text-msheireb-bronze">Smart CMS</p>
          </div>
        </div>

        <nav className="flex-1 px-4 py-6 space-y-2">
          <NavItem 
            icon={<LayoutDashboard size={20} />} 
            label="Dashboard" 
            active={currentView === 'dashboard'} 
            onClick={() => setView('dashboard')}
          />
          <NavItem 
            icon={<Files size={20} />} 
            label="Documents" 
            active={currentView === 'documents'} 
            onClick={() => setView('documents')}
          />
          <NavItem 
            icon={<Globe size={20} />} 
            label="Web Scraping" 
            active={currentView === 'scraping'} 
            onClick={() => setView('scraping')}
          />
          <div className="pt-4 pb-2 px-4 text-xs font-bold text-gray-400 uppercase tracking-wider">
            Infrastructure
          </div>
          <div className="flex items-center px-4 py-3 text-msheireb-charcoal/70 gap-3 hover:bg-gray-50 rounded-lg cursor-not-allowed opacity-60">
            <Database size={20} />
            <span className="font-medium">Couchbase</span>
            <div className="w-2 h-2 rounded-full bg-green-500 ml-auto animate-pulse"></div>
          </div>
          <div className="flex items-center px-4 py-3 text-msheireb-charcoal/70 gap-3 hover:bg-gray-50 rounded-lg cursor-not-allowed opacity-60">
            <Server size={20} />
            <span className="font-medium">Milvus DB</span>
            <div className="w-2 h-2 rounded-full bg-green-500 ml-auto animate-pulse"></div>
          </div>
        </nav>

        <div className="p-4 border-t border-msheireb-sand">
          <button 
            onClick={onLogout}
            className="flex items-center gap-3 w-full px-4 py-3 text-msheireb-charcoal/70 hover:bg-red-50 hover:text-red-600 rounded-lg transition-colors"
          >
            <LogOut size={20} />
            <span className="font-medium">Sign Out</span>
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0 bg-msheireb-limestone relative">
        {/* Decorative background pattern */}
        <div className="absolute inset-0 opacity-5 pointer-events-none bg-doha-pattern mix-blend-multiply"></div>
        
        {/* Header */}
        <header className="h-20 bg-white/80 backdrop-blur-md border-b border-msheireb-sand flex items-center justify-between px-8 sticky top-0 z-10">
          <h2 className="text-2xl font-serif text-msheireb-charcoal">
            {currentView.charAt(0).toUpperCase() + currentView.slice(1)}
          </h2>
          
          <div className="flex items-center gap-6">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={18} />
              <input 
                type="text" 
                placeholder="Search repository..." 
                className="pl-10 pr-4 py-2 bg-gray-100 border-none rounded-full text-sm focus:outline-none focus:ring-2 focus:ring-msheireb-gold/50 w-64 transition-all"
              />
            </div>
            <div className="w-10 h-10 rounded-full bg-msheireb-charcoal text-white flex items-center justify-center font-serif text-lg border-2 border-msheireb-gold shadow-md">
              A
            </div>
          </div>
        </header>

        {/* Content Body */}
        <div className="flex-1 overflow-auto p-8 relative z-0">
          {children}
        </div>
      </main>
    </div>
  );
};

const NavItem = ({ icon, label, active, onClick }: { icon: any, label: string, active: boolean, onClick: () => void }) => (
  <button 
    onClick={onClick}
    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-300 ${
      active 
        ? 'bg-msheireb-charcoal text-white shadow-lg translate-x-1' 
        : 'text-msheireb-charcoal/70 hover:bg-msheireb-sand/30 hover:text-msheireb-charcoal'
    }`}
  >
    {React.cloneElement(icon, { className: active ? 'text-msheireb-gold' : '' })}
    <span className="font-medium">{label}</span>
  </button>
);
