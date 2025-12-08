import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  MessageSquare, 
  FolderOpen, 
  Settings, 
  BrainCircuit, 
  BarChart2, 
  Activity, 
  ChevronLeft, 
  ChevronRight 
} from 'lucide-react';

const Sidebar = ({ collapsed, toggle }) => {
  const navClass = ({ isActive }) =>
    `flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 ${
      isActive 
        ? 'bg-primary text-white shadow-lg shadow-primary/30' 
        : 'text-slate-400 hover:bg-slate-800 hover:text-white'
    } ${collapsed ? 'justify-center' : ''}`;

  return (
    <div className={`h-screen bg-secondary text-white flex flex-col p-4 fixed left-0 top-0 transition-all duration-300 z-50 ${collapsed ? 'w-20' : 'w-64'}`}>
      
      {/* Header / Logo */}
      <div className={`flex items-center gap-2 mb-10 px-2 ${collapsed ? 'justify-center' : ''}`}>
        <BrainCircuit className="w-8 h-8 text-accent flex-shrink-0" />
        {!collapsed && (
          <h1 className="text-xl font-bold tracking-wider whitespace-nowrap overflow-hidden transition-opacity duration-300">
            QwenRAG
          </h1>
        )}
      </div>
      
      {/* Navigation Links */}
      <nav className="flex flex-col gap-2">
        <NavLink to="/" className={navClass} title={collapsed ? "Chat" : ""}>
          <MessageSquare className="w-5 h-5 flex-shrink-0" />
          {!collapsed && <span className="whitespace-nowrap overflow-hidden">Chat</span>}
        </NavLink>
        
        <NavLink to="/files" className={navClass} title={collapsed ? "File Manager" : ""}>
          <FolderOpen className="w-5 h-5 flex-shrink-0" />
          {!collapsed && <span className="whitespace-nowrap overflow-hidden">File Manager</span>}
        </NavLink>
        
        <NavLink to="/benchmark" className={navClass} title={collapsed ? "Benchmarks" : ""}>
          <BarChart2 className="w-5 h-5 flex-shrink-0" />
          {!collapsed && <span className="whitespace-nowrap overflow-hidden">Benchmarks</span>}
        </NavLink>

        <NavLink to="/stats" className={navClass} title={collapsed ? "System Logs" : ""}>
          <Activity className="w-5 h-5 flex-shrink-0" />
          {!collapsed && <span className="whitespace-nowrap overflow-hidden">Chat History</span>}
        </NavLink>

        <NavLink to="/config" className={navClass} title={collapsed ? "Configuration" : ""}>
          <Settings className="w-5 h-5 flex-shrink-0" />
          {!collapsed && <span className="whitespace-nowrap overflow-hidden">Configuration</span>}
        </NavLink>
      </nav>

      {/* Toggle Button */}
      <div className="mt-auto flex flex-col gap-4">
        <button 
          onClick={toggle}
          className="p-2 rounded-lg bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-white transition-colors self-center"
        >
          {collapsed ? <ChevronRight className="w-5 h-5" /> : <ChevronLeft className="w-5 h-5" />}
        </button>

        {!collapsed && (
          <div className="text-xs text-slate-500 px-4 text-center whitespace-nowrap overflow-hidden">
            v1.1.0 Connected :8050
          </div>
        )}
      </div>
    </div>
  );
};

export default Sidebar;
