import React from 'react';
import { NavLink } from 'react-router-dom';
// Add 'BarChart2' to imports
import { MessageSquare, FolderOpen, Settings, BrainCircuit, BarChart2 } from 'lucide-react';

const Sidebar = () => {
  const navClass = ({ isActive }) =>
    `flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 ${
      isActive 
        ? 'bg-primary text-white shadow-lg shadow-primary/30' 
        : 'text-slate-400 hover:bg-slate-800 hover:text-white'
    }`;

  return (
    <div className="w-64 h-screen bg-secondary text-white flex flex-col p-4 fixed left-0 top-0">
      <div className="flex items-center gap-2 mb-10 px-2">
        <BrainCircuit className="w-8 h-8 text-accent" />
        <h1 className="text-xl font-bold tracking-wider">QwenRAG</h1>
      </div>
      
      <nav className="flex flex-col gap-2">
        <NavLink to="/" className={navClass}>
          <MessageSquare className="w-5 h-5" />
          <span>Chat</span>
        </NavLink>
        <NavLink to="/files" className={navClass}>
          <FolderOpen className="w-5 h-5" />
          <span>File Manager</span>
        </NavLink>
        
        {/* NEW LINK HERE */}
        <NavLink to="/benchmark" className={navClass}>
          <BarChart2 className="w-5 h-5" />
          <span>Benchmarks</span>
        </NavLink>

        <NavLink to="/config" className={navClass}>
          <Settings className="w-5 h-5" />
          <span>Configuration</span>
        </NavLink>
      </nav>

      <div className="mt-auto text-xs text-slate-500 px-4">
        v1.1.0 Connected to :8050
      </div>
    </div>
  );
};

export default Sidebar;
