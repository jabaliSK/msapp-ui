import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import ChatPage from './pages/ChatPage';
import FilesPage from './pages/FilesPage';
import ConfigPage from './pages/ConfigPage';
import StatsPage from './pages/StatsPage';
import BenchmarkPage from './pages/BenchmarkPage';
// 1. Import the new page
import ConversationPage from './pages/ConversationPage'; 

const App = () => {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <Router>
      <div className="flex h-screen bg-slate-50">
        {/* Sidebar */}
        <Sidebar collapsed={collapsed} toggle={() => setCollapsed(!collapsed)} />

        {/* Main Content Area */}
        <div 
          className={`flex-1 transition-all duration-300 flex flex-col overflow-hidden ${
            collapsed ? 'ml-20' : 'ml-64'
          }`}
        >
          <Routes>
            <Route path="/" element={<ChatPage />} />
            {/* 2. Add the new Route here */}
            <Route path="/conversation" element={<ConversationPage />} />
            
            <Route path="/files" element={<FilesPage />} />
            <Route path="/config" element={<ConfigPage />} />
            <Route path="/stats" element={<StatsPage />} />
            <Route path="/benchmark" element={<BenchmarkPage />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
};

export default App;
