import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import ChatPage from './pages/ChatPage';
import FilesPage from './pages/FilesPage';
import ConfigPage from './pages/ConfigPage';
import BenchmarkPage from './pages/BenchmarkPage';
import StatsPage from './pages/StatsPage'; // Assuming you added this in the previous step

function App() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <Router>
      <div className="flex min-h-screen bg-slate-50">
        <Sidebar collapsed={collapsed} toggle={() => setCollapsed(!collapsed)} />
        
        {/* Dynamic margin based on collapsed state */}
	  <div className={`transition-all duration-300 min-w-0 flex-1 ${collapsed ? 'ml-20' : 'ml-64'}`}>
          <Routes>
            <Route path="/" element={<ChatPage />} />
            <Route path="/files" element={<FilesPage />} />
            <Route path="/benchmark" element={<BenchmarkPage />} />
            <Route path="/config" element={<ConfigPage />} />
            <Route path="/stats" element={<StatsPage />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
