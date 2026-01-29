import React, { useState } from 'react';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { DocumentManager } from './pages/DocumentManager';
import { Scraper } from './pages/Scraper';
import { Login } from './pages/Login';
import { ViewState } from './types';

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [currentView, setCurrentView] = useState<ViewState>('dashboard');

  if (!isLoggedIn) {
    return <Login onLogin={() => setIsLoggedIn(true)} />;
  }

  return (
    <Layout 
      currentView={currentView} 
      setView={setCurrentView}
      onLogout={() => setIsLoggedIn(false)}
    >
      {currentView === 'dashboard' && <Dashboard />}
      {currentView === 'documents' && <DocumentManager />}
      {currentView === 'scraping' && <Scraper />}
      {/* Settings or other pages could go here */}
    </Layout>
  );
}

export default App;
