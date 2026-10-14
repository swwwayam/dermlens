import React, { useState } from 'react';
import Navbar from './components/Navbar';
import OverviewPage from './pages/OverviewPage';
import AnalyzePage from './pages/AnalyzePage';
import DashboardPage from './pages/DashboardPage';
import MethodologyPage from './pages/MethodologyPage';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');

  return (
    <div className={`app-shell ${activeTab === 'overview' ? 'is-overview' : ''}`}>
      
      {/* Floating Glass Navigation */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Page Area */}
      <main className="app-main">
        <div className="container">
          {activeTab === 'overview' && <OverviewPage onNavigate={setActiveTab} />}
          {activeTab === 'analyze' && <AnalyzePage />}
          {activeTab === 'dashboard' && <DashboardPage />}
          {activeTab === 'methodology' && <MethodologyPage />}
        </div>
      </main>

      {/* Glass Footer */}
      <footer className="app-footer">
        <div className="container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <strong style={{ color: 'var(--text-primary)' }}>DermLens</strong> &bull; Dermatology image retrieval
          </div>
          <div style={{ display: 'flex', gap: '20px' }}>
            <button onClick={() => setActiveTab('overview')} style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer' }}>Overview</button>
            <button onClick={() => setActiveTab('analyze')} style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer' }}>Analyze</button>
            <button onClick={() => setActiveTab('dashboard')} style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer' }}>Dashboard</button>
            <button onClick={() => setActiveTab('methodology')} style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer' }}>Methodology</button>
          </div>
        </div>
      </footer>

    </div>
  );
}
