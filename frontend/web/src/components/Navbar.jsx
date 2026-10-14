import React from 'react';
import { Layers, BarChart3, BookOpen, UploadCloud } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab }) {
  const navItems = [
    { id: 'overview', label: 'Overview', icon: Layers },
    { id: 'analyze', label: 'Upload & Analyze', icon: UploadCloud },
    { id: 'dashboard', label: 'Metrics & Ablation', icon: BarChart3 },
    { id: 'methodology', label: 'Methodology', icon: BookOpen },
  ];

  return (
    <div className="floating-navbar-container">
      <header className="floating-nav-pill">
        
        {/* Brand Capsule */}
        <button
          type="button"
          className="nav-brand"
          aria-label="DermLens overview"
          onClick={() => setActiveTab('overview')} 
          style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', padding: '4px 10px 4px 6px' }}
        >
          <span style={{ fontSize: '1.05rem', fontWeight: '700', letterSpacing: '-0.02em', color: 'var(--text-primary)' }}>
            DermLens
          </span>
        </button>

        {/* Center Nav Pills */}
        <nav aria-label="Main navigation" style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                aria-current={isActive ? 'page' : undefined}
                onClick={() => setActiveTab(item.id)}
                className={`nav-pill-btn ${isActive ? 'active' : ''}`}
              >
                <Icon size={14} color={isActive ? '#0d1527' : 'currentColor'} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Right CTA */}
        <button 
          onClick={() => setActiveTab('analyze')} 
          className="btn btn-liquid-white btn-sm"
          style={{ marginLeft: '4px' }}
        >
          <UploadCloud size={14} />
          <span>New Query</span>
        </button>

      </header>
    </div>
  );
}
