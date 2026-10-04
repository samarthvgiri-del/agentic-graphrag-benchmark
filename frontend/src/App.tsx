import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import { Activity, LayoutDashboard, GitMerge, BarChart2, Server, Info, Search } from 'lucide-react';
import QueryPage from './pages/QueryPage';
import ComparePage from './pages/ComparePage';
import TracePage from './pages/TracePage';
import BenchmarkPage from './pages/BenchmarkPage';
import MetricsPage from './pages/MetricsPage';
import ArchitecturePage from './pages/ArchitecturePage';
import AboutPage from './pages/AboutPage';

function App() {
  return (
    <Router>
      <div className="app-container">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-header">
            <Activity className="text-accent" size={24} color="#3b82f6" />
            <span>Agentic Bench</span>
          </div>
          <nav className="sidebar-nav">
            <NavLink to="/" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <Search size={20} />
              <span>Query Testing</span>
            </NavLink>
            <NavLink to="/compare" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <LayoutDashboard size={20} />
              <span>Pipeline Compare</span>
            </NavLink>
            <NavLink to="/trace" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <GitMerge size={20} />
              <span>Agent Trace</span>
            </NavLink>
            <NavLink to="/benchmark" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <Activity size={20} />
              <span>Benchmark Runs</span>
            </NavLink>
            <NavLink to="/metrics" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <BarChart2 size={20} />
              <span>Metrics</span>
            </NavLink>
            <div style={{ margin: '2rem 1.5rem 1rem', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              System
            </div>
            <NavLink to="/architecture" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <Server size={20} />
              <span>Architecture</span>
            </NavLink>
            <NavLink to="/about" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <Info size={20} />
              <span>About</span>
            </NavLink>
          </nav>
        </aside>

        {/* Main Content */}
        <main className="main-content">
          <header className="topbar">
            <div style={{ flex: 1 }}></div>
            <div className="badge badge-blue">Hackathon Edition</div>
          </header>

          <div className="page-content">
            <Routes>
              <Route path="/" element={<QueryPage />} />
              <Route path="/compare" element={<ComparePage />} />
              <Route path="/trace" element={<TracePage />} />
              <Route path="/benchmark" element={<BenchmarkPage />} />
              <Route path="/metrics" element={<MetricsPage />} />
              <Route path="/architecture" element={<ArchitecturePage />} />
              <Route path="/about" element={<AboutPage />} />
            </Routes>
          </div>
        </main>
      </div>
    </Router>
  );
}

export default App;
