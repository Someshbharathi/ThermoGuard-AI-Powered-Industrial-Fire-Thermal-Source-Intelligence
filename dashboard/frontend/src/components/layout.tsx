import { useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { Activity, AlertTriangle, BarChart3, Bell, ChevronDown, Flame, LayoutDashboard, Menu, Radio, Search, Settings, Shield, Sparkles, X } from 'lucide-react'

const navigation = [
  { label: 'Overview', to: '/', icon: LayoutDashboard },
  { label: 'Live Map', to: '/map', icon: Radio },
  { label: 'Events', to: '/events', icon: Flame },
  { label: 'Alerts', to: '/alerts', icon: AlertTriangle },
  { label: 'Analytics', to: '/analytics', icon: BarChart3 },
  { label: 'AI Insights', to: '/insights', icon: Sparkles },
  { label: 'System', to: '/system', icon: Settings },
]

function Sidebar({ open, onClose }: { open: boolean; onClose: () => void }) {
  return <aside className={`sidebar ${open ? 'sidebar-open' : ''}`}>
    <div className="brand"><div className="brand-mark"><Shield size={19} /></div><div><div className="brand-name">THERMO<span>GUARD</span></div><div className="brand-subtitle">AI-POWERED INTELLIGENCE</div></div><button className="icon-button mobile-close" title="Close navigation" onClick={onClose}><X size={18} /></button></div>
    <div className="nav-label">Command modules</div>
    <nav>{navigation.map(({ label, to, icon: Icon }) => <NavLink key={to} to={to} end={to === '/'} onClick={onClose} className={({ isActive }) => isActive ? 'nav-item active' : 'nav-item'}><Icon size={17} /><span>{label}</span>{label === 'Alerts' && <span className="nav-count">03</span>}</NavLink>)}</nav>
    <div className="sidebar-footer"><div className="nav-label">System status</div><div className="status-row"><span className="status-dot" />AI engine online</div><div className="status-row"><span className="status-dot" />Data pipeline operational</div><div className="demo-mode"><span>DEMO / DEVELOPMENT MODE</span><small>Dataset snapshot · 2024</small></div></div>
  </aside>
}

function Topbar({ onMenu }: { onMenu: () => void }) {
  const location = useLocation()
  const current = navigation.find((item) => item.to === location.pathname) ?? navigation[0]
  return <header className="topbar"><div className="topbar-title"><button className="icon-button menu-button" title="Open navigation" onClick={onMenu}><Menu size={20} /></button><div><div className="breadcrumb">THERMOGUARD <span>/</span> COMMAND CENTER</div><h1>{current.label === 'Overview' ? 'Thermal Intelligence Overview' : current.label}</h1><p>{current.label === 'Overview' ? 'Satellite-derived anomaly monitoring and risk intelligence' : 'ThermoGuard intelligence module · integration surface ready'}</p></div></div><div className="topbar-actions"><button className="icon-button" title="Search"><Search size={18} /></button><button className="icon-button notification" title="Notifications"><Bell size={18} /><span /></button><div className="top-status"><span className="status-dot" />Systems nominal</div><div className="user-menu"><div className="avatar">TG</div><span>Operator</span><ChevronDown size={14} /></div></div></header>
}

export function DashboardLayout() {
  const [menuOpen, setMenuOpen] = useState(false)
  return <div className="app-shell"><Sidebar open={menuOpen} onClose={() => setMenuOpen(false)} /><div className="main-shell"><Topbar onMenu={() => setMenuOpen(true)} /><main className="content"><Outlet /></main><footer className="app-footer"><span>THERMOGUARD INTELLIGENCE PLATFORM</span><span>Demo environment · Source: NASA FIRMS / Sentinel-2 / OSM</span></footer></div>{menuOpen && <button className="scrim" aria-label="Close navigation" onClick={() => setMenuOpen(false)} />}</div>
}

export { Activity }
