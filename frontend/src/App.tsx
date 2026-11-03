import { Outlet, NavLink, useLocation } from 'react-router-dom';

export default function App() {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Dashboard' },
    { path: '/ask', label: 'Ask Operations Intelligence' },
    { path: '/datasets', label: 'Datasets' },
  ];

  return (
    <div className="app-layout">
      <aside className="app-sidebar">
        <div className="app-sidebar-header">
          <h1>Operations Intelligence</h1>
          <p>Phynapse</p>
        </div>
        <nav className="app-nav">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => (isActive ? 'active' : '')}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="app-sidebar-footer">
          Phase 1 — Deterministic Analytics
        </div>
      </aside>
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  );
}
