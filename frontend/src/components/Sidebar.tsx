import { Activity, Cpu, Database, Globe, Settings, ShieldAlert } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';
import type { NavItem } from '../types/navigation';
import { cn } from '../lib/utils';

const navItems: NavItem[] = [
  { label: 'Risk Assessment', path: '/', icon: <ShieldAlert size={18} /> },
  { label: 'Monitoring', path: '/monitoring', icon: <Activity size={18} /> },
  { label: 'Model', path: '/model', icon: <Database size={18} /> },
  { label: 'System Status', path: '/system', icon: <Settings size={18} /> },
];

interface SidebarProps {
  className?: string;
}

export function Sidebar({ className }: SidebarProps) {
  const location = useLocation();

  return (
    <aside className={cn('flex h-screen w-72 shrink-0 flex-col border-r border-slate-200 bg-gradient-to-b from-slate-950 via-slate-900 to-slate-800 text-slate-300', className)}>
      <div className="border-b border-slate-800/80 p-6">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-teal-500/15 ring-1 ring-inset ring-teal-400/30">
            <ShieldAlert size={20} className="text-teal-300" />
          </div>
          <div>
            <div className="text-lg font-black tracking-tight text-white">Credit Risk</div>
            <div className="text-xs font-medium uppercase tracking-[0.2em] text-teal-300">Intelligence</div>
          </div>
        </div>
      </div>

      <nav className="flex-1 space-y-2 px-4 py-5">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <Link
              key={item.path}
              to={item.path}
              className={cn(
                'flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-all duration-200',
                isActive ? 'bg-teal-500/12 text-teal-200 shadow-inner ring-1 ring-inset ring-teal-400/20' : 'text-slate-300 hover:bg-slate-800 hover:text-white'
              )}
            >
              <span className={cn('transition-colors', isActive ? 'text-teal-300' : 'text-slate-400')}>
                {item.icon}
              </span>
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="space-y-3 border-t border-slate-800/80 p-4 text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-emerald-400" />
          API: operational
        </div>
        <div className="flex items-center gap-2">
          <Cpu size={12} />
          Model: v0.1.0
        </div>
        <div className="flex items-center gap-2">
          <Globe size={12} />
          Env: production
        </div>
      </div>
    </aside>
  );
}

