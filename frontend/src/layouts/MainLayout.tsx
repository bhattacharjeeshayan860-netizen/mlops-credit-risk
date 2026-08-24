import { Outlet } from 'react-router-dom';
import { Sidebar } from '../components/Sidebar';

export default function MainLayout() {
  return (
    <div className="flex h-screen w-full overflow-hidden">
      <Sidebar />
      <main className="app-main flex-1 overflow-y-auto relative">
        <div className="page-topbar">
          <span className="page-topbar__eyebrow">MLOps control center</span>
          <div className="page-topbar__status"><span /> Production environment</div>
        </div>
        <div className="min-h-full">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
