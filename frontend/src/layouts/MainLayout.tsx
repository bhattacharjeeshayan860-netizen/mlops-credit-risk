import { Outlet } from 'react-router-dom';
import { Sidebar } from '../components/Sidebar';

export default function MainLayout() {
  return (
    <div className="flex h-screen w-full bg-[#f8fafc] overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-y-auto relative">
        <div className="min-h-full">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
