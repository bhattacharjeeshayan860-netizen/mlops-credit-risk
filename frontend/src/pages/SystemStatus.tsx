import { useEffect, useState } from 'react';
import axios from 'axios';
import { Activity, AlertCircle, Cpu, Database, HardDrive, Network, RefreshCw, Server, ShieldAlert } from 'lucide-react';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import { Skeleton } from '../components/ui/Skeleton';
import { StatusIndicator } from '../components/StatusIndicator';
import { cn } from '../lib/utils';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface SystemStatusResponse {
  api: { status: string; detail: string };
  model: { status: string; detail: string };
  mlflow: { status: string; detail: string };
  monitoring: { status: string; detail: string };
}

function getFriendlyDetail(service: keyof SystemStatusResponse, detail?: string) {
  if (!detail) return 'Checking...';
  if (service === 'mlflow' && detail.toLowerCase().includes('connectivity issue')) {
    return 'MLflow server is unavailable';
  }
  if (detail.length > 72) return `${detail.slice(0, 69)}...`;
  return detail;
}

export default function SystemStatus() {
  const [statusData, setStatusData] = useState<SystemStatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const checkAll = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.get<SystemStatusResponse>(`${API_URL}/system/status`);
      setStatusData(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Unable to reach the system infrastructure.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void Promise.resolve().then(checkAll);
  }, []);

  const getStatusVariant = (status: string) => {
    if (status === 'operational') return 'success';
    if (status === 'warning') return 'warning';
    if (status === 'failure') return 'error';
    return 'neutral';
  };

  return (
    <div className="mx-auto max-w-5xl space-y-10 p-6 lg:p-10">
      <header className="flex flex-col justify-between gap-6 md:flex-row md:items-end">
        <div className="space-y-2">
          <div className="mb-2 flex items-center gap-2">
            <ShieldAlert className="text-teal-500" size={22} />
            <Badge variant="teal">Infrastructure</Badge>
          </div>
          <h1 className="text-4xl font-black tracking-tight text-slate-900">System Status</h1>
          <p className="text-lg text-slate-500">Operational health of the MLOps stack.</p>
        </div>

        <Button variant="outline" onClick={checkAll} disabled={loading}>
          <RefreshCw size={16} className={cn('mr-2', loading && 'animate-spin')} />
          Check health
        </Button>
      </header>

      {error && !statusData && (
        <div className="flex items-center gap-3 rounded-2xl border border-red-200 bg-red-50 p-4 text-red-700">
          <AlertCircle size={18} />
          <div className="flex-1">
            <p className="text-sm font-bold">Connection Error</p>
            <p className="text-xs opacity-90">{error}</p>
          </div>
          <Button variant="outline" size="sm" onClick={checkAll}>Retry</Button>
        </div>
      )}

      <div className="grid grid-cols-1 gap-8 md:grid-cols-2">
        <Card header="Core services">
          <div className="space-y-6">
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="rounded-xl bg-sky-50 p-2.5 text-sky-600">
                  <Server size={20} />
                </div>
                <div>
                  <p className="font-bold text-slate-900">FastAPI engine</p>
                  <p className="text-xs text-slate-500" title={statusData?.api.detail}>{getFriendlyDetail('api', statusData?.api.detail)}</p>
                </div>
              </div>
              <StatusIndicator variant={getStatusVariant(statusData?.api.status || 'loading')} label={(statusData?.api.status || 'loading').toUpperCase()} />
            </div>

            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="rounded-xl bg-teal-50 p-2.5 text-teal-600">
                  <Database size={20} />
                </div>
                <div>
                  <p className="font-bold text-slate-900">Champion model</p>
                  <p className="text-xs text-slate-500" title={statusData?.model.detail}>{getFriendlyDetail('model', statusData?.model.detail)}</p>
                </div>
              </div>
              <StatusIndicator variant={getStatusVariant(statusData?.model.status || 'loading')} label={(statusData?.model.status || 'loading').toUpperCase()} />
            </div>
          </div>
        </Card>

        <Card header="MLOps infrastructure">
          <div className="space-y-6">
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="rounded-xl bg-indigo-50 p-2.5 text-indigo-600">
                  <Activity size={20} />
                </div>
                <div>
                  <p className="font-bold text-slate-900">MLflow registry</p>
                  <p className="text-xs text-slate-500" title={statusData?.mlflow.detail}>{getFriendlyDetail('mlflow', statusData?.mlflow.detail)}</p>
                </div>
              </div>
              <StatusIndicator variant={getStatusVariant(statusData?.mlflow.status || 'loading')} label={(statusData?.mlflow.status || 'loading').toUpperCase()} />
            </div>

            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="rounded-xl bg-amber-50 p-2.5 text-amber-600">
                  <ShieldAlert size={20} />
                </div>
                <div>
                  <p className="font-bold text-slate-900">Drift detection</p>
                  <p className="text-xs text-slate-500" title={statusData?.monitoring.detail}>{getFriendlyDetail('monitoring', statusData?.monitoring.detail)}</p>
                </div>
              </div>
              <StatusIndicator variant={getStatusVariant(statusData?.monitoring.status || 'loading')} label={(statusData?.monitoring.status || 'loading').toUpperCase()} />
            </div>
          </div>
        </Card>
      </div>

      <Card header="Infrastructure details">
        <div className="grid grid-cols-1 gap-8 sm:grid-cols-3">
          <div className="space-y-1">
            <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">Compute</p>
            <div className="flex items-center gap-2 text-slate-900">
              <Cpu size={16} className="text-slate-400" />
              <span className="text-sm font-medium">Containerized</span>
            </div>
          </div>

          <div className="space-y-1">
            <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">Storage</p>
            <div className="flex items-center gap-2 text-slate-900">
              <HardDrive size={16} className="text-slate-400" />
              <span className="text-sm font-medium">Persistent bind mounts</span>
            </div>
          </div>

          <div className="space-y-1">
            <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">Network</p>
            <div className="flex items-center gap-2 text-slate-900">
              <Network size={16} className="text-slate-400" />
              <span className="text-sm font-medium">Docker bridge</span>
            </div>
          </div>
        </div>
      </Card>

      {loading && statusData && (
        <div className="grid grid-cols-1 gap-8 md:grid-cols-2">
          <div className="space-y-6 opacity-50">
            <Skeleton className="h-12 w-full" />
            <Skeleton className="h-12 w-full" />
          </div>
          <div className="space-y-6 opacity-50">
            <Skeleton className="h-12 w-full" />
            <Skeleton className="h-12 w-full" />
          </div>
        </div>
      )}
    </div>
  );
}
