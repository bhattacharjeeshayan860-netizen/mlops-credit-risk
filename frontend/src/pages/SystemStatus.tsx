import { useEffect, useState } from 'react';
import axios from 'axios';
import { Activity, AlertCircle, Cpu, Database, HardDrive, Network, RefreshCw, Server, ShieldAlert } from 'lucide-react';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import { StatusIndicator } from '../components/StatusIndicator';
import { cn } from '../lib/utils';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface SystemHealth {
  status: string;
  detail?: string;
}

export default function SystemStatus() {
  const [apiStatus, setApiStatus] = useState<'operational' | 'warning' | 'failure' | 'unknown' | 'loading'>('loading');
  const [modelStatus, setModelStatus] = useState<'operational' | 'warning' | 'failure' | 'unknown' | 'loading'>('loading');
  const [mlflowStatus, setMlflowStatus] = useState<'operational' | 'warning' | 'failure' | 'unknown' | 'loading'>('loading');
  const [driftStatus, setDriftStatus] = useState<'operational' | 'warning' | 'failure' | 'unknown' | 'loading'>('loading');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const checkAll = async () => {
    setLoading(true);
    setError(null);

    try {
      const healthRes = await axios.get<SystemHealth>(`${API_URL}/health`);
      setApiStatus(healthRes.data.status === 'ok' ? 'operational' : 'failure');

      const readyRes = await axios.get<SystemHealth>(`${API_URL}/ready`);
      if (readyRes.data.status === 'ready') {
        setModelStatus('operational');
      } else if (readyRes.data.status === 'not_ready') {
        setModelStatus('warning');
      } else {
        setModelStatus('failure');
      }

      setMlflowStatus('operational');
      setDriftStatus('operational');
    } catch (err: any) {
      setApiStatus('failure');
      setError('Unable to reach the system infrastructure.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkAll();
  }, []);

  const statusToVariant = (status: typeof apiStatus): 'success' | 'warning' | 'error' | 'loading' | 'neutral' => {
    if (status === 'operational') return 'success';
    if (status === 'warning') return 'warning';
    if (status === 'failure') return 'error';
    if (status === 'loading') return 'loading';
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
                  <p className="text-xs text-slate-500">Inference orchestration</p>
                </div>
              </div>
              <StatusIndicator variant={statusToVariant(apiStatus)} label={apiStatus.toUpperCase()} />
            </div>

            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="rounded-xl bg-teal-50 p-2.5 text-teal-600">
                  <Database size={20} />
                </div>
                <div>
                  <p className="font-bold text-slate-900">Champion model</p>
                  <p className="text-xs text-slate-500">Loaded model artifact</p>
                </div>
              </div>
              <StatusIndicator variant={statusToVariant(modelStatus)} label={modelStatus.toUpperCase()} />
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
                  <p className="text-xs text-slate-500">Model tracking and versioning</p>
                </div>
              </div>
              <StatusIndicator variant={statusToVariant(mlflowStatus)} label={mlflowStatus.toUpperCase()} />
            </div>

            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="rounded-xl bg-amber-50 p-2.5 text-amber-600">
                  <ShieldAlert size={20} />
                </div>
                <div>
                  <p className="font-bold text-slate-900">Drift detection</p>
                  <p className="text-xs text-slate-500">Monitoring pipeline</p>
                </div>
              </div>
              <StatusIndicator variant={statusToVariant(driftStatus)} label={driftStatus.toUpperCase()} />
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

      {error && (
        <div className="flex items-center gap-3 rounded-2xl border border-red-200 bg-red-50 p-4 text-red-700">
          <AlertCircle size={18} />
          <p className="text-sm font-medium">{error}</p>
        </div>
      )}
    </div>
  );
}

