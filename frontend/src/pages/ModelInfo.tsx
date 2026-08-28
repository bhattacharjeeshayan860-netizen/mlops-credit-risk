import { useEffect, useState } from 'react';
import axios from 'axios';
import { Activity, Calendar, CheckCircle2, Database, Hash, Layers, Zap, RefreshCw } from 'lucide-react';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import { Skeleton } from '../components/ui/Skeleton';
import { StatusIndicator } from '../components/StatusIndicator';

const API_URL = import.meta.env.VITE_API_URL || 'https://mlops-credit-risk.onrender.con';

interface ModelInfoResponse {
  model_type: string;
  version: string;
  trained_at: string;
  mlflow_run_id?: string;
  roc_auc?: number;
  average_precision?: number;
}

interface SystemStatusResponse {
  mlflow: { status: string; detail: string };
}

export default function ModelInfo() {
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [mlflowStatus, setMlflowStatus] = useState<SystemStatusResponse['mlflow'] | null>(null);

  const fetchModelInfo = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get<ModelInfoResponse>(`${API_URL}/model/info`);
      setModelInfo(response.data);
      try {
        const statusResponse = await axios.get<SystemStatusResponse>(`${API_URL}/system/status`);
        setMlflowStatus(statusResponse.data.mlflow);
      } catch {
        setMlflowStatus({ status: 'unknown', detail: 'Status check unavailable' });
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Unable to retrieve model information.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void Promise.resolve().then(fetchModelInfo);
  }, []);

  if (loading) {
    return (
      <div className="space-y-8 p-10">
        <Skeleton className="h-10 w-64" />
        <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
          <Skeleton className="h-40" />
          <Skeleton className="h-40" />
          <Skeleton className="h-40" />
        </div >
        <Skeleton className="h-80" />
      </div >
    );
  }

  if (error || !modelInfo) {
    return (
      <div className="flex min-h-[60vh] flex-col items-center justify-center space-y-4">
        <div className="rounded-full bg-red-50 p-4 text-red-500">
          <Database size={32} />
        </div >
        <div className="text-center">
          <h2 className="text-xl font-bold text-slate-900">Model information unavailable</h2>
          <p className="text-slate-500">{error || 'No model registry entry was found in MLflow.'}</p>
        </div >
        <Button variant="outline" onClick={fetchModelInfo} disabled={loading}>
          <RefreshCw size={16} className="mr-2" />
          Retry
        </Button>
      </div >
    );
  }

  return (
    <div className="mx-auto max-w-6xl space-y-10 p-6 lg:p-10">
      <header className="flex flex-col justify-between gap-6 md:flex-row md:items-end">
        <div className="space-y-2">
          <div className="mb-2 flex items-center gap-2">
            <Badge variant="teal">Registry</Badge>
            <span className="text-xs text-slate-400">Verified champion</span >
          </div >
          <h1 className="text-4xl font-black tracking-tight text-slate-900">Model Registry</h1>
          <p className="text-lg text-slate-500">Technical specification for the active production model.</p>
        </div >

        <div className="flex items-center gap-3 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-2">
          <div className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
          <span className="text-xs font-bold uppercase tracking-[0.2em] text-emerald-700">Serving live</span >
        </div >
      </header>

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-3">
        <div className="space-y-8 lg:col-span-2">
          <Card header="Model specifications">
            <div className="grid grid-cols-1 gap-x-12 gap-y-8 sm:grid-cols-2">
              <div className="space-y-1.5">
                <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">Model type</p>
                <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                  <Layers size={18} className="text-teal-500" />
                  {modelInfo.model_type}
                </div >
              </div >

              <div className="space-y-1.5">
                <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">Version</p>
                <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                  <Hash size={18} className="text-teal-500" />
                  v{modelInfo.version}
                </div >
              </div >

              <div className="space-y-1.5">
                <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">Training date</p>
                <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                  <Calendar size={18} className="text-teal-500" />
                  {modelInfo.trained_at}
                </div >
              </div >

              <div className="space-y-1.5">
                <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">MLflow run id</p>
                <div className="flex items-center gap-2 truncate font-mono text-xs text-slate-800">
                  <Activity size={18} className="shrink-0 text-teal-500" />
                  {modelInfo.mlflow_run_id || 'N/A'}
                </div >
              </div >
            </div >
          </Card>

          <Card header="Performance metrics">
            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
              <div className="space-y-2 rounded-2xl border border-slate-200 bg-slate-50 p-6">
                <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">ROC-AUC</p>
                <div className="flex items-baseline gap-2">
                  <span className="text-3xl font-black text-slate-900">{modelInfo.roc_auc ? modelInfo.roc_auc.toFixed(4) : 'N/A'}</span>
                  <span className="text-sm font-semibold text-emerald-600">High</span >
                </div >
              </div >

              <div className="space-y-2 rounded-2xl border border-slate-200 bg-slate-50 p-6">
                <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">PR-AUC</p>
                <div className="flex items-baseline gap-2">
                  <span className="text-3xl font-black text-slate-900">{modelInfo.average_precision ? modelInfo.average_precision.toFixed(4) : 'N/A'}</span>
                  <span className="text-sm font-semibold text-emerald-600">Stable</span >
                </div >
              </div >
            </div >
          </Card>
        </div >

        <div className="space-y-8">
          <Card header="Deployment lifecycle">
            <div className="relative space-y-8 before:absolute before:bottom-2 before:left-[15px] before:top-2 before:w-0.5 before:bg-slate-100">
              <div className="group relative flex items-center gap-4 pl-8">
                <div className="absolute left-0 flex h-8 w-8 items-center justify-center rounded-full border-4 border-white bg-slate-100 text-slate-400">
                  <Layers size={14} />
                </div >
                <div>
                  <p className="text-sm font-bold text-slate-400">Candidate</p>
                  <p className="text-[11px] text-slate-500">Evaluation phase</p>
                </div >
              </div >

              <div className="group relative flex items-center gap-4 pl-8">
                <div className="absolute left-0 flex h-8 w-8 items-center justify-center rounded-full border-4 border-white bg-teal-500 text-white shadow-sm">
                  <CheckCircle2 size={14} />
                </div >
                <div>
                  <p className="text-sm font-bold text-teal-600">Champion</p>
                  <p className="text-[11px] font-medium text-teal-700">Currently serving</p>
                </div >
              </div >

              <div className="group relative flex items-center gap-4 pl-8">
                <div className="absolute left-0 flex h-8 w-8 items-center justify-center rounded-full border-4 border-white bg-slate-100 text-slate-400">
                  <Zap size={14} />
                </div >
                <div>
                  <p className="text-sm font-bold text-slate-400">Production</p>
                  <p className="text-[11px] text-slate-500">FastAPI endpoint</p>
                </div >
              </div >
            </div >
          </Card>

          <div className="rounded-2xl bg-slate-900 p-6 text-white shadow-xl">
            <h3 className="mb-4 text-[10px] font-bold uppercase tracking-[0.22em] text-slate-400">System readiness</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-300">API service</span >
                <StatusIndicator variant="success" label="Ready" className="text-xs" />
              </div >
              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-300">MLflow registry</span >
                <StatusIndicator
                  variant={mlflowStatus?.status === 'operational' ? 'success' : mlflowStatus?.status === 'warning' ? 'warning' : 'neutral'}
                  label={mlflowStatus?.status === 'operational' ? 'Connected' : mlflowStatus?.status?.toUpperCase() || 'Checking'}
                  className="text-xs"
                />
              </div >
              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-300">Inference engine</span >
                <StatusIndicator variant="success" label="Active" className="text-xs" />
              </div >
            </div >
          </div >
        </div >
      </div >
    </div >
  );
}
