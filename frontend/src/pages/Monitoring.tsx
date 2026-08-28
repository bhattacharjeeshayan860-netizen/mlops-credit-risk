import React from 'react';
import axios from 'axios';
import { Activity, AlertTriangle, BarChart3, Database, TrendingUp, RefreshCw } from 'lucide-react';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import { Skeleton } from '../components/ui/Skeleton';
import { cn } from '../lib/utils';

const API_URL = import.meta.env.VITE_API_URL || 'https://mlops-credit-risk.onrender.con';

interface MonitoringStats {
  total_predictions: number;
  high_risk_rate: number;
  avg_default_probability: number;
  recent_volume: number;
}

interface SystemStatusResponse {
  monitoring: { status: string; detail: string };
}

interface LatestReportResponse {
  report_name: string;
}

export default function Monitoring() {
  const [stats, setStats] = React.useState<MonitoringStats | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [driftStatus, setDriftStatus] = React.useState<SystemStatusResponse['monitoring'] | null>(null);
  const [latestReport, setLatestReport] = React.useState<string | null>(null);

  const fetchStats = async () => {
    setLoading(true);
    setError(null);
    try {
      const [statsRes, statusRes, reportRes] = await Promise.allSettled([
        axios.get<MonitoringStats>(`${API_URL}/monitoring/stats`),
        axios.get<SystemStatusResponse>(`${API_URL}/system/status`),
        axios.get<LatestReportResponse>(`${API_URL}/reports/latest`)
      ]);

      if (statsRes.status === 'fulfilled') {
        setStats(statsRes.value.data);
      } else {
        throw statsRes.reason;
      }

      if (statusRes.status === 'fulfilled') {
        setDriftStatus(statusRes.value.data.monitoring);
      } else {
        setDriftStatus({ status: 'unknown', detail: 'Status check unavailable' });
      }

      if (reportRes.status === 'fulfilled') {
        setLatestReport(reportRes.value.data.report_name);
      } else {
        setLatestReport(null);
      }

    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Unable to fetch monitoring data.');
    } finally {
      setLoading(false);
    }
  };

  React.useEffect(() => {
    void Promise.resolve().then(fetchStats);
  }, []);

  const handleViewReport = () => {
    if (latestReport) {
      window.open(`${API_URL}/reports/${latestReport}`, '_blank');
    }
  };

  if (loading && !stats) {
    return (
      <div className="space-y-8 p-10">
        <div className="space-y-2">
          <Skeleton className="h-10 w-64" />
          <Skeleton className="h-6 w-96" />
        </div >
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
          {[1, 2, 3, 4].map((item) => (
            <div key={item} className="space-y-4 rounded-2xl border border-slate-200 bg-white p-6">
              <Skeleton className="h-4 w-24" />
              <Skeleton className="h-8 w-32" />
            </div >
          ))}
        </div >
      </div >
    );
  }

  if (error) {
    return (
      <div className="flex min-h-[60vh] flex-col items-center justify-center space-y-4 p-10 text-center">
        <div className="rounded-full bg-red-50 p-4 text-red-500">
          <Activity size={32} />
        </div >
        <div className="space-y-1">
          <h2 className="text-xl font-bold text-slate-900">Monitoring unavailable</h2>
          <p className="text-slate-500">{error}</p>
        </div >
        <Button variant="outline" onClick={fetchStats}>
          <RefreshCw size={16} className="mr-2" />
          Retry connection
        </Button>
      </div >
    );
  }

  if (!stats || stats.total_predictions === 0) {
    return (
      <div className="flex min-h-[60vh] flex-col items-center justify-center space-y-4 p-10 text-center">
        <div className="rounded-full bg-slate-100 p-8 text-slate-400">
          <BarChart3 size={56} />
        </div >
        <div className="space-y-2">
          <h2 className="text-2xl font-bold text-slate-900">No production data yet</h2>
          <p className="max-w-md text-slate-500">
            Historical prediction logs are required before monitoring metrics can be generated.
          </p>
        </div >
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={fetchStats} disabled={loading}>
            <RefreshCw size={16} className={cn('mr-2', loading && 'animate-spin')} />
            Refresh
          </Button>
        </div >
      </div >
    );
  }

  return (
    <div className="space-y-10 p-6 lg:p-10">
      <header className="flex flex-col justify-between gap-4 md:flex-row md:items-end">
        <div className="space-y-2">
          <Badge variant="teal">Live Insights</Badge>
          <h1 className="text-4xl font-black tracking-tight text-slate-900">Monitoring Dashboard</h1>
          <p className="text-lg text-slate-500">Operational overview of model performance and prediction trends.</p>
        </div >

        <div className="flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-semibold text-emerald-700">
          <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
          Live data feed
        </div >

        <Button variant="outline" onClick={fetchStats} disabled={loading}>
          <RefreshCw size={16} className={cn('mr-2', loading && 'animate-spin')} />
          Refresh
        </Button>
      </header>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
        <StatCard title="Total Predictions" value={stats.total_predictions.toLocaleString()} description="Inferences served" icon={<Database className="text-sky-500" />} />
        <StatCard title="High Risk Rate" value={`${(stats.high_risk_rate * 100).toFixed(1)}%`} description="Share of high-risk cases" variant={stats.high_risk_rate > 0.2 ? 'danger' : 'default'} icon={<AlertTriangle className={cn('text-amber-500', stats.high_risk_rate > 0.2 && 'text-red-500')} />} />
        <StatCard title="Avg. Probability" value={`${(stats.avg_default_probability * 100).toFixed(1)}%`} description="Mean predicted default probability" icon={<TrendingUp className="text-teal-500" />} />
        <StatCard title="Recent Volume" value={stats.recent_volume.toString()} description="Latest 100 requests" icon={<Activity className="text-slate-500" />} />
      </div >

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-3">
        <Card className="lg:col-span-2" header="Prediction Trends">
          <div className="flex h-80 flex-col items-center justify-center space-y-3 rounded-2xl border border-dashed border-slate-200 bg-slate-50 text-slate-400">
            <BarChart3 size={42} className="opacity-20" />
            <div className="text-center">
              <p className="text-sm font-semibold text-slate-600">Insufficient historical data</p>
              <p className="text-xs text-slate-500">This chart will populate as more production requests are recorded.</p>
            </div >
          </div >
        </Card>

        <div className="space-y-6">
          <Card header="Drift Status">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-slate-500">Current state</span >
                <Badge variant={driftStatus?.status === 'operational' ? 'success' : driftStatus?.status === 'warning' ? 'warning' : 'outline'}>
                  {driftStatus?.status === 'operational' ? 'Healthy' : driftStatus?.status?.toUpperCase() || 'CHECKING'}
                </Badge>
              </div >
              <div className="space-y-1">
                <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">Summary</p>
                <p className="text-sm text-slate-700">{driftStatus?.detail || 'Checking monitoring pipeline...'}</p>
              </div >
              <Button 
                variant="outline" 
                className="w-full text-xs"
                onClick={handleViewReport}
                disabled={!latestReport}
              >
                {latestReport ? 'View drift report' : 'No report available'}
              </Button>
            </div >
          </Card>

          <Card header="Active Model">
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-teal-50 text-teal-600">
                  <Database size={22} />
                </div >
                <div>
                  <p className="text-sm font-bold text-slate-900">Champion v0.1.0</p>
                  <p className="text-xs text-slate-500">Logistic regression</p>
                </div >
              </div >

              <div className="pt-2">
                <div className="mb-1.5 flex items-center justify-between text-xs">
                  <span className="font-medium text-slate-500">Model health</span >
                  <span className="font-bold text-emerald-600">98.2% stable</span >
                </div >
                <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
                  <div className="h-full w-[98%] rounded-full bg-emerald-500" />
                </div >
              </div >
            </div >
          </Card>
        </div >
      </div >
    </div >
  );
}

interface StatCardProps {
  title: string;
  value: string | number;
  description: string;
  icon: React.ReactNode;
  variant?: 'default' | 'warning' | 'danger' | 'success';
}

function StatCard({ title, value, description, icon, variant = 'default' }: StatCardProps) {
  return (
    <Card>
      <div className="mb-4 flex items-start justify-between">
        <div className="rounded-xl bg-slate-100 p-2.5 text-slate-600">{icon}</div >
        <Badge variant={variant === 'danger' ? 'danger' : variant === 'warning' ? 'warning' : variant === 'success' ? 'success' : 'default'}>Live</Badge>
      </div >
      <div className="space-y-1">
        <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-400">{title}</p>
        <p className="text-3xl font-black tracking-tight text-slate-900">{value}</p>
        <p className="text-xs text-slate-500">{description}</p>
      </div >
    </Card>
  );
}
