import { CheckCircle2, AlertCircle, Info } from 'lucide-react';
import type { PredictionResponse } from '../../types/prediction';
import { cn } from '../../lib/utils';
import { Badge } from '../../components/ui/Badge';

interface RiskResultProps {
  result: PredictionResponse;
}

export default function RiskResult({ result }: RiskResultProps) {
  const isHighRisk = result.risk_label === 'high_risk';
  const probability = (result.default_probability * 100).toFixed(1);

  const statusColors = {
    low_risk: 'text-emerald-600 bg-emerald-50 border-emerald-200',
    high_risk: 'text-red-600 bg-red-50 border-red-200',
  };

  const barColors = {
    low_risk: 'bg-emerald-500',
    high_risk: 'bg-red-500',
  };

  return (
    <div className="bg-white rounded-2xl shadow-xl border border-slate-200 overflow-hidden animate-in zoom-in-95 fade-in duration-500">
      <div className={cn("px-6 py-5 border-b text-center flex items-center justify-center gap-2", statusColors[result.risk_label])}>
        {isHighRisk ? <AlertCircle size={22} /> : <CheckCircle2 size={22} />}
        <span className="text-lg font-extrabold tracking-widest uppercase">
          {result.risk_label.replace('_', ' ')}
        </span>
      </div>

      <div className="p-8 space-y-8">
        <div className="text-center space-y-2">
          <p className="text-[11px] font-bold text-slate-400 uppercase tracking-[0.2em]">Estimated Default Probability</p>
          <div className="flex items-baseline justify-center gap-1">
            <span className="text-6xl font-black text-slate-900 tracking-tight">{probability}</span>
            <span className="text-2xl font-bold text-slate-400">%</span>
          </div>
        </div>

        {/* Risk Meter */}
        <div className="space-y-4">
          <div className="h-4 w-full bg-slate-100 rounded-full overflow-hidden shadow-inner ring-1 ring-slate-200/50">
            <div 
              className={cn("h-full transition-all duration-1000 cubic-bezier(0.34, 1.56, 0.64, 1) rounded-full shadow-sm", barColors[result.risk_label])}
              style={{ width: `${result.default_probability * 100}%` }}
            />
          </div>
          <div className="flex justify-between text-[10px] font-black text-slate-400 uppercase tracking-tighter">
            <span className="opacity-60">Low Risk</span>
            <span className="opacity-60">Moderate</span>
            <span className="opacity-60">High Risk</span>
          </div>
        </div>

        <div className="pt-8 border-t border-slate-100 space-y-5">
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-1">
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Model Version</p>
              <div className="flex items-center gap-2">
                <Badge variant="teal">v{result.model_version}</Badge>
              </div>
            </div>
            <div className="space-y-1 text-right">
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Timestamp</p>
              <p className="text-xs font-bold text-slate-700">{new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</p>
            </div>
          </div>
          <div className="space-y-1">
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Run ID</p>
            <div className="flex items-center justify-between bg-slate-50 px-3 py-1.5 rounded-lg border border-slate-100">
              <p className="text-[10px] font-mono text-slate-500 truncate">{result.mlflow_run_id}</p>
              <button 
                onClick={() => navigator.clipboard.writeText(result.mlflow_run_id)}
                className="text-[10px] text-teal-600 font-bold hover:text-teal-700 transition-colors"
              >
                Copy
              </button>
            </div>
          </div>
        </div>

        <div className="pt-6 border-t border-slate-100 flex gap-3 items-start text-[11px] text-slate-400 italic leading-relaxed">
          <Info size={14} className="shrink-0 mt-0.5 opacity-60" />
          <p>
            This prediction is an estimate produced by a machine-learning model and is not a lending decision or a substitute for human review.
          </p>
        </div>
      </div>
    </div>
  );
}
