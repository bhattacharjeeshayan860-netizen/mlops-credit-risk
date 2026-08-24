import { 
  CheckCircle2, 
  AlertCircle, 
  XCircle, 
  Circle, 
  Loader2 
} from 'lucide-react';
import { cn } from '../lib/utils';

export type StatusVariant = 'success' | 'warning' | 'error' | 'info' | 'loading' | 'neutral';

interface StatusIndicatorProps {
  variant: StatusVariant;
  label: string;
  className?: string;
}

export const StatusIndicator = ({ variant, label, className }: StatusIndicatorProps) => {
  const variants = {
    success: {
      icon: <CheckCircle2 className="text-emerald-500" size={16} />,
      labelClass: 'text-emerald-700',
    },
    warning: {
      icon: <AlertCircle className="text-amber-500" size={16} />,
      labelClass: 'text-amber-700',
    },
    error: {
      icon: <XCircle className="text-red-500" size={16} />,
      labelClass: 'text-red-700',
    },
    info: {
      icon: <Circle className="text-blue-500" size={16} />,
      labelClass: 'text-blue-700',
    },
    loading: {
      icon: <Loader2 className="animate-spin text-slate-400" size={16} />,
      labelClass: 'text-slate-500',
    },
    neutral: {
      icon: <Circle className="text-slate-400" size={16} />,
      labelClass: 'text-slate-500',
    },
  };

  const config = variants[variant];

  return (
    <div className={cn("flex items-center gap-2 font-medium text-sm", className)}>
      {config.icon}
      <span className={config.labelClass}>{label}</span>
    </div>
  );
};
