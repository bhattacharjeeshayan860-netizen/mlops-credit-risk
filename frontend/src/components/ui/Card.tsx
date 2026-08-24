import React from 'react';
import { cn } from '../../lib/utils';

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  header?: string;
  headerClassName?: string;
  footer?: React.ReactNode;
}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, header, headerClassName, footer, children, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn('bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden', className)}
        {...props}
      >
        {header && (
          <div className={cn('px-6 py-4 border-b border-slate-100 font-semibold text-slate-800', headerClassName)}>
            {header}
          </div>
        )}
        <div className="p-6">{children}</div>
        {footer && <div className="px-6 py-4 border-t border-slate-100 bg-slate-50/50">{footer}</div>}
      </div>
    );
  }
);

Card.displayName = 'Card';
