'use client';

interface StatCardProps {
  label: string;
  value: string;
  change?: number;
  subvalue?: string;
  annotation?: string;
}

export default function StatCard({
  label,
  value,
  change,
  subvalue,
  annotation,
}: StatCardProps) {
  return (
    <div className="bg-stock border border-border rounded-sm p-5 flex flex-col justify-between gap-4">
      <div>
        <div className="flex items-start justify-between mb-3">
          <p className="dive-tag text-paper-tertiary">{label}</p>
          {annotation && (
            <span className="dive-tag text-paper-muted">{annotation}</span>
          )}
        </div>
        <div className="flex items-baseline gap-3">
          <p className="text-data-lg text-paper">{value}</p>
          {change !== undefined && (
            <span
              className="inline-block px-1.5 py-0.5 rounded-xs font-data text-[11px] font-semibold"
              style={{
                backgroundColor: change >= 0 ? 'var(--up-subtle)' : 'var(--down-subtle)',
                color: change >= 0 ? 'var(--up)' : 'var(--down)',
              }}
            >
              {change >= 0 ? '+' : ''}{change.toFixed(1)}%
            </span>
          )}
        </div>
      </div>

      {subvalue && (
        <p className="text-[10px] font-data text-paper-muted">{subvalue}</p>
      )}
    </div>
  );
}
