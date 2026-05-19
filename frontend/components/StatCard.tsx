'use client';

interface StatCardProps {
  label: string;
  value: string | number;
  highlight?: 'primary' | 'secondary' | 'warning';
  isPositive?: boolean;
  change?: number;
  subvalue?: string;
  icon?: React.ReactNode;
}

export default function StatCard({
  label,
  value,
  highlight = 'primary',
  isPositive,
  change,
  subvalue,
  icon
}: StatCardProps) {
  let accentColor = 'var(--accent-primary)';
  let accentBg = 'var(--data-up-subtle)';

  if (isPositive === true) {
    accentColor = 'var(--data-up)';
    accentBg = 'var(--data-up-subtle)';
  } else if (isPositive === false) {
    accentColor = 'var(--data-down)';
    accentBg = 'var(--data-down-subtle)';
  } else if (highlight === 'secondary') {
    accentColor = 'var(--accent-secondary)';
    accentBg = 'rgba(245, 158, 11, 0.08)';
  } else if (highlight === 'warning') {
    accentColor = 'var(--accent-warning)';
    accentBg = 'rgba(255, 107, 107, 0.08)';
  }

  return (
    <div
      style={{
        backgroundColor: accentBg,
        borderColor: accentColor,
        borderWidth: '1px',
        borderRadius: '10px',
        padding: '18px',
        position: 'relative',
        overflow: 'hidden'
      }}
      className="card-base group hover:border-opacity-100 transition-all duration-300"
    >
      {/* Top accent bar */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '3px',
          backgroundColor: accentColor,
          opacity: 0.6
        }}
      />

      {/* Content */}
      <div className="relative">
        <div className="flex items-start justify-between mb-2">
          <p className="text-xs font-medium uppercase tracking-wide" style={{ color: 'var(--text-tertiary)' }}>
            {label}
          </p>
          {icon && (
            <div style={{ color: accentColor, opacity: 0.6 }} className="text-lg">
              {icon}
            </div>
          )}
        </div>

        {/* Main value */}
        <p
          className="text-2xl font-mono font-bold mb-2"
          style={{ color: accentColor }}
        >
          {value}
        </p>

        {/* Sub-value and change indicator */}
        <div className="flex items-baseline gap-2">
          {subvalue && (
            <span className="text-xs font-mono" style={{ color: 'var(--text-tertiary)' }}>
              {subvalue}
            </span>
          )}
          {change !== undefined && (
            <span
              className="text-xs font-semibold"
              style={{
                color: change >= 0 ? 'var(--data-up)' : 'var(--data-down)'
              }}
            >
              {change >= 0 ? '↑' : '↓'} {Math.abs(change).toFixed(1)}%
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
