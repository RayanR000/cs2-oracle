'use client';

import Link from 'next/link';

interface ItemCardProps {
  itemId: string;
  name: string;
  type: string;
  imageUrl?: string;
  currentPrice?: number;
  priceChange7d?: number;
  annotation?: string;
  pinned?: boolean;
}

export default function ItemCard({
  itemId,
  name,
  type,
  imageUrl,
  currentPrice,
  priceChange7d,
  annotation,
  pinned = false,
}: ItemCardProps) {
  return (
    <Link
      href={`/items/${itemId}`}
      className="relative block bg-stock border border-border rounded-sm overflow-hidden transition-colors duration-250 hover:border-border-accent hover:bg-surface group"
    >
      {pinned && <span aria-hidden className="sounding-rule" />}
      <span aria-hidden className="card-sweep" />

      <div className="px-4 pt-4">
        <div className="flex items-baseline justify-between gap-2 mb-2">
          <span className="dive-tag text-paper-tertiary">{type}</span>
          {annotation && (
            <span className="dive-tag text-paper-muted">{annotation}</span>
          )}
        </div>
        <h3 className="text-[13px] font-semibold text-paper tracking-tight line-clamp-1 mb-3">
          {name}
        </h3>
      </div>

      <div className="aspect-square mx-4 mb-4 bg-recess rounded-xs flex items-center justify-center p-6 overflow-hidden">
        {imageUrl ? (
          <img
            src={imageUrl}
            alt={name}
            loading="lazy"
            className="max-w-full max-h-full object-contain transition-transform duration-500 ease-out group-hover:scale-[1.03]"
          />
        ) : (
          <div className="dive-tag text-paper-muted">No Plate</div>
        )}
      </div>

      <div className="px-4 pb-4 flex items-end justify-between gap-3">
        <p className="text-data-lg text-paper">
          {currentPrice !== undefined ? `$${currentPrice.toFixed(2)}` : '\u2014'}
        </p>
        {priceChange7d !== undefined && (
          <span
            className="inline-block px-1.5 py-0.5 rounded-xs font-data text-[11px] font-semibold"
            style={{
              backgroundColor: priceChange7d >= 0 ? 'var(--up-subtle)' : 'var(--down-subtle)',
              color: priceChange7d >= 0 ? 'var(--up)' : 'var(--down)',
            }}
          >
            {priceChange7d >= 0 ? '+' : ''}{priceChange7d.toFixed(1)}%
          </span>
        )}
      </div>
    </Link>
  );
}
