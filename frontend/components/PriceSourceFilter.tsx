'use client';

interface PriceSourceFilterProps {
  selectedSources: string[];
  onSourceChange: (sources: string[]) => void;
  availableSources?: string[];
}

const SOURCE_META: Record<string, string> = {
  aggregator_sync: 'Live',
  market_csgo: 'Market.CSGO',
  steam_historical: 'Steam (weekly)',
  steam_batch: 'Steam',
  steam: 'Steam',
  csfloat: 'CSFloat',
};

const DEFAULT_SOURCES = ['aggregator_sync'];

export default function PriceSourceFilter({
  selectedSources,
  onSourceChange,
  availableSources = DEFAULT_SOURCES,
}: PriceSourceFilterProps) {
  const handleToggle = (sourceId: string) => {
    if (selectedSources.includes(sourceId)) {
      onSourceChange(selectedSources.filter(s => s !== sourceId));
    } else {
      onSourceChange([...selectedSources, sourceId]);
    }
  };

  return (
    <div className="flex flex-col gap-2">
      <span className="specimen-tag text-paper-tertiary">Sources</span>
      <div className="flex gap-1.5 flex-wrap">
        {availableSources.map(sourceId => {
          const isActive = selectedSources.includes(sourceId);
          return (
            <button
              key={sourceId}
              onClick={() => handleToggle(sourceId)}
              className={`wear-pill text-[11px] font-semibold uppercase tracking-[0.12em] ${
                isActive ? 'wear-pill-active' : ''
              }`}
            >
              {SOURCE_META[sourceId] ?? sourceId}
            </button>
          );
        })}
      </div>
    </div>
  );
}
