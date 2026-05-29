import React from 'react';

const LABELS = {
  SHORT: { name: 'Short', sub: '1mo · 7d forecast' },
  MID: { name: 'Mid', sub: '6mo · 30d forecast' },
  LONG: { name: 'Long', sub: '2y · 90d forecast' },
};

function HorizonTabs({ horizons, active, onChange }) {
  return (
    <div className="horizon-tabs" role="tablist">
      {horizons.map((h) => {
        const meta = LABELS[h] || { name: h, sub: '' };
        const isActive = h === active;
        return (
          <button
            key={h}
            type="button"
            role="tab"
            aria-selected={isActive}
            className={`horizon-tab${isActive ? ' horizon-tab--active' : ''}`}
            onClick={() => onChange(h)}
          >
            <span className="horizon-tab__name">{meta.name}</span>
            <span className="horizon-tab__sub">{meta.sub}</span>
          </button>
        );
      })}
    </div>
  );
}

export default HorizonTabs;
