import React from 'react';

/**
 * Tiny inline SVG sparkline — no chart library.
 * `values` is a list of numbers; nulls/undefined are skipped from the path
 * but reserve x-positions so points align across siblings.
 */
function Sparkline({
  values,
  width = 120,
  height = 32,
  stroke = 'var(--brand, #4f46e5)',
  fill = 'rgba(79, 70, 229, 0.12)',
}) {
  const clean = (values || []).map((v) => (typeof v === 'number' && Number.isFinite(v) ? v : null));
  const nums = clean.filter((v) => v !== null);

  if (nums.length === 0) {
    return (
      <svg width={width} height={height} className="sparkline sparkline--empty" aria-hidden="true">
        <line x1="0" y1={height - 1} x2={width} y2={height - 1}
              stroke="var(--border, #e2e8f0)" strokeDasharray="2 3" />
      </svg>
    );
  }

  const min = Math.min(...nums);
  const max = Math.max(...nums);
  const span = max - min || 1;
  const dx = clean.length > 1 ? width / (clean.length - 1) : 0;
  const points = [];
  clean.forEach((v, i) => {
    if (v === null) return;
    const x = i * dx;
    const y = height - 1 - ((v - min) / span) * (height - 2);
    points.push([x, y]);
  });

  const linePath = points.map(([x, y], i) => `${i === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`).join(' ');
  const areaPath = `${linePath} L ${points[points.length - 1][0].toFixed(2)} ${height} L ${points[0][0].toFixed(2)} ${height} Z`;
  const last = points[points.length - 1];

  return (
    <svg width={width} height={height} className="sparkline" aria-hidden="true">
      <path d={areaPath} fill={fill} stroke="none" />
      <path d={linePath} fill="none" stroke={stroke} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={last[0]} cy={last[1]} r="2.4" fill={stroke} />
    </svg>
  );
}

export default Sparkline;
