import React, { useState } from 'react';

// Tooltip width + viewport margin must match .kpi-help::after in App.css.
const TIP_WIDTH = 240;
const EDGE_MARGIN = 16;

/**
 * One metric tile for the KPI grids (SnapshotView "At a glance", InsightsPanel).
 *
 * `help` renders a small "?" badge next to the label whose explanation shows
 * as a tooltip on hover — and on keyboard focus, so it isn't mouse-only. The
 * text is also exposed as the badge's aria-label for screen readers. The
 * tooltip is pure CSS (see .kpi-help in App.css), driven by the data-tip attr.
 *
 * The bubble is centered on the badge by default; on hover/focus we measure
 * whether that would spill past the viewport (badges in the first/last grid
 * column at tablet widths do) and flip it to hang from the badge's edge
 * instead — otherwise the text clips and the page gains horizontal scroll.
 */
function KpiCell({ label, value, tone, help }) {
  const [edge, setEdge] = useState(null); // null | 'left' | 'right'

  const measure = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const center = r.left + r.width / 2;
    // clientWidth, not innerWidth: the latter includes the vertical scrollbar.
    const vw = document.documentElement.clientWidth;
    if (center + TIP_WIDTH / 2 > vw - EDGE_MARGIN) setEdge('right');
    else if (center - TIP_WIDTH / 2 < EDGE_MARGIN) setEdge('left');
    else setEdge(null);
  };

  return (
    <div className={`kpi-cell${tone ? ` kpi-cell--${tone}` : ''}`}>
      <div className="kpi-cell__label">
        <span>{label}</span>
        {help && (
          <span
            className={`kpi-help${edge ? ` kpi-help--edge-${edge}` : ''}`}
            data-tip={help}
            tabIndex={0}
            role="img"
            aria-label={`${label}: ${help}`}
            onMouseEnter={measure}
            onFocus={measure}
          >
            ?
          </span>
        )}
      </div>
      <div className="kpi-cell__value">{value}</div>
    </div>
  );
}

export default KpiCell;
