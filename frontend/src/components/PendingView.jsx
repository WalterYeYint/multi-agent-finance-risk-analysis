import React, { useEffect, useState } from 'react';
import { Loader2 } from 'lucide-react';

const PHASE_COPY = {
  analyzing: 'Running the chain agents (data → sentiment ‖ valuation ‖ fundamental → risk → writer)…',
  debating: 'Running the round-robin debate to converge on a recommendation…',
  queued: 'Job is queued — the worker will pick it up shortly.',
  running: 'Pipeline is running.',
};

function PendingView({ pending }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const start = Date.now();
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 1000);
    return () => clearInterval(id);
  }, [pending?.job_id]);

  const status = pending?.status || 'queued';
  const phase = pending?.progress || status;
  const copy = PHASE_COPY[phase] || `Working… (${phase})`;

  return (
    <div className="pending-state">
      <div className="pending-state__spinner">
        <Loader2 size={28} className="spin" />
      </div>
      <div className="pending-state__title">Generating snapshot</div>
      <div className="pending-state__copy">{copy}</div>
      <div className="pending-state__meta">
        job #{pending?.job_id ?? '—'} · {elapsed}s elapsed · usually 3–6 minutes
      </div>
      <div className="pending-state__hint">
        This page polls every few seconds. You can leave the tab open or come back later.
      </div>
    </div>
  );
}

export default PendingView;
