// Shared BUY/HOLD/SELL helpers. The debate's consensus_summary embeds the
// recommendation token as free text; we parse it in priority order — explicit
// "SELL" wins over "BUY" if both appear, because recommendations are more often
// negated ("avoid a BUY") than promoted.
export function pickRecommendation(text) {
  if (!text || typeof text !== 'string') return null;
  const upper = text.toUpperCase();
  if (/\bSELL\b/.test(upper)) return 'SELL';
  if (/\bBUY\b/.test(upper)) return 'BUY';
  if (/\bHOLD\b/.test(upper)) return 'HOLD';
  return null;
}

export function recTagClass(rec) {
  if (rec === 'BUY') return 'tag tag--positive';
  if (rec === 'SELL') return 'tag tag--negative';
  if (rec === 'HOLD') return 'tag tag--neutral';
  return 'tag';
}

// Tone token ('positive'|'negative'|'neutral') for non-tag styling (e.g. the hero).
export function recTone(rec) {
  if (rec === 'BUY') return 'positive';
  if (rec === 'SELL') return 'negative';
  if (rec === 'HOLD') return 'neutral';
  return null;
}
