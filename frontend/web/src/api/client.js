/**
 * DermLens API Client
 * Connects to FastAPI backend at port 8000 or relative proxy
 */

const API_BASE = (import.meta.env?.VITE_API_URL || '').trim().replace(/\/+$/, '');

export async function analyzeLesion(formData) {
  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Analysis failed' }));
    throw new Error(err.detail || 'Inference error');
  }
  return res.json();
}

export async function getStats() {
  const res = await fetch(`${API_BASE}/api/stats`);
  if (!res.ok) throw new Error('Failed to fetch dataset statistics');
  return res.json();
}

export async function getMetrics() {
  const res = await fetch(`${API_BASE}/api/metrics`);
  if (!res.ok) throw new Error('Failed to fetch research metrics');
  return res.json();
}

export function getImageUrl(pathOrId) {
  if (!pathOrId) return '';
  if (pathOrId.startsWith('http')) return pathOrId;
  if (pathOrId.startsWith('/api/')) return `${API_BASE}${pathOrId}`;
  return `${API_BASE}/api/image/${pathOrId}`;
}
