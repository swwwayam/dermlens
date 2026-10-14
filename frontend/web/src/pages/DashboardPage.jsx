import React, { useEffect, useState } from 'react';
import GlassPanel from '../components/GlassPanel';
import { getMetrics } from '../api/client';
import { BarChart2, TrendingUp, CheckCircle, ShieldAlert, Award, ArrowUpRight } from 'lucide-react';

const DX_NAMES = {
  'nv': 'Melanocytic Nevi',
  'mel': 'Melanoma',
  'bkl': 'Benign Keratosis',
  'bcc': 'Basal Cell Carcinoma',
  'akiec': 'Actinic Keratoses',
  'vasc': 'Vascular Lesions',
  'df': 'Dermatofibroma'
};

const DX_CATS = {
  'nv': 'Benign',
  'mel': 'Malignant',
  'bkl': 'Benign',
  'bcc': 'Malignant',
  'akiec': 'Pre-malignant',
  'vasc': 'Benign',
  'df': 'Benign'
};

export default function DashboardPage() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getMetrics()
      .then((data) => {
        setMetrics(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  const formatPct = (val) => {
    if (val === undefined || val === null) return '0.0%';
    const p = val > 1 ? val : val * 100;
    return `${Number(p).toFixed(1)}%`;
  };

  const formatScore = (val) => (val !== undefined && val !== null ? Number(val).toFixed(3) : '0.000');

  const baselineList = metrics?.baseline?.per_class || [];
  const combinedList = metrics?.combined?.per_class || [];
  const baseMap = {};
  baselineList.forEach((b) => { baseMap[b.query_dx] = b; });

  return (
    <div className="animate-fade-up">
      
      {/* Header */}
      <div style={{ marginBottom: '32px' }}>
        <div className="liquid-pill" style={{ marginBottom: '14px' }}>
          <span className="liquid-pill-dot"></span>
          Empirical Research Benchmark
        </div>
        <h1 style={{ fontFamily: 'var(--font-serif)', fontSize: '3rem', fontWeight: '400', letterSpacing: '-0.02em', marginBottom: '8px' }}>
          Quantitative Evaluation Dashboard
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.96rem' }}>
          Direct empirical findings across 1,502 stratified zero-leakage test queries in HAM10000.
        </p>
      </div>

      {/* Top Metrics Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))', gap: '20px', marginBottom: '36px' }}>
        
        <GlassPanel elevated interactive style={{ padding: '26px' }}>
          <div style={{ color: 'var(--text-tertiary)', fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: '600', marginBottom: '8px' }}>
            Combined Top-10 NDCG
          </div>
          <div style={{ fontSize: '2.6rem', fontWeight: '700', letterSpacing: '-0.03em', color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
            80.06%
          </div>
          <div style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', marginTop: '6px' }}>
            Overall ranking quality (1,502 queries)
          </div>
        </GlassPanel>

        <GlassPanel elevated interactive style={{ padding: '26px' }}>
          <div style={{ color: 'var(--text-tertiary)', fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: '600', marginBottom: '8px' }}>
            Macro MRR
          </div>
          <div style={{ fontSize: '2.6rem', fontWeight: '700', letterSpacing: '-0.03em', color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
            0.598
          </div>
          <div style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', marginTop: '6px' }}>
            Mean reciprocal rank of first true case
          </div>
        </GlassPanel>

        <GlassPanel elevated interactive style={{ padding: '26px' }}>
          <div style={{ color: 'var(--text-tertiary)', fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: '600', marginBottom: '8px' }}>
            Rare Dermatofibroma Gain
          </div>
          <div style={{ fontSize: '2.6rem', fontWeight: '700', letterSpacing: '-0.03em', color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
            +109.6%
          </div>
          <div style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', marginTop: '6px' }}>
            NDCG@10 jumped from 0.252 &rarr; 0.529
          </div>
        </GlassPanel>

        <GlassPanel elevated interactive style={{ padding: '26px' }}>
          <div style={{ color: 'var(--text-tertiary)', fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: '600', marginBottom: '8px' }}>
            Wilcoxon Paired Test
          </div>
          <div style={{ fontSize: '2.6rem', fontWeight: '700', letterSpacing: '-0.03em', color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
            p &lt; 10⁻⁶
          </div>
          <div style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', marginTop: '6px' }}>
            Statistically confirmed significant gain
          </div>
        </GlassPanel>

      </div>

      {/* Per-Class Table */}
      <GlassPanel style={{ padding: '30px', marginBottom: '36px' }}>
        
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '22px', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.8rem', fontWeight: '400' }}>Per-Class Retrieval Performance</h2>
            <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
              Visual Baseline vs Final Combined System (Visual + Clinical Metadata + Similarity-Gated Rarity)
            </p>
          </div>
          <span className="badge badge-benign" style={{ fontFamily: 'var(--font-mono)' }}>1,502 Queries</span>
        </div>

        <div className="glass-table-wrap">
          <table className="glass-table">
            <thead>
              <tr>
                <th>Diagnostic Class</th>
                <th>Category</th>
                <th>Queries (n)</th>
                <th>Baseline P@5</th>
                <th>Final P@5</th>
                <th>Baseline NDCG</th>
                <th>Final NDCG</th>
                <th>Final MRR</th>
              </tr>
            </thead>
            <tbody>
              {combinedList.length > 0 ? (
                combinedList.map((c) => {
                  const b = baseMap[c.dx] || {};
                  const cat = DX_CATS[c.dx] || 'Benign';
                  const isMalignant = cat === 'Malignant';
                  const isPre = cat === 'Pre-malignant';
                  const badgeClass = isMalignant ? 'badge-malignant' : (isPre ? 'badge-premalignant' : 'badge-benign');

                  return (
                    <tr key={c.dx}>
                      <td>
                        <div style={{ fontWeight: '600', color: 'var(--text-primary)' }}>{DX_NAMES[c.dx] || c.dx}</div>
                        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.74rem', color: 'var(--text-tertiary)' }}>{c.dx}</span>
                      </td>
                      <td><span className={`badge ${badgeClass}`}>{cat}</span></td>
                      <td style={{ fontFamily: 'var(--font-mono)' }}>{c.n_queries}</td>
                      <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)' }}>{formatPct(b.precision_at_5)}</td>
                      <td style={{ fontFamily: 'var(--font-mono)', fontWeight: '600', color: 'var(--text-primary)' }}>{formatPct(c.precision_at_5)}</td>
                      <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)' }}>{formatScore(b.ndcg_at_10)}</td>
                      <td style={{ fontFamily: 'var(--font-mono)', fontWeight: '600', color: 'var(--text-primary)' }}>{formatScore(c.ndcg_at_10)}</td>
                      <td style={{ fontFamily: 'var(--font-mono)' }}>{formatScore(c.mrr)}</td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan="8" style={{ textAlign: 'center', padding: '32px', color: 'var(--text-tertiary)' }}>
                    {loading ? 'Loading empirical metrics from backend...' : 'No metric tables found.'}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

      </GlassPanel>

      {/* Rarity Sweep & Significance Tests */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '24px', marginBottom: '40px' }}>
        
        {/* Rarity Sweep Table */}
        <GlassPanel style={{ padding: '30px' }}>
          <h3 style={{ fontSize: '1.2rem', fontWeight: '600', marginBottom: '8px' }}>
            Rarity Weight Ablation Study
          </h3>
          <p style={{ fontSize: '0.86rem', color: 'var(--text-secondary)', marginBottom: '18px' }}>
            Demonstrates the precision-fairness trade-off as rarity weighting factor <em>w</em> scales from 0.0 to 0.10.
          </p>

          <div className="glass-table-wrap">
            <table className="glass-table">
              <thead>
                <tr>
                  <th>Rarity Weight (w)</th>
                  <th>Overall NDCG</th>
                  <th>Macro NDCG</th>
                  <th>Macro MRR</th>
                </tr>
              </thead>
              <tbody>
                {(metrics?.ablation || []).map((row) => (
                  <tr key={row.rarity_weight}>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: '600' }}>w = {row.rarity_weight}</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{formatPct(row.ndcg_at_10)}</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: '600', color: 'var(--text-primary)' }}>{formatPct(row.macro_ndcg_at_10)}</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{formatPct(row.macro_mrr)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </GlassPanel>

        {/* Wilcoxon Significance Table */}
        <GlassPanel style={{ padding: '30px' }}>
          <h3 style={{ fontSize: '1.2rem', fontWeight: '600', marginBottom: '8px' }}>
            Wilcoxon Statistical Significance
          </h3>
          <p style={{ fontSize: '0.86rem', color: 'var(--text-secondary)', marginBottom: '18px' }}>
            Paired non-parametric test on n=1,502 query pairs with strict significance threshold &alpha; = 0.05.
          </p>

          <div className="glass-table-wrap">
            <table className="glass-table">
              <thead>
                <tr>
                  <th>Model Comparison</th>
                  <th>Mean NDCG Delta</th>
                  <th>p-value</th>
                  <th>Significance</th>
                </tr>
              </thead>
              <tbody>
                {(metrics?.significance || []).map((row, idx) => {
                  const isSig = row.significant === true || row.significant === 'True';
                  return (
                    <tr key={idx}>
                      <td style={{ fontSize: '0.84rem', fontWeight: '500' }}>{row.comparison}</td>
                      <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.82rem' }}>
                        {formatScore(row.baseline_mean)} &rarr; {formatScore(row.model_mean)}
                      </td>
                      <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.82rem' }}>
                        {Number(row.p_value).toExponential(2)}
                      </td>
                      <td>
                        <span className={`badge ${isSig ? 'badge-malignant' : 'badge-benign'}`}>
                          {isSig ? 'Significant' : 'Not Sig.'}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </GlassPanel>

      </div>

    </div>
  );
}
