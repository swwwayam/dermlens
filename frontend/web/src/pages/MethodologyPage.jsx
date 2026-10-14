import React from 'react';
import GlassPanel from '../components/GlassPanel';
import { BookOpen, ShieldCheck, Binary, Sliders, Scale, CheckCircle2 } from 'lucide-react';

export default function MethodologyPage() {
  return (
    <div className="animate-fade-up" style={{ maxWidth: '980px', margin: '0 auto' }}>
      
      {/* Header */}
      <div style={{ marginBottom: '36px' }}>
        <div className="liquid-pill" style={{ marginBottom: '14px' }}>
          <span className="liquid-pill-dot"></span>
          Clinical Decision Support Framework
        </div>
        <h1 style={{ fontFamily: 'var(--font-serif)', fontSize: '3rem', fontWeight: '400', letterSpacing: '-0.02em', marginBottom: '8px' }}>
          Methodology & Mathematical Rigor
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.96rem' }}>
          A principled case-based retrieval formulation solving black-box opacity and extreme class imbalance in dermatological AI.
        </p>
      </div>

      {/* Philosophy */}
      <GlassPanel elevated style={{ padding: '34px', marginBottom: '34px' }}>
        <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.8rem', fontWeight: '400', marginBottom: '12px', color: 'var(--text-primary)' }}>
          Evidence-Based Retrieval vs. Black-Box Softmax
        </h2>
        <p style={{ color: 'var(--text-secondary)', lineHeight: '1.65', marginBottom: '16px', fontSize: '0.95rem' }}>
          Standard convolutional networks output ungrounded probability distributions without clinical justification. In dermatological practice, clinicians rely on differential case comparison—comparing visual morphologies and patient contexts against verified historical encounters.
        </p>
        <p style={{ color: 'var(--text-secondary)', lineHeight: '1.65', fontSize: '0.95rem' }}>
          DermLens surfaces concrete reference cases with biopsy-verified histology (<code style={{ fontFamily: 'var(--font-mono)' }}>histo</code>) alongside full patient metadata to enable transparent, clinician-in-the-loop diagnostic auditing.
        </p>
      </GlassPanel>

      {/* 4 Pipeline Stages */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '24px', marginBottom: '40px' }}>
        
        <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.8rem', fontWeight: '400' }}>Mathematical Scoring Pipeline</h2>

        {/* Stage 1 */}
        <GlassPanel style={{ padding: '28px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '10px' }}>
            <span className="badge badge-benign" style={{ fontFamily: 'var(--font-mono)' }}>EQUATION 01</span>
            <h3 style={{ fontSize: '1.15rem', fontWeight: '600' }}>ResNet-50 2,048-D L2 Feature Extraction</h3>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: '1.6' }}>
            The dermoscopy query is preprocessed (224 &times; 224, ImageNet normalization) and projected through ResNet-50 (fc &rarr; Identity). Embeddings are unit-normalized so inner products directly compute cosine similarity:
          </p>
          <div style={{
            background: 'rgba(255, 255, 255, 0.06)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            padding: '14px 18px',
            borderRadius: 'var(--radius-sm)',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.88rem',
            color: 'var(--text-primary)',
            marginTop: '12px'
          }}>
            e_norm = e / ||e||_2 &isin; &real;^2048 &rArr; &lang;e_q, e_c&rang; = cos_sim(e_q, e_c)
          </div>
        </GlassPanel>

        {/* Stage 2 */}
        <GlassPanel style={{ padding: '28px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '10px' }}>
            <span className="badge badge-benign" style={{ fontFamily: 'var(--font-mono)' }}>EQUATION 02</span>
            <h3 style={{ fontSize: '1.15rem', fontWeight: '600' }}>FAISS IndexFlatIP Candidate Retrieval</h3>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: '1.6' }}>
            A flat inner-product vector index indexed with all 6,981 training images retrieves the Top-50 candidate pool (K<sub>cand</sub> = 50) in sub-millisecond latency.
          </p>
        </GlassPanel>

        {/* Stage 3 */}
        <GlassPanel style={{ padding: '28px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '10px' }}>
            <span className="badge badge-benign" style={{ fontFamily: 'var(--font-mono)' }}>EQUATION 03</span>
            <h3 style={{ fontSize: '1.15rem', fontWeight: '600' }}>Multimodal Clinical Metadata Fusion</h3>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: '1.6' }}>
            Patient demographics (Age delta linearly scaled over 50 years, biological Sex match, and Anatomical Localization match) contribute to clinical concordance:
          </p>
          <div style={{
            background: 'rgba(255, 255, 255, 0.06)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            padding: '14px 18px',
            borderRadius: 'var(--radius-sm)',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.88rem',
            color: 'var(--text-primary)',
            marginTop: '12px',
            lineHeight: '1.6'
          }}>
            Score_meta = 0.03 &times; max(0, 1 - |Age_q - Age_c| / 50) + 0.01 &times; &Iopf;(Sex_q == Sex_c) + 0.03 &times; &Iopf;(Loc_q == Loc_c)
          </div>
        </GlassPanel>

        {/* Stage 4 */}
        <GlassPanel style={{ padding: '28px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '10px' }}>
            <span className="badge badge-benign" style={{ fontFamily: 'var(--font-mono)' }}>EQUATION 04</span>
            <h3 style={{ fontSize: '1.15rem', fontWeight: '600' }}>Similarity-Gated Rarity Re-ranking</h3>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: '1.6' }}>
            To mitigate severe class imbalance (~67% nevi), an inverse-square-root frequency rarity adjustment is strictly gated on visual plausibility (&ge; 0.80):
          </p>
          <div style={{
            background: 'rgba(255, 255, 255, 0.06)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            padding: '14px 18px',
            borderRadius: 'var(--radius-sm)',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.88rem',
            color: 'var(--text-primary)',
            marginTop: '12px',
            lineHeight: '1.6'
          }}>
            Final Score = Visual_Sim + Score_meta + &Iopf;(Visual_Sim &ge; 0.80) &times; (0.03 &times; rarity_norm(c))
          </div>
        </GlassPanel>

      </div>

      {/* Scientific Rigor & Skin-Tone Audit */}
      <GlassPanel style={{ padding: '34px', marginBottom: '48px' }}>
        <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.8rem', fontWeight: '400', marginBottom: '16px' }}>Data Integrity & Fairness Auditing</h2>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '24px' }}>
          <div>
            <h4 style={{ fontSize: '1.05rem', fontWeight: '600', marginBottom: '8px', color: 'var(--text-primary)' }}>
              Lesion-Level Stratified Splits
            </h4>
            <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
              Partitioned strictly by <code style={{ fontFamily: 'var(--font-mono)' }}>lesion_id</code> to guarantee 0% data leakage across Train (70%), Validation (15%), and Test (15%) splits.
            </p>
          </div>

          <div>
            <h4 style={{ fontSize: '1.05rem', fontWeight: '600', marginBottom: '8px', color: 'var(--text-primary)' }}>
              Individual Typology Angle (ITA)
            </h4>
            <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
              Background peri-lesional skin pigmentation evaluated in CIELAB space (<code style={{ fontFamily: 'var(--font-mono)' }}>ITA = arctan((L* - 50) / b*) &times; 180 / &pi;</code>) across tertiles to audit and prevent algorithmic bias.
            </p>
          </div>
        </div>
      </GlassPanel>

    </div>
  );
}
