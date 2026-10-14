import React, { useEffect, useState } from 'react';
import GlassPanel from '../components/GlassPanel';
import { getStats } from '../api/client';
import { 
  ArrowRight, 
  Search, 
  ShieldCheck, 
  Sparkles,
  Cpu,
  Sliders,
  Scale,
  Database,
  Activity
} from 'lucide-react';

export default function OverviewPage({ onNavigate }) {
  const [stats, setStats] = useState(null);

  useEffect(() => {
    getStats().then(setStats).catch(() => {});
  }, []);

  const totalImages = stats ? stats.total_images.toLocaleString() : '10,015';
  const refImages = stats ? stats.reference_images.toLocaleString() : '6,981';

  return (
    <div className="animate-fade-up">
      
      {/* Hero Section */}
      <section className="overview-hero">
        <img className="hero-background" src="/images/dermlens-lab.png" alt="" fetchPriority="high" width="1678" height="937" />
        <div className="hero-content">
        
        <div className="hero-eyebrow">
          Dermatology, in focus.
        </div>

        <h1>
          Content-Based Image Retrieval for Dermatology
        </h1>

        <p className="hero-description">
          Explore similar dermoscopic cases through image features, clinical context, and the HAM10000 reference collection.
        </p>

        <div className="hero-actions">
          <button 
            onClick={() => onNavigate('analyze')} 
            className="btn btn-liquid-white btn-lg"
          >
            <span>Upload & Analyze Query</span>
            <ArrowRight size={18} />
          </button>
          
          <button 
            onClick={() => onNavigate('dashboard')} 
            className="btn btn-liquid-glass btn-lg"
          >
            <span>Explore Validation Metrics</span>
          </button>
        </div>
        </div>
      </section>

      <section className="overview-stats" aria-label="Retrieval at a glance">
        <div className="stats-grid">
          
          <GlassPanel elevated interactive style={{ padding: '28px', textAlign: 'left' }}>
            <div style={{ color: 'var(--text-tertiary)', fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: '600', marginBottom: '8px' }}>
              Reference Knowledge Base
            </div>
            <div style={{ fontSize: '2.6rem', fontWeight: '700', letterSpacing: '-0.03em', color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
              {refImages}
            </div>
            <div style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', marginTop: '6px' }}>
              Biopsy-verified training cases
            </div>
          </GlassPanel>

          <GlassPanel elevated interactive style={{ padding: '28px', textAlign: 'left' }}>
            <div style={{ color: 'var(--text-tertiary)', fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: '600', marginBottom: '8px' }}>
              Feature Embedding
            </div>
            <div style={{ fontSize: '2.6rem', fontWeight: '700', letterSpacing: '-0.03em', color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
              2,048-D
            </div>
            <div style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', marginTop: '6px' }}>
              L2-normalized ResNet-50 features
            </div>
          </GlassPanel>

          <GlassPanel elevated interactive style={{ padding: '28px', textAlign: 'left' }}>
            <div style={{ color: 'var(--text-tertiary)', fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: '600', marginBottom: '8px' }}>
              Top-10 Ranking NDCG
            </div>
            <div style={{ fontSize: '2.6rem', fontWeight: '700', letterSpacing: '-0.03em', color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
              80.1%
            </div>
            <div style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', marginTop: '6px' }}>
              Combined visual & metadata fusion
            </div>
          </GlassPanel>

          <GlassPanel elevated interactive style={{ padding: '28px', textAlign: 'left' }}>
            <div style={{ color: 'var(--text-tertiary)', fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: '600', marginBottom: '8px' }}>
              Rare-Class Gain (df)
            </div>
            <div style={{ fontSize: '2.6rem', fontWeight: '700', letterSpacing: '-0.03em', color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
              +109.6%
            </div>
            <div style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', marginTop: '6px' }}>
              NDCG jumped from 0.252 &rarr; 0.529
            </div>
          </GlassPanel>

        </div>

      </section>

      {/* End-to-End Pipeline Walkthrough */}
      <section style={{ marginTop: '24px', marginBottom: '72px' }}>
        <div style={{ textAlign: 'center', marginBottom: '40px' }}>
          <div className="liquid-pill" style={{ marginBottom: '14px' }}>
            Multi-Stage Architecture
          </div>
          <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '2.6rem', fontWeight: '400', letterSpacing: '-0.01em', marginBottom: '10px' }}>
            Transparent Retrieval Pipeline
          </h2>
          <p style={{ color: 'var(--text-secondary)', maxWidth: '640px', margin: '0 auto', fontSize: '0.96rem' }}>
            Surfaces verified past patients with histological documentation to assist differential diagnosis.
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '24px' }}>
          
          <GlassPanel interactive style={{ padding: '30px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '18px' }}>
              <span className="badge badge-benign" style={{ fontFamily: 'var(--font-mono)' }}>STAGE 01</span>
              <Cpu size={22} color="var(--text-primary)" />
            </div>
            <h3 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '10px' }}>Deep Feature Extraction</h3>
            <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
              Query photo is projected through ResNet-50 (fc &rarr; Identity) and L2-normalized into a 2,048-D unit vector capturing lesion pigment networks.
            </p>
          </GlassPanel>

          <GlassPanel interactive style={{ padding: '30px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '18px' }}>
              <span className="badge badge-benign" style={{ fontFamily: 'var(--font-mono)' }}>STAGE 02</span>
              <Search size={22} color="var(--text-primary)" />
            </div>
            <h3 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '10px' }}>FAISS Vector Search</h3>
            <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
              Exact inner product cosine matching against 6,981 reference cases retrieves the Top-50 candidates in sub-millisecond latency.
            </p>
          </GlassPanel>

          <GlassPanel interactive style={{ padding: '30px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '18px' }}>
              <span className="badge badge-benign" style={{ fontFamily: 'var(--font-mono)' }}>STAGE 03</span>
              <Sliders size={22} color="var(--text-primary)" />
            </div>
            <h3 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '10px' }}>Clinical Metadata Fusion</h3>
            <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
              Patient demographic concordance (Age delta, Sex match, and Anatomical site) is weighted into a composite concordance score.
            </p>
          </GlassPanel>

          <GlassPanel interactive style={{ padding: '30px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '18px' }}>
              <span className="badge badge-benign" style={{ fontFamily: 'var(--font-mono)' }}>STAGE 04</span>
              <Scale size={22} color="var(--text-primary)" />
            </div>
            <h3 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '10px' }}>Similarity-Gated Rarity</h3>
            <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
              Inverse frequency rarity boost is strictly gated on visual plausibility (&ge; 0.80), elevating rare malignancies without spurious matching.
            </p>
          </GlassPanel>

        </div>
      </section>

      {/* 7 Supported Diagnoses Catalog */}
      <section style={{ marginBottom: '64px' }}>
        <div style={{ textAlign: 'center', marginBottom: '36px' }}>
          <div className="liquid-pill" style={{ marginBottom: '14px' }}>
            HAM10000 Benchmark
          </div>
          <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '2.6rem', fontWeight: '400', letterSpacing: '-0.01em', marginBottom: '8px' }}>
            7 Diagnostic Classes Catalog
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.94rem' }}>
            Full clinical coverage across malignant, pre-malignant, and benign lesion entities.
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px' }}>
          
          <GlassPanel style={{ padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <h4 style={{ fontSize: '1.1rem', fontWeight: '600' }}>Melanoma</h4>
              <span className="badge badge-malignant">mel &bull; Malignant</span>
            </div>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
              Invasive malignancy of melanocytes. Critical early detection target with asymmetric pigment networks.
            </p>
          </GlassPanel>

          <GlassPanel style={{ padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <h4 style={{ fontSize: '1.1rem', fontWeight: '600' }}>Basal Cell Carcinoma</h4>
              <span className="badge badge-malignant">bcc &bull; Malignant</span>
            </div>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
              Most frequent non-melanoma skin cancer. Translucent papules with prominent arborizing telangiectasia.
            </p>
          </GlassPanel>

          <GlassPanel style={{ padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <h4 style={{ fontSize: '1.1rem', fontWeight: '600' }}>Actinic Keratoses</h4>
              <span className="badge badge-premalignant">akiec &bull; Pre-malignant</span>
            </div>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
              Intraepithelial carcinoma / Bowen's disease. Scaly erythematous plaques from chronic UV radiation damage.
            </p>
          </GlassPanel>

          <GlassPanel style={{ padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <h4 style={{ fontSize: '1.1rem', fontWeight: '600' }}>Melanocytic Nevi</h4>
              <span className="badge badge-benign">nv &bull; Benign</span>
            </div>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
              Common benign moles. Comprises ~67% of typical dermatology datasets, creating majority-class baseline bias.
            </p>
          </GlassPanel>

          <GlassPanel style={{ padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <h4 style={{ fontSize: '1.1rem', fontWeight: '600' }}>Benign Keratosis</h4>
              <span className="badge badge-benign">bkl &bull; Benign</span>
            </div>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
              Seborrheic keratoses, solar lentigines, and lichen planus-like keratoses with sharp, regular borders.
            </p>
          </GlassPanel>

          <GlassPanel style={{ padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <h4 style={{ fontSize: '1.1rem', fontWeight: '600' }}>Dermatofibroma</h4>
              <span className="badge badge-benign">df &bull; Benign (Rare)</span>
            </div>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
              Benign dermal histiocytoma. Rare class (1.15% frequency) boosted effectively by gated calibration.
            </p>
          </GlassPanel>

        </div>
      </section>

    </div>
  );
}
