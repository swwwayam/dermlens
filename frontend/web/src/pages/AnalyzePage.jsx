import React, { useState, useRef } from 'react';
import GlassPanel from '../components/GlassPanel';
import { analyzeLesion, getImageUrl } from '../api/client';
import { 
  Upload, 
  Sparkles, 
  User, 
  MapPin, 
  Clock, 
  Layers, 
  AlertCircle, 
  CheckCircle2, 
  Sliders, 
  FileText,
  Zap,
  Info
} from 'lucide-react';

export default function AnalyzePage() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [age, setAge] = useState('');
  const [sex, setSex] = useState('');
  const [localization, setLocalization] = useState('');
  const [topK, setTopK] = useState(5);

  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState(0);
  const [error, setError] = useState('');
  const [results, setResults] = useState(null);

  const fileInputRef = useRef(null);

  const handleFile = (f) => {
    if (!f || !f.type.startsWith('image/')) {
      alert('Please upload a valid image file (JPEG or PNG).');
      return;
    }
    setFile(f);
    const url = URL.createObjectURL(f);
    setPreviewUrl(url);
    setError('');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    setError('');
    setStage(1);

    const timer1 = setTimeout(() => setStage(2), 300);
    const timer2 = setTimeout(() => setStage(3), 600);
    const timer3 = setTimeout(() => setStage(4), 900);

    const formData = new FormData();
    formData.append('image', file);
    if (age) formData.append('age', age);
    if (sex) formData.append('sex', sex);
    if (localization) formData.append('localization', localization);
    formData.append('top_k', topK);

    try {
      const data = await analyzeLesion(formData);
      setStage(5);
      setResults(data);
    } catch (err) {
      console.error(err);
      setError(err.message || 'Retrieval failed. Ensure FastAPI server is running on port 8000.');
    } finally {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      setLoading(false);
    }
  };

  const formatScore = (val) => (val !== undefined && val !== null ? Number(val).toFixed(4) : '0.0000');

  return (
    <div className="animate-fade-up">
      
      {/* Header */}
      <div style={{ marginBottom: '32px' }}>
        <div className="liquid-pill" style={{ marginBottom: '14px' }}>
          <span className="liquid-pill-dot"></span>
          Direct GPU Vector Retrieval
        </div>
        <h1 style={{ fontFamily: 'var(--font-serif)', fontSize: '3rem', fontWeight: '400', letterSpacing: '-0.02em', marginBottom: '8px' }}>
          Dermatology Case Retrieval
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.96rem' }}>
          Upload a query lesion to extract 2,048-D ResNet-50 feature vectors, query the FAISS index, and perform multimodal metadata fusion.
        </p>
      </div>

      {/* Main Analysis Layout */}
      <div className="analysis-layout">
        
        {/* Left: Query Input Panel */}
        <GlassPanel elevated style={{ padding: '26px' }}>
          
          <h3 style={{ fontSize: '1.15rem', fontWeight: '600', marginBottom: '18px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Upload size={18} color="var(--text-primary)" />
            <span>Query Dermoscopy</span>
          </h3>

          {/* Upload Dropzone */}
          {!previewUrl ? (
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              style={{
                border: '1.5px dashed rgba(255, 255, 255, 0.35)',
                background: 'rgba(255, 255, 255, 0.05)',
                backdropFilter: 'var(--blur-glass)',
                borderRadius: 'var(--radius-md)',
                padding: '38px 20px',
                textAlign: 'center',
                cursor: 'pointer',
                transition: 'all 0.25s ease',
              }}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png,image/jpg"
                style={{ display: 'none' }}
                onChange={(e) => e.target.files && handleFile(e.target.files[0])}
              />
              <Upload size={34} color="var(--text-primary)" style={{ margin: '0 auto 12px', opacity: 0.8 }} />
              <div style={{ fontSize: '0.94rem', fontWeight: '600', color: 'var(--text-primary)', marginBottom: '4px' }}>
                Select a dermoscopy photo
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-tertiary)' }}>
                Drag & drop or click to browse (JPG, PNG)
              </div>
            </div>
          ) : (
            <div>
              <div style={{
                width: '100%',
                height: '230px',
                borderRadius: 'var(--radius-md)',
                overflow: 'hidden',
                background: 'rgba(0, 0, 0, 0.4)',
                border: '1px solid rgba(255, 255, 255, 0.3)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                position: 'relative',
                boxShadow: 'inset 0 0 20px rgba(0,0,0,0.5)'
              }}>
                <img src={previewUrl} alt="Query Lesion Preview" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                <span className="badge badge-benign" style={{ position: 'absolute', top: '10px', left: '10px', backdropFilter: 'blur(8px)' }}>
                  Query Image
                </span>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '12px' }}>
                <span style={{ fontSize: '0.8rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                  {file?.name}
                </span>
                <button 
                  onClick={() => { setFile(null); setPreviewUrl(''); setResults(null); }}
                  className="btn btn-liquid-glass btn-sm" 
                  style={{ fontSize: '0.75rem' }}
                >
                  Change Image
                </button>
              </div>
            </div>
          )}

          {/* Clinical Metadata Inputs */}
          <div style={{ marginTop: '26px', paddingTop: '18px', borderTop: '1px solid rgba(255, 255, 255, 0.15)' }}>
            <div style={{ fontSize: '0.82rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-tertiary)', marginBottom: '14px' }}>
              Clinical Metadata (Optional)
            </div>

            <div style={{ marginBottom: '14px' }}>
              <label style={{ display: 'block', fontSize: '0.84rem', color: 'var(--text-secondary)', marginBottom: '6px' }}>
                Patient Age
              </label>
              <input
                type="number"
                placeholder="e.g. 55"
                min="0"
                max="100"
                value={age}
                onChange={(e) => setAge(e.target.value)}
                className="form-input"
              />
            </div>

            <div style={{ marginBottom: '14px' }}>
              <label style={{ display: 'block', fontSize: '0.84rem', color: 'var(--text-secondary)', marginBottom: '6px' }}>
                Biological Sex
              </label>
              <select value={sex} onChange={(e) => setSex(e.target.value)} className="form-select">
                <option value="">-- Not specified --</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
              </select>
            </div>

            <div style={{ marginBottom: '14px' }}>
              <label style={{ display: 'block', fontSize: '0.84rem', color: 'var(--text-secondary)', marginBottom: '6px' }}>
                Anatomical Localization
              </label>
              <select value={localization} onChange={(e) => setLocalization(e.target.value)} className="form-select">
                <option value="">-- Not specified --</option>
                <option value="back">Back</option>
                <option value="lower extremity">Lower Extremity</option>
                <option value="trunk">Trunk</option>
                <option value="upper extremity">Upper Extremity</option>
                <option value="abdomen">Abdomen</option>
                <option value="face">Face</option>
                <option value="chest">Chest</option>
                <option value="foot">Foot</option>
                <option value="neck">Neck</option>
                <option value="scalp">Scalp</option>
                <option value="hand">Hand</option>
                <option value="ear">Ear</option>
                <option value="genital">Genital</option>
                <option value="acral">Acral</option>
              </select>
            </div>

            <div style={{ marginBottom: '22px' }}>
              <label style={{ display: 'block', fontSize: '0.84rem', color: 'var(--text-secondary)', marginBottom: '6px' }}>
                Retrieval Matches (Top-K)
              </label>
              <select value={topK} onChange={(e) => setTopK(Number(e.target.value))} className="form-select">
                <option value={5}>Top 5 Nearest Cases</option>
                <option value={10}>Top 10 Nearest Cases</option>
                <option value={15}>Top 15 Nearest Cases</option>
              </select>
            </div>

            <button
              onClick={handleAnalyze}
              disabled={!file || loading}
              className="btn btn-liquid-white"
              style={{ width: '100%', opacity: (!file || loading) ? 0.6 : 1, cursor: (!file || loading) ? 'not-allowed' : 'pointer' }}
            >
              {loading ? (
                <>
                  <Zap size={16} className="animate-spin" />
                  <span>Processing Retrieval Pipeline...</span>
                </>
              ) : (
                <>
                  <Sparkles size={16} />
                  <span>Execute Retrieval Pipeline</span>
                </>
              )}
            </button>
          </div>

          {/* Pipeline Stages Feedback */}
          {loading && (
            <div style={{
              marginTop: '20px',
              padding: '18px',
              background: 'rgba(255, 255, 255, 0.08)',
              borderRadius: 'var(--radius-md)',
              border: '1px solid rgba(255, 255, 255, 0.3)',
              display: 'flex',
              flexDirection: 'column',
              gap: '12px',
              fontSize: '0.84rem',
            }}>
              <div style={{ fontWeight: '600', color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Pipeline Execution Status
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: stage >= 1 ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                <CheckCircle2 size={14} color={stage >= 1 ? 'var(--text-primary)' : 'currentColor'} />
                <span>ResNet-50 2048-D Embedding</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: stage >= 2 ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                <CheckCircle2 size={14} color={stage >= 2 ? 'var(--text-primary)' : 'currentColor'} />
                <span>FAISS IndexFlatIP Cosine Search</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: stage >= 3 ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                <CheckCircle2 size={14} color={stage >= 3 ? 'var(--text-primary)' : 'currentColor'} />
                <span>Clinical Metadata Concordance</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: stage >= 4 ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                <CheckCircle2 size={14} color={stage >= 4 ? 'var(--text-primary)' : 'currentColor'} />
                <span>Similarity-Gated Rarity Re-ranking</span>
              </div>
            </div>
          )}

          {error && (
            <div style={{
              marginTop: '16px',
              padding: '14px 18px',
              borderRadius: 'var(--radius-sm)',
              background: 'rgba(255, 75, 110, 0.2)',
              border: '1px solid rgba(255, 110, 140, 0.5)',
              color: 'var(--text-primary)',
              fontSize: '0.86rem',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
            }}>
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}

        </GlassPanel>

        {/* Right: Results Stream */}
        <div>
          
          {!results ? (
            <GlassPanel style={{ padding: '64px 32px', textAlign: 'center' }}>
              <Layers size={52} color="var(--text-primary)" style={{ margin: '0 auto 16px', opacity: 0.5 }} />
              <h3 style={{ fontSize: '1.3rem', fontWeight: '600', marginBottom: '8px' }}>
                No Query Executed
              </h3>
              <p style={{ color: 'var(--text-secondary)', maxWidth: '440px', margin: '0 auto', fontSize: '0.92rem', lineHeight: '1.6' }}>
                Select a skin lesion image on the left panel and click "Execute Retrieval Pipeline" to search the reference library.
              </p>
            </GlassPanel>
          ) : (
            <div>
              
              {/* Summary Bar */}
              <GlassPanel elevated style={{ padding: '18px 24px', marginBottom: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
                <div>
                  <div style={{ fontSize: '0.74rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-tertiary)' }}>
                    Query Complete
                  </div>
                  <div style={{ fontSize: '1.1rem', fontWeight: '600', color: 'var(--text-primary)' }}>
                    Retrieved {results.results.length} Verified Reference Cases
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '18px', fontFamily: 'var(--font-mono)', fontSize: '0.86rem', color: 'var(--text-secondary)' }}>
                  <div>Latency: <span style={{ color: 'var(--text-primary)', fontWeight: '600' }}>{results.processing_time_ms} ms</span></div>
                  <div>Device: <span style={{ color: 'var(--text-primary)' }}>{results.pipeline.device}</span></div>
                </div>
              </GlassPanel>

              {/* Case Cards List */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
                {results.results.map((item) => {
                  const isMalignant = item.diagnosis_category === 'Malignant';
                  const isPre = item.diagnosis_category === 'Pre-malignant';
                  const badgeClass = isMalignant ? 'badge-malignant' : (isPre ? 'badge-premalignant' : 'badge-benign');

                  return (
                    <GlassPanel key={item.rank} interactive style={{ padding: '22px', display: 'grid', gridTemplateColumns: '150px 1fr', gap: '22px', alignItems: 'center' }}>
                      
                      {/* Thumbnail with Rank Badge */}
                      <div style={{
                        width: '150px',
                        height: '150px',
                        borderRadius: 'var(--radius-md)',
                        overflow: 'hidden',
                        background: 'rgba(0, 0, 0, 0.4)',
                        border: '1px solid rgba(255, 255, 255, 0.25)',
                        position: 'relative',
                        boxShadow: '0 8px 20px rgba(0,0,0,0.3)'
                      }}>
                        <img 
                          src={getImageUrl(item.image_url)} 
                          alt={`Reference ${item.image_id}`} 
                          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                          loading="lazy"
                        />
                        <span style={{
                          position: 'absolute',
                          top: '8px',
                          left: '8px',
                          background: 'var(--image-label-bg)',
                          backdropFilter: 'blur(8px)',
                          border: '1px solid rgba(255, 255, 255, 0.4)',
                          color: 'var(--text-primary)',
                          fontSize: '0.78rem',
                          fontWeight: '700',
                          padding: '2px 8px',
                          borderRadius: '9999px',
                          fontFamily: 'var(--font-mono)'
                        }}>
                          #{item.rank}
                        </span>
                      </div>

                      {/* Details & Transparent Scores */}
                      <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                        
                        <div>
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px', marginBottom: '8px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                              <h3 style={{ fontSize: '1.25rem', fontWeight: '600', color: 'var(--text-primary)', margin: 0 }}>
                                {item.diagnosis_full}
                              </h3>
                              <span className={`badge ${badgeClass}`}>{item.diagnosis.toUpperCase()}</span>
                              <span className="badge badge-benign">{item.diagnosis_category}</span>
                            </div>
                            <span style={{ fontSize: '0.78rem', fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)' }}>
                              {item.image_id} / {item.lesion_id}
                            </span>
                          </div>

                          <div style={{ display: 'flex', gap: '14px', fontSize: '0.86rem', color: 'var(--text-secondary)', marginBottom: '16px', flexWrap: 'wrap' }}>
                            <span><strong>Age:</strong> {item.age !== null ? `${item.age} yrs` : 'N/A'}</span>
                            <span>&bull;</span>
                            <span><strong>Sex:</strong> {item.sex ? item.sex.toUpperCase() : 'N/A'}</span>
                            <span>&bull;</span>
                            <span><strong>Site:</strong> {item.localization || 'N/A'}</span>
                            <span>&bull;</span>
                            <span><strong>Verification:</strong> {item.diagnosis_type || 'histo'}</span>
                          </div>
                        </div>

                        {/* Transparent Sub-Score Breakdown */}
                        <div style={{
                          display: 'grid',
                          gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))',
                          gap: '10px',
                          background: 'rgba(255, 255, 255, 0.06)',
                          border: '1px solid rgba(255, 255, 255, 0.16)',
                          borderRadius: 'var(--radius-sm)',
                          padding: '12px 16px',
                          backdropFilter: 'var(--blur-glass)'
                        }}>
                          <div>
                            <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-tertiary)' }}>
                              Final Score
                            </div>
                            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.05rem', fontWeight: '700', color: 'var(--text-primary)' }}>
                              {formatScore(item.scores.final_score)}
                            </div>
                          </div>

                          <div>
                            <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-tertiary)' }}>
                              Visual Cosine
                            </div>
                            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.94rem', fontWeight: '600', color: 'var(--text-secondary)' }}>
                              {formatScore(item.scores.visual_similarity)}
                            </div>
                          </div>

                          <div>
                            <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-tertiary)' }}>
                              Metadata Match
                            </div>
                            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.94rem', fontWeight: '600', color: 'var(--text-secondary)' }}>
                              +{formatScore(item.scores.metadata_score)}
                            </div>
                          </div>

                          <div>
                            <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-tertiary)' }}>
                              Gated Rarity
                            </div>
                            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.94rem', fontWeight: '600', color: 'var(--text-secondary)' }}>
                              +{formatScore(item.scores.rarity_adjustment)}
                            </div>
                          </div>
                        </div>

                      </div>

                    </GlassPanel>
                  );
                })}
              </div>

            </div>
          )}

        </div>

      </div>

    </div>
  );
}
