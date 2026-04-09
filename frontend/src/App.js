import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import InputForm      from './components/InputForm';
import AQIGauge       from './components/AQIGauge';
import HealthAdvisory from './components/HealthAdvisory';
import ModelComparison from './components/ModelComparison';
import FuzzyPanel     from './components/FuzzyPanel';
import NNPanel        from './components/NNPanel';
import ANFISPanel     from './components/ANFISPanel';

const API = 'http://localhost:5000';

const TABS = [
  { id: 'predict',  label: 'Predict',       icon: '🎯', dot: '#58a6ff' },
  { id: 'fuzzy',    label: 'Fuzzy Logic',   icon: '🔵', dot: '#00d2ff' },
  { id: 'nn',       label: 'Neural Net',    icon: '🟣', dot: '#a855f7' },
  { id: 'anfis',    label: 'ANFIS',         icon: '🟠', dot: '#ff6b35' },
];

export default function App() {
  const [tab, setTab]           = useState('predict');
  const [form, setForm]         = useState({ 'PM2.5': 50, 'PM10': 80, 'NO2': 20, 'CO': 1.2 });
  const [result, setResult]     = useState(null);
  const [fuzzyEx, setFuzzyEx]   = useState(null);
  const [anfisEx, setAnfisEx]   = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const [liveCity, setLiveCity] = useState('mumbai');

  // Fetch model architecture metadata once
  useEffect(() => {
    axios.get(`${API}/model-info`)
      .then(r => setModelInfo(r.data))
      .catch(() => {});
  }, []);

  const handlePredict = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [predRes, fuzzyRes, anfisRes] = await Promise.all([
        axios.post(`${API}/predict`,       form),
        axios.post(`${API}/explain/fuzzy`, form),
        axios.post(`${API}/explain/anfis`, form),
      ]);
      setResult(predRes.data);
      setFuzzyEx(fuzzyRes.data);
      setAnfisEx(anfisRes.data);
    } catch (e) {
      setError(e.response?.data?.error || 'Backend unavailable. Is Flask running?');
    } finally {
      setLoading(false);
    }
  }, [form]);

  const handleLive = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await axios.get(`${API}/live/${liveCity}`);
      if (r.data.error) { setError(r.data.error); return; }
      setForm({
        'PM2.5': r.data['PM2.5'] || 0,
        'PM10':  r.data['PM10']  || 0,
        'NO2':   r.data['NO2']   || 0,
        'CO':    r.data['CO']    || 0,
      });
    } catch (e) {
      setError('Could not fetch live data.');
    } finally {
      setLoading(false);
    }
  }, [liveCity]);

  const avgAQI = result
    ? Math.round((result.ANFIS + result.NN + result.Fuzzy) / 3)
    : null;

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* ── Header ───────────────────────────────────────────────────── */}
      <header style={{
        padding: '16px 32px',
        borderBottom: '1px solid var(--border)',
        background: 'var(--bg-surface)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{
            width: 40, height: 40, borderRadius: 10,
            background: 'linear-gradient(135deg,#1f6feb,#ff6b35)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 20,
          }}>🌍</div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 17 }}>AQI Soft Computing</div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              Fuzzy Logic · Neural Networks · ANFIS
            </div>
          </div>
        </div>
        {avgAQI !== null && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Predicted AQI</span>
            <span style={{
              fontFamily: 'JetBrains Mono', fontWeight: 700, fontSize: 20,
              color: aqiColor(avgAQI),
            }}>{avgAQI}</span>
            <span className="badge" style={{ background: aqiColor(avgAQI)+'22', color: aqiColor(avgAQI) }}>
              {aqiCategory(avgAQI)}
            </span>
          </div>
        )}
      </header>

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* ── Sidebar ────────────────────────────────────────────────── */}
        <aside style={{
          width: 220, padding: '20px 12px',
          borderRight: '1px solid var(--border)',
          background: 'var(--bg-surface)',
          display: 'flex', flexDirection: 'column', gap: 4,
          flexShrink: 0,
        }}>
          {TABS.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              style={{
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '10px 14px', borderRadius: 8,
                background: tab === t.id ? 'rgba(255,255,255,0.07)' : 'transparent',
                border: tab === t.id ? '1px solid var(--border-lit)' : '1px solid transparent',
                color: tab === t.id ? 'var(--text-primary)' : 'var(--text-secondary)',
                cursor: 'pointer', fontSize: 14, fontFamily: 'inherit',
                fontWeight: tab === t.id ? 600 : 400,
                textAlign: 'left', width: '100%',
                transition: 'all 0.15s',
              }}
            >
              <span style={{
                width: 8, height: 8, borderRadius: '50%',
                background: tab === t.id ? t.dot : 'var(--text-muted)',
                flexShrink: 0, transition: 'background 0.15s',
              }} />
              {t.label}
            </button>
          ))}

          {/* Error banner */}
          {error && (
            <div style={{
              marginTop: 'auto', padding: '10px 12px',
              background: 'rgba(239,68,68,0.1)',
              border: '1px solid rgba(239,68,68,0.3)',
              borderRadius: 8, fontSize: 12, color: '#fca5a5',
            }}>
              ⚠️ {error}
            </div>
          )}
        </aside>

        {/* ── Main ───────────────────────────────────────────────────── */}
        <main style={{ flex: 1, overflowY: 'auto', padding: '28px 32px' }}>

          {tab === 'predict' && (
            <div className="fade-in">
              <div className="section-title">🎯 Predict Air Quality</div>
              <div className="section-subtitle">
                Enter pollutant levels or fetch live data, then run all three models simultaneously.
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: 24, alignItems: 'start' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                  <InputForm
                    form={form} setForm={setForm}
                    onPredict={handlePredict}
                    onLive={handleLive}
                    liveCity={liveCity} setLiveCity={setLiveCity}
                    loading={loading}
                  />
                  {result && (
                    <ModelComparison result={result} />
                  )}
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                  <AQIGauge value={avgAQI} result={result} loading={loading} />
                  {result && <HealthAdvisory aqi={avgAQI} explanation={result.explanation} />}
                </div>
              </div>
            </div>
          )}

          {tab === 'fuzzy' && (
            <div className="fade-in">
              <FuzzyPanel fuzzyEx={fuzzyEx} modelInfo={modelInfo} form={form} />
            </div>
          )}

          {tab === 'nn' && (
            <div className="fade-in">
              <NNPanel modelInfo={modelInfo} result={result} form={form} />
            </div>
          )}

          {tab === 'anfis' && (
            <div className="fade-in">
              <ANFISPanel anfisEx={anfisEx} modelInfo={modelInfo} result={result} />
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

// ── AQI helpers (shared) ─────────────────────────────────────────────────────
export function aqiColor(aqi) {
  if (!aqi) return '#8b949e';
  if (aqi <= 50)  return '#22c55e';
  if (aqi <= 100) return '#eab308';
  if (aqi <= 150) return '#f97316';
  if (aqi <= 200) return '#ef4444';
  if (aqi <= 300) return '#a855f7';
  return '#7c3aed';
}

export function aqiCategory(aqi) {
  if (!aqi) return '—';
  if (aqi <= 50)  return 'Good';
  if (aqi <= 100) return 'Moderate';
  if (aqi <= 150) return 'Unhealthy (Sensitive)';
  if (aqi <= 200) return 'Unhealthy';
  if (aqi <= 300) return 'Very Unhealthy';
  return 'Hazardous';
}
