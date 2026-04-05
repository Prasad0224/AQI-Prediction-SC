import React, { useMemo } from 'react';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, Filler, Tooltip, Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { aqiColor, aqiCategory } from '../App';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Filler, Tooltip, Legend);

function gaussian(x, c, sigma) {
  return Math.exp(-Math.pow(x - c, 2) / (2 * sigma * sigma));
}

const X_RANGE = Array.from({ length: 100 }, (_, i) => i / 99);

function MFChart({ title, mfParams, currentVal, accentColor }) {
  const LABEL_ORDER = ['Low', 'Medium', 'High'];
  const COLORS      = ['#22c55e', '#eab308', '#ef4444'];

  const datasets = LABEL_ORDER.map((lbl, idx) => {
    const p   = mfParams[lbl];
    const col = COLORS[idx];
    return {
      label: lbl,
      data:  X_RANGE.map(x => gaussian(x, p.c, p.sigma)),
      borderColor: col,
      backgroundColor: col + '18',
      borderWidth: 2,
      fill: true,
      pointRadius: 0,
      tension: 0.4,
    };
  });

  const currentPlugin = {
    id: 'currentLine',
    beforeDraw(chart) {
      if (currentVal == null) return;
      const { ctx, scales } = chart;
      const xPixel = scales.x.getPixelForValue(currentVal * 99);
      ctx.save();
      ctx.beginPath();
      ctx.strokeStyle = accentColor;
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.moveTo(xPixel, scales.y.top);
      ctx.lineTo(xPixel, scales.y.bottom);
      ctx.stroke();
      ctx.restore();
    },
  };

  const chartData = { labels: X_RANGE.map(x => x.toFixed(2)), datasets };

  const options = {
    responsive: true,
    animation: { duration: 300 },
    plugins: {
      legend: {
        labels: { color: '#8b949e', font: { family: 'Inter', size: 11 }, boxWidth: 12 },
      },
      tooltip: {
        backgroundColor: '#161b22', borderColor: 'rgba(255,255,255,0.1)',
        borderWidth: 1, titleColor: '#e6edf3', bodyColor: '#8b949e',
      },
    },
    scales: {
      x: {
        ticks: { color: '#484f58', font: { size: 10 }, maxTicksLimit: 6,
          callback: (_, i) => X_RANGE[i]?.toFixed(1) },
        grid: { color: 'rgba(255,255,255,0.04)' },
        title: { display: true, text: 'Normalised value', color: '#8b949e', font: { size: 10 } },
      },
      y: {
        min: 0, max: 1.05,
        ticks: { color: '#484f58', font: { size: 10 } },
        grid: { color: 'rgba(255,255,255,0.04)' },
        title: { display: true, text: 'µ (membership)', color: '#8b949e', font: { size: 10 } },
      },
    },
  };

  return (
    <div className="card" style={{ borderTop: `3px solid ${accentColor}` }}>
      <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 10, color: accentColor }}>{title}</div>
      <Line data={chartData} options={options} plugins={[currentPlugin]} />
      {currentVal != null && (
        <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-muted)' }}>
          ↑ Dashed line = normalised current input ({currentVal.toFixed(3)})
        </div>
      )}
    </div>
  );
}

// Approximate normalised values for the dashed-line marker
// (visual only — actual inference uses server-side scaler)
const APPROX_MAX = { 'PM2.5': 500, 'PM10': 900, 'NO2': 400, 'CO': 50 };

export default function FuzzyPanel({ fuzzyEx, modelInfo, form }) {
  const mfParams = modelInfo?.fuzzy?.mf_params;

  const normVals = useMemo(() => {
    if (!form) return {};
    return Object.fromEntries(
      Object.entries(APPROX_MAX).map(([k, max]) => [k, Math.min(form[k] / max, 1)])
    );
  }, [form]);

  const FEATURES = ['PM2.5', 'PM10', 'NO2', 'CO'];

  return (
    <div>
      {/* Title */}
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 4 }}>
        <div style={{ width: 10, height: 10, borderRadius: '50%', background: '#00d2ff' }} />
        <div className="section-title" style={{ color: '#00d2ff' }}>Fuzzy Logic System</div>
      </div>
      <div className="section-subtitle">
        Improved Mamdani FIS — uses all 4 pollutants. Each pollutant independently infers a
        sub-AQI through calibrated Gaussian MFs. Final AQI = max(sub-AQIs), mirroring the
        real AQI standard where the dominant pollutant determines the overall index.
      </div>

      {/* How it works */}
      <div className="card" style={{ marginBottom: 20, borderLeft: '3px solid #00d2ff' }}>
        <div style={{ fontWeight: 600, marginBottom: 8 }}>⚙️ How It Works</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12 }}>
          {[
            { step: '1', title: 'Fuzzification',       desc: 'Each pollutant value mapped to Low / Medium / High via calibrated Gaussian MFs' },
            { step: '2', title: 'Sub-AQI Inference',   desc: 'Centroid defuzz: sub-AQI = Σ(μ × singleton) / Σμ for each of 4 pollutants' },
            { step: '3', title: 'Max Aggregation',     desc: 'Final AQI = max(sub-AQIs) — dominant pollutant wins, true to real AQI standard' },
            { step: '4', title: 'Calibration',         desc: 'MF centres set to 25th/75th percentiles of training data — not hardcoded' },
          ].map(s => (
            <div key={s.step} style={{
              background: 'var(--bg-base)', borderRadius: 8,
              padding: 12, border: '1px solid var(--border)',
            }}>
              <div style={{
                width: 24, height: 24, borderRadius: '50%',
                background: '#00d2ff22', border: '1px solid #00d2ff',
                color: '#00d2ff', fontSize: 12, fontWeight: 700,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                marginBottom: 8,
              }}>{s.step}</div>
              <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 3 }}>{s.title}</div>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{s.desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Per-pollutant sub-AQI result — shown when prediction is run */}
      {fuzzyEx && (
        <div className="card" style={{ marginBottom: 20, borderLeft: '3px solid #00d2ff' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
            <div style={{ fontWeight: 600, fontSize: 14 }}>📊 Per-Pollutant Sub-AQI</div>
            <div style={{
              background: aqiColor(fuzzyEx.sub_aqis[fuzzyEx.dominant_pollutant]) + '22',
              border: `1px solid ${aqiColor(fuzzyEx.sub_aqis[fuzzyEx.dominant_pollutant])}66`,
              borderRadius: 8, padding: '4px 10px', fontSize: 12, fontWeight: 600,
              color: aqiColor(fuzzyEx.sub_aqis[fuzzyEx.dominant_pollutant]),
            }}>
              Dominant: {fuzzyEx.dominant_pollutant}
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
            {FEATURES.map(f => {
              const val       = fuzzyEx.sub_aqis[f];
              const isDominant = f === fuzzyEx.dominant_pollutant;
              const col        = aqiColor(val);
              return (
                <div key={f} style={{
                  background: col + (isDominant ? '18' : '0a'),
                  border: `1px solid ${col}${isDominant ? '66' : '33'}`,
                  borderRadius: 10, padding: '12px 10px', textAlign: 'center',
                }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>{f}</div>
                  <div style={{ fontFamily: 'JetBrains Mono', fontSize: 22, fontWeight: 700, color: col }}>
                    {Math.round(val)}
                  </div>
                  <div style={{ fontSize: 10, color: col, marginTop: 3 }}>{aqiCategory(val)}</div>
                  {isDominant && (
                    <div style={{ fontSize: 9, color: col, marginTop: 4, fontWeight: 600 }}>★ DOMINANT</div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Membership degrees for each pollutant */}
          <div style={{ marginTop: 14, display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 10 }}>
            {FEATURES.map(f => {
              const mu = fuzzyEx.memberships[f];
              const COLORS = { Low: '#22c55e', Medium: '#eab308', High: '#ef4444' };
              return (
                <div key={f} style={{ background: 'var(--bg-base)', borderRadius: 8, padding: 10 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: 'var(--text-secondary)' }}>
                    {f} Memberships
                  </div>
                  {Object.entries(mu).map(([lbl, v]) => (
                    <div key={lbl} style={{ marginBottom: 7 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 3 }}>
                        <span style={{ color: COLORS[lbl] }}>{lbl}</span>
                        <span style={{ fontFamily: 'JetBrains Mono', color: '#00d2ff', fontSize: 11 }}>{v.toFixed(4)}</span>
                      </div>
                      <div className="progress-track">
                        <div className="progress-fill" style={{ width: `${v * 100}%`, background: COLORS[lbl] }} />
                      </div>
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* 4 MF Charts in 2×2 grid */}
      {mfParams
        ? (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
            {FEATURES.map(f => (
              <MFChart
                key={f}
                title={`${f} Membership Functions`}
                mfParams={mfParams[f]}
                currentVal={normVals[f] ?? null}
                accentColor="#00d2ff"
              />
            ))}
          </div>
        )
        : (
          <div className="card" style={{ marginBottom: 20, color: 'var(--text-muted)', fontSize: 13 }}>
            Run a prediction first to load calibrated MF parameters.
          </div>
        )
      }
    </div>
  );
}
