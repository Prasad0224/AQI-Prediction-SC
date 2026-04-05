import React, { useMemo } from 'react';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, Filler, Tooltip, Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Filler, Tooltip, Legend);

// Compute Gaussian MF values client-side
function gaussian(x, c, sigma) {
  return Math.exp(-Math.pow(x - c, 2) / (2 * sigma * sigma));
}

const X_RANGE = Array.from({ length: 100 }, (_, i) => i / 99);

function MFChart({ title, mfParams, currentVal, accentColor }) {
  const datasets = Object.entries(mfParams).map(([label, { c, sigma }], idx) => {
    const colors = ['#22c55e', '#eab308', '#ef4444'];
    const col    = colors[idx] || '#58a6ff';
    return {
      label,
      data:  X_RANGE.map(x => gaussian(x, c, sigma)),
      borderColor: col,
      backgroundColor: col + '18',
      borderWidth: 2,
      fill: true,
      pointRadius: 0,
      tension: 0.4,
    };
  });

  // Vertical line annotation via plugin
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
    animation: { duration: 400 },
    plugins: {
      legend: {
        labels: { color: '#8b949e', font: { family: 'Inter', size: 11 }, boxWidth: 12 },
      },
      tooltip: {
        backgroundColor: '#161b22',
        borderColor: 'rgba(255,255,255,0.1)',
        borderWidth: 1,
        titleColor: '#e6edf3',
        bodyColor: '#8b949e',
      },
    },
    scales: {
      x: {
        ticks: {
          color: '#484f58', font: { size: 10 },
          maxTicksLimit: 6,
          callback: (_, i) => X_RANGE[i]?.toFixed(1),
        },
        grid: { color: 'rgba(255,255,255,0.04)' },
        title: { display: true, text: 'Normalised input value', color: '#8b949e', font: { size: 11 } },
      },
      y: {
        min: 0, max: 1.05,
        ticks: { color: '#484f58', font: { size: 10 } },
        grid: { color: 'rgba(255,255,255,0.04)' },
        title: { display: true, text: 'µ (membership)', color: '#8b949e', font: { size: 11 } },
      },
    },
  };

  return (
    <div className="card" style={{ borderTop: `3px solid ${accentColor}` }}>
      <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 12, color: accentColor }}>{title}</div>
      <Line data={chartData} options={options} plugins={[currentPlugin]} />
      {currentVal != null && (
        <div style={{ marginTop: 8, fontSize: 12, color: 'var(--text-secondary)' }}>
          ↑ Dashed line = normalised current input value ({currentVal.toFixed(3)})
        </div>
      )}
    </div>
  );
}

export default function FuzzyPanel({ fuzzyEx, modelInfo, form }) {
  const mfParams = modelInfo?.fuzzy?.mf_params;

  // Approximate normalisation (assume scaler max ≈ typical max)
  const normPM25 = form ? Math.min(form['PM2.5'] / 300, 1) : null;
  const normNO2  = form ? Math.min(form['NO2']   / 200, 1) : null;

  return (
    <div>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 4 }}>
        <div style={{
          width: 10, height: 10, borderRadius: '50%', background: '#00d2ff',
        }} />
        <div className="section-title" style={{ color: '#00d2ff' }}>Fuzzy Logic System</div>
      </div>
      <div className="section-subtitle">
        Mamdani Fuzzy Inference System — uses linguistic rules and Gaussian membership functions
        to reason about air quality, mimicking human expert knowledge.
      </div>

      {/* How it works */}
      <div className="card" style={{ marginBottom: 20, borderLeft: '3px solid #00d2ff' }}>
        <div style={{ fontWeight: 600, marginBottom: 8 }}>⚙️ How It Works</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12 }}>
          {[
            { step: '1', title: 'Fuzzification',    desc: 'Convert crisp inputs to membership degrees using Gaussian MFs' },
            { step: '2', title: 'Rule Evaluation',  desc: 'Apply 9 IF-THEN rules using min (AND) operator' },
            { step: '3', title: 'Aggregation',      desc: 'Combine all fired rule outputs' },
            { step: '4', title: 'Defuzzification',  desc: 'Centroid method maps aggregated output to crisp AQI value' },
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

      {/* MF Charts */}
      {mfParams
        ? <div className="grid-2" style={{ marginBottom: 20 }}>
            <MFChart
              title="PM2.5 Membership Functions"
              mfParams={mfParams['PM2.5']}
              currentVal={normPM25}
              accentColor="#00d2ff"
            />
            <MFChart
              title="NO2 Membership Functions"
              mfParams={mfParams['NO2']}
              currentVal={normNO2}
              accentColor="#00d2ff"
            />
          </div>
        : <div className="card" style={{ marginBottom: 20, color: 'var(--text-muted)', fontSize: 13 }}>
            Run a prediction first to load model parameters.
          </div>
      }

      {/* Rule table */}
      <div className="card">
        <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 12 }}>📋 Rule Base</div>
        {fuzzyEx
          ? <table className="rule-table">
              <thead>
                <tr>
                  <th>#</th><th>IF PM2.5 is</th><th>AND NO2 is</th>
                  <th>THEN AQI is</th><th>Strength (min)</th>
                </tr>
              </thead>
              <tbody>
                {fuzzyEx.rule_activations.map((r, i) => (
                  <tr key={i} style={{ opacity: r.strength < 0.01 ? 0.4 : 1 }}>
                    <td style={{ color: 'var(--text-muted)', fontFamily: 'JetBrains Mono' }}>{i + 1}</td>
                    <td>
                      <span className="badge" style={{
                        background: r.pm25_label === 'High' ? '#ef444422' : r.pm25_label === 'Medium' ? '#f9731622' : '#22c55e22',
                        color:      r.pm25_label === 'High' ? '#ef4444'   : r.pm25_label === 'Medium' ? '#f97316'   : '#22c55e',
                      }}>{r.pm25_label}</span>
                    </td>
                    <td>
                      <span className="badge" style={{
                        background: r.no2_label === 'High' ? '#ef444422' : r.no2_label === 'Medium' ? '#f9731622' : '#22c55e22',
                        color:      r.no2_label === 'High' ? '#ef4444'   : r.no2_label === 'Medium' ? '#f97316'   : '#22c55e',
                      }}>{r.no2_label}</span>
                    </td>
                    <td style={{ fontWeight: 500 }}>{r.aqi_output}</td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div className="progress-track" style={{ flex: 1 }}>
                          <div className="progress-fill"
                            style={{ width: `${r.strength * 100}%`, background: '#00d2ff' }} />
                        </div>
                        <span style={{ fontFamily: 'JetBrains Mono', fontSize: 11, color: '#00d2ff', minWidth: 40 }}>
                          {r.strength.toFixed(3)}
                        </span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          : <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>
              Run a prediction to see rule activations.
            </div>
        }
      </div>

      {/* Membership degrees */}
      {fuzzyEx && (
        <div className="grid-2" style={{ marginTop: 16 }}>
          {[
            { name: 'PM2.5 Membership Degrees', data: fuzzyEx.pm25_memberships },
            { name: 'NO2 Membership Degrees',   data: fuzzyEx.no2_memberships  },
          ].map(g => (
            <div key={g.name} className="card">
              <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 12 }}>{g.name}</div>
              {Object.entries(g.data).map(([label, val]) => (
                <div key={label} style={{ marginBottom: 10 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, marginBottom: 4 }}>
                    <span>{label}</span>
                    <span style={{ fontFamily: 'JetBrains Mono', color: '#00d2ff' }}>{val.toFixed(4)}</span>
                  </div>
                  <div className="progress-track">
                    <div className="progress-fill" style={{ width: `${val * 100}%`, background: '#00d2ff' }} />
                  </div>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
