import React from 'react';
import {
  Chart as ChartJS, CategoryScale, LinearScale, BarElement, Tooltip, Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

// SVG 5-layer ANFIS architecture diagram
function ANFISDiagram({ anfisEx }) {
  const W = 620, H = 230;
  const LAYERS = [
    { name: 'L1\nFuzzify', nodes: 4, color: '#58a6ff', desc: 'Gaussian MFs' },
    { name: 'L2\nFire',    nodes: 6, color: '#a855f7', desc: 'Product T-norm' },
    { name: 'L3\nNorm',    nodes: 6, color: '#00d2ff', desc: 'Normalise w̄' },
    { name: 'L4\nConseq',  nodes: 6, color: '#ff6b35', desc: 'Sugeno Linear' },
    { name: 'L5\nOutput',  nodes: 1, color: '#22c55e', desc: 'Weighted Sum' },
  ];

  const MAX_VIS = 4;
  const lxArr  = LAYERS.map((_, i) => 50 + i * (W - 80) / (LAYERS.length - 1));

  const nodeY = (count, j) => {
    const vis = Math.min(count, MAX_VIS);
    const gap = Math.min(38, (H - 60) / Math.max(vis, 1));
    const total = (vis - 1) * gap;
    return H / 2 - total / 2 + j * gap + 10;
  };

  const elements = [];

  // Connections
  LAYERS.forEach((la, li) => {
    if (li === LAYERS.length - 1) return;
    const lb  = LAYERS[li + 1];
    const visA = Math.min(la.nodes, MAX_VIS);
    const visB = Math.min(lb.nodes, MAX_VIS);
    const alpha = li === 1 ? '0.15' : '0.07';
    for (let a = 0; a < visA; a++) {
      for (let b = 0; b < visB; b++) {
        elements.push(
          <line key={`e${li}-${a}-${b}`}
            x1={lxArr[li]} y1={nodeY(la.nodes, a)}
            x2={lxArr[li+1]} y2={nodeY(lb.nodes, b)}
            stroke={`rgba(255,255,255,${alpha})`} strokeWidth="1"
          />
        );
      }
    }
  });

  // Nodes + labels
  LAYERS.forEach((layer, li) => {
    const vis = Math.min(layer.nodes, MAX_VIS);

    // Activation-based sizing from anfisEx
    const activations = anfisEx?.normalized || [];

    for (let j = 0; j < vis; j++) {
      const cy  = nodeY(layer.nodes, j);
      const act = li >= 1 && li <= 3 ? (activations[j] ?? 0) : 1;
      const opacity = li >= 1 && li <= 3 && activations.length ? 0.3 + act * 0.7 : 1;
      elements.push(
        <circle key={`n${li}-${j}`}
          cx={lxArr[li]} cy={cy} r={11}
          fill={layer.color + '33'} stroke={layer.color}
          strokeWidth={opacity > 0.7 ? 2.5 : 1.5}
          opacity={opacity}
        />
      );
    }

    if (layer.nodes > MAX_VIS) {
      elements.push(
        <text key={`d${li}`} x={lxArr[li]} y={nodeY(layer.nodes, MAX_VIS - 1) + 20}
          textAnchor="middle" fill={layer.color} fontSize="13">⋮</text>
      );
    }

    // Layer label (bottom)
    elements.push(
      <text key={`lbl${li}`} x={lxArr[li]} y={H - 14}
        textAnchor="middle" fill={layer.color} fontSize="9"
        fontFamily="JetBrains Mono">{layer.name.replace('\n', ' ')}</text>
    );
    elements.push(
      <text key={`desc${li}`} x={lxArr[li]} y={H - 4}
        textAnchor="middle" fill="#484f58" fontSize="8.5"
        fontFamily="Inter">{layer.desc}</text>
    );
  });

  // Hybrid learning arrows
  elements.push(
    <g key="hybrid">
      <path d={`M ${lxArr[0]} ${H-40} C ${(lxArr[0]+lxArr[3])/2} ${H+10} ${(lxArr[0]+lxArr[3])/2} ${H+10} ${lxArr[3]} ${H-40}`}
        stroke="#22c55e" strokeWidth="1.5" strokeDasharray="4 3" fill="none" opacity="0.5" />
      <text x={(lxArr[0]+lxArr[3])/2} y={H+22} textAnchor="middle" fill="#22c55e" fontSize="8.5" opacity="0.7">
        LSE (consequents)
      </text>
    </g>
  );
  elements.push(
    <g key="gd">
      <path d={`M ${lxArr[3]} ${H-52} C ${(lxArr[0]+lxArr[3])/2} ${H-65} ${(lxArr[0]+lxArr[3])/2} ${H-65} ${lxArr[0]} ${H-52}`}
        stroke="#ff6b35" strokeWidth="1.5" strokeDasharray="4 3" fill="none" opacity="0.4" markerEnd="url(#arrow)" />
      <text x={(lxArr[0]+lxArr[3])/2} y={H-70} textAnchor="middle" fill="#ff6b35" fontSize="8.5" opacity="0.7">
        GD (premises)
      </text>
    </g>
  );

  return (
    <svg width={W} height={H + 30} style={{ maxWidth: '100%', display: 'block' }}>
      <defs>
        <marker id="arrow" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#ff6b35" />
        </marker>
      </defs>
      {elements}
    </svg>
  );
}

export default function ANFISPanel({ anfisEx, modelInfo, result }) {
  const anfisInfo = modelInfo?.anfis;

  const ruleChartData = anfisEx
    ? {
        labels: anfisEx.normalized.map((_, i) => `R${i + 1}`),
        datasets: [
          {
            label: 'Normalised Firing Strength',
            data:  anfisEx.normalized.map(v => parseFloat(v.toFixed(4))),
            backgroundColor: anfisEx.normalized.map(v =>
              v > 0.3 ? '#ff6b35cc' : v > 0.1 ? '#ff6b3566' : '#ff6b3522'
            ),
            borderColor:     '#ff6b35',
            borderWidth: 1.5,
            borderRadius: 6,
          },
        ],
      }
    : null;

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#161b22', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1,
        titleColor: '#e6edf3', bodyColor: '#8b949e',
      },
    },
    scales: {
      x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#8b949e' } },
      y: { min: 0, max: 1, grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#8b949e' } },
    },
  };

  return (
    <div>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 4 }}>
        <div style={{ width: 10, height: 10, borderRadius: '50%', background: '#ff6b35' }} />
        <div className="section-title" style={{ color: '#ff6b35' }}>
          ANFIS — Adaptive Neuro-Fuzzy Inference System
        </div>
      </div>
      <div className="section-subtitle">
        Combines the interpretability of fuzzy logic with the learning power of neural networks.
        Uses a hybrid algorithm: LSE for consequent parameters + gradient descent for premise parameters.
      </div>

      {/* Explanation cards */}
      <div className="card" style={{ marginBottom: 20, borderLeft: '3px solid #ff6b35' }}>
        <div style={{ fontWeight: 600, marginBottom: 10 }}>⚙️ Hybrid Learning Algorithm</div>
        <div className="grid-2">
          <div style={{ background: 'var(--bg-base)', borderRadius: 8, padding: 14, border: '1px solid #22c55e44' }}>
            <div style={{ color: '#22c55e', fontWeight: 600, marginBottom: 6 }}>
              Forward Pass — LSE
            </div>
            <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              Given fixed premise parameters (c, σ), the consequent parameters
              (p, q, r) of the Sugeno rule <em>f = px + qy + r</em> are solved
              analytically using Least Squares Estimation. This gives the globally
              optimal consequent for the current MF configuration.
            </div>
          </div>
          <div style={{ background: 'var(--bg-base)', borderRadius: 8, padding: 14, border: '1px solid #ff6b3544' }}>
            <div style={{ color: '#ff6b35', fontWeight: 600, marginBottom: 6 }}>
              Backward Pass — Gradient Descent
            </div>
            <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              With fixed consequent params, the error gradient flows back
              through the Sugeno output → normalisation → product T-norm →
              Gaussian MFs. Chain rule updates MF centers (c) and widths (σ).
            </div>
          </div>
        </div>
      </div>

      {/* Architecture diagram */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 4 }}>
          🏗️ 5-Layer ANFIS Architecture
        </div>
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 12 }}>
          {anfisInfo ? `${anfisInfo.n_inputs} inputs · ${anfisInfo.n_rules} fuzzy rules` : '4 inputs · 6 fuzzy rules'}
          {anfisEx && ' — node brightness = current firing strength'}
        </div>
        <ANFISDiagram anfisEx={anfisEx} />
        <div style={{ display: 'flex', gap: 8, marginTop: 12, flexWrap: 'wrap' }}>
          {[
            { label: 'L1: Fuzzify',   color: '#58a6ff', desc: 'Gaussian MFs — μ = exp(-(x-c)²/2σ²)' },
            { label: 'L2: Fire',      color: '#a855f7', desc: 'Rule strength — w = ∏μᵢ' },
            { label: 'L3: Norm',      color: '#00d2ff', desc: 'w̄ₖ = wₖ/Σwₖ' },
            { label: 'L4: Conseq',    color: '#ff6b35', desc: 'fₖ = pₖ·x + bₖ (Sugeno)' },
            { label: 'L5: Output',    color: '#22c55e', desc: 'y = Σ w̄ₖ fₖ' },
          ].map(l => (
            <div key={l.label} style={{
              fontSize: 11, background: 'var(--bg-base)',
              border: `1px solid ${l.color}44`, borderRadius: 6,
              padding: '5px 10px', color: 'var(--text-secondary)',
            }}>
              <span style={{ color: l.color, fontWeight: 600 }}>{l.label}</span> — {l.desc}
            </div>
          ))}
        </div>
      </div>

      {/* Rule activations chart */}
      <div className="grid-2" style={{ marginBottom: 20 }}>
        <div className="card">
          <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 12 }}>
            🔥 Rule Firing Strengths
          </div>
          {ruleChartData
            ? <Bar data={ruleChartData} options={chartOptions} />
            : <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>
                Run a prediction to see rule activations.
              </div>
          }
        </div>

        <div className="card">
          <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 14 }}>📊 Metrics</div>
          {anfisInfo?.train_rmse != null
            ? <>
                <div style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
                  <div className="metric-chip" style={{ borderColor: '#ff6b3544' }}>
                    <div className="value" style={{ color: '#ff6b35' }}>{anfisInfo.train_rmse.toFixed(4)}</div>
                    <div className="key">Train RMSE</div>
                  </div>
                  <div className="metric-chip" style={{ borderColor: '#ff6b3544' }}>
                    <div className="value" style={{ color: '#ff6b35' }}>{anfisInfo.r2.toFixed(4)}</div>
                    <div className="key">R² Score</div>
                  </div>
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.6 }}>
                  Rules: {anfisInfo.n_rules} | Inputs: {anfisInfo.n_inputs}<br />
                  Premise params: centers + sigmas per rule per input<br />
                  Consequent params: {anfisInfo.n_rules * (anfisInfo.n_inputs + 1)} total (Sugeno linear)
                </div>
              </>
            : <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>
                Load model info by running a prediction.
              </div>
          }
          {result && (
            <div style={{ marginTop: 14, paddingTop: 14, borderTop: '1px solid var(--border)' }}>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>Current prediction</div>
              <div style={{ fontFamily: 'JetBrains Mono', fontSize: 28, fontWeight: 700, color: '#ff6b35' }}>
                {Math.round(result.ANFIS)} <span style={{ fontSize: 14, color: 'var(--text-secondary)' }}>AQI</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Raw activations table */}
      {anfisEx && (
        <div className="card">
          <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 10 }}>📋 Rule Activation Detail</div>
          <table className="rule-table">
            <thead>
              <tr>
                <th>Rule</th>
                <th>Raw Firing w</th>
                <th>Normalised w̄</th>
                <th>Contribution</th>
              </tr>
            </thead>
            <tbody>
              {anfisEx.normalized.map((norm, i) => (
                <tr key={i} style={{ opacity: norm < 0.005 ? 0.35 : 1 }}>
                  <td style={{ fontFamily: 'JetBrains Mono', color: '#ff6b35' }}>R{i + 1}</td>
                  <td style={{ fontFamily: 'JetBrains Mono', fontSize: 12 }}>
                    {anfisEx.firing_strengths[i]?.toFixed(5)}
                  </td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div className="progress-track" style={{ flex: 1 }}>
                        <div className="progress-fill"
                          style={{ width: `${norm * 100}%`, background: '#ff6b35' }} />
                      </div>
                      <span style={{ fontFamily: 'JetBrains Mono', fontSize: 11, color: '#ff6b35', minWidth: 50 }}>
                        {norm.toFixed(4)}
                      </span>
                    </div>
                  </td>
                  <td style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                    {norm > 0.3 ? '🔴 Dominant' : norm > 0.1 ? '🟡 Active' : '⚪ Weak'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
