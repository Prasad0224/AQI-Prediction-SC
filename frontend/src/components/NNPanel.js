import React from 'react';

// SVG-based MLP architecture diagram
function NNDiagram({ arch }) {
  const inputSize  = arch?.input_size    || 4;
  const hidden     = arch?.hidden_layers || [64, 32];
  const outputSize = arch?.output_size   || 1;

  const layers  = [inputSize, ...hidden, outputSize];
  const MAX_VIS = 5;  // max visible nodes per layer

  const W   = 560, H = 220;
  const lx  = layers.map((_, i) => 60 + i * (W - 120) / (layers.length - 1));
  const col = ['#58a6ff', '#a855f7', '#a855f7', '#ff6b35'];

  const nodeY = (count, j) => {
    const vis = Math.min(count, MAX_VIS);
    const gap = Math.min(34, (H - 40) / vis);
    const total = (vis - 1) * gap;
    return H / 2 - total / 2 + j * gap;
  };

  const elements = [];

  // Draw connections (light, thin)
  layers.forEach((countA, li) => {
    if (li === layers.length - 1) return;
    const countB = layers[li + 1];
    const visA = Math.min(countA, MAX_VIS);
    const visB = Math.min(countB, MAX_VIS);
    for (let a = 0; a < visA; a++) {
      for (let b = 0; b < visB; b++) {
        elements.push(
          <line key={`e${li}-${a}-${b}`}
            x1={lx[li]} y1={nodeY(countA, a)}
            x2={lx[li+1]} y2={nodeY(countB, b)}
            stroke="rgba(255,255,255,0.05)" strokeWidth="1"
          />
        );
      }
    }
  });

  // Draw nodes
  layers.forEach((count, li) => {
    const vis = Math.min(count, MAX_VIS);
    const c   = col[Math.min(li, col.length - 1)];
    for (let j = 0; j < vis; j++) {
      const cy = nodeY(count, j);
      elements.push(
        <circle key={`n${li}-${j}`}
          cx={lx[li]} cy={cy} r={10}
          fill={c + '22'} stroke={c} strokeWidth="1.5"
        />
      );
    }
    // Ellipsis indicator for large layers
    if (count > MAX_VIS) {
      elements.push(
        <text key={`dots${li}`} x={lx[li]} y={H / 2 + 20}
          textAnchor="middle" fill={c} fontSize="14" fontWeight="bold">⋮</text>
      );
      elements.push(
        <text key={`cnt${li}`} x={lx[li]} y={H - 8}
          textAnchor="middle" fill={c} fontSize="9" fontFamily="JetBrains Mono">{count}</text>
      );
    }
    // Layer label
    const labels = ['Input', ...hidden.map((_, i) => `H${i+1}`), 'Output'];
    elements.push(
      <text key={`lbl${li}`} x={lx[li]} y={14}
        textAnchor="middle" fill="#8b949e" fontSize="10" fontFamily="Inter">{labels[li]}</text>
    );
  });

  return (
    <svg width={W} height={H} style={{ maxWidth: '100%', display: 'block' }}>
      {elements}
    </svg>
  );
}

export default function NNPanel({ modelInfo, result, form }) {
  const nn   = modelInfo?.nn;
  const arch = nn;

  return (
    <div>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 4 }}>
        <div style={{ width: 10, height: 10, borderRadius: '50%', background: '#a855f7' }} />
        <div className="section-title" style={{ color: '#a855f7' }}>Neural Network (MLP)</div>
      </div>
      <div className="section-subtitle">
        Multilayer Perceptron trained with backpropagation and the Adam optimiser.
        Learns non-linear mappings from pollutant concentrations to AQI.
      </div>

      {/* How it works */}
      <div className="card" style={{ marginBottom: 20, borderLeft: '3px solid #a855f7' }}>
        <div style={{ fontWeight: 600, marginBottom: 8 }}>⚙️ How It Works</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12 }}>
          {[
            { step: '1', title: 'Forward Pass',    desc: 'Input propagates through layers. Each neuron computes weighted sum + ReLU activation' },
            { step: '2', title: 'Loss',            desc: 'Mean Squared Error between predicted AQI and actual AQI is computed' },
            { step: '3', title: 'Backpropagation', desc: '∂Loss/∂w is computed via chain rule through all layers' },
            { step: '4', title: 'Adam Update',     desc: 'Adaptive learning rate adjusts each weight using momentum and variance estimates' },
          ].map(s => (
            <div key={s.step} style={{
              background: 'var(--bg-base)', borderRadius: 8,
              padding: 12, border: '1px solid var(--border)',
            }}>
              <div style={{
                width: 24, height: 24, borderRadius: '50%',
                background: '#a855f722', border: '1px solid #a855f7',
                color: '#a855f7', fontSize: 12, fontWeight: 700,
                display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 8,
              }}>{s.step}</div>
              <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 3 }}>{s.title}</div>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{s.desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Architecture diagram */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 12 }}>🧬 Network Architecture</div>
        <NNDiagram arch={arch} />
        <div style={{ display: 'flex', gap: 12, marginTop: 12, flexWrap: 'wrap' }}>
          {[
            { label: 'Input Layer',   value: `${arch?.input_size || 4} neurons (PM2.5, PM10, NO2, CO)`, color: '#58a6ff' },
            { label: 'Hidden 1',      value: `${arch?.hidden_layers?.[0] || 64} neurons · ReLU`, color: '#a855f7' },
            { label: 'Hidden 2',      value: `${arch?.hidden_layers?.[1] || 32} neurons · ReLU`, color: '#a855f7' },
            { label: 'Output Layer',  value: '1 neuron (AQI)', color: '#ff6b35' },
          ].map(l => (
            <div key={l.label} style={{
              display: 'flex', gap: 8, alignItems: 'center',
              background: 'var(--bg-base)', padding: '6px 12px', borderRadius: 7,
              border: `1px solid ${l.color}44`, fontSize: 12,
            }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: l.color, flexShrink: 0 }} />
              <strong style={{ color: l.color }}>{l.label}:</strong>
              <span style={{ color: 'var(--text-secondary)' }}>{l.value}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Metrics */}
      {nn?.val_rmse != null && (
        <div className="card" style={{ marginBottom: 20 }}>
          <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 16 }}>📊 Training Metrics</div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            {[
              { key: 'Train RMSE', value: nn.train_rmse?.toFixed(4), unit: '(norm.)' },
              { key: 'Val RMSE',   value: nn.val_rmse?.toFixed(4),   unit: '(norm.)' },
              { key: 'R² Score',   value: nn.r2?.toFixed(4),         unit: '' },
            ].map(m => (
              <div key={m.key} className="metric-chip" style={{ borderColor: '#a855f744' }}>
                <div className="value" style={{ color: '#a855f7' }}>{m.value ?? '—'}</div>
                <div className="key">{m.key} {m.unit}</div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 12, fontSize: 12, color: 'var(--text-secondary)', padding: '8px 12px', background: 'var(--bg-base)', borderRadius: 7 }}>
            RMSE and R² are computed on a held-out 20% test set. Values are on the normalised [0,1] AQI scale.
          </div>
        </div>
      )}

      {/* Current prediction */}
      {result && (
        <div className="card" style={{ borderColor: '#a855f744' }}>
          <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 8 }}>🎯 Current Prediction</div>
          <div style={{ fontFamily: 'JetBrains Mono', fontSize: 32, fontWeight: 700, color: '#a855f7' }}>
            {Math.round(result.NN)} <span style={{ fontSize: 16, fontWeight: 400, color: 'var(--text-secondary)' }}>AQI</span>
          </div>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 4 }}>
            For inputs: PM2.5={form?.['PM2.5']}, PM10={form?.['PM10']}, NO2={form?.['NO2']}, CO={form?.['CO']}
          </div>
        </div>
      )}
    </div>
  );
}
