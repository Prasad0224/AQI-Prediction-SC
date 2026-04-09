import React, { useRef, useEffect } from 'react';
import { aqiColor, aqiCategory } from '../App';

const MAX_AQI = 400;

export default function AQIGauge({ value, result, loading }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const cx = W / 2, cy = H - 20;
    const R  = Math.min(W, H) * 0.88 / 2;

    ctx.clearRect(0, 0, W, H);

    // ── Background arc ──────────────────────────────────────────
    const grad = ctx.createConicGradient(Math.PI, cx, cy);
    grad.addColorStop(0,    '#22c55e');
    grad.addColorStop(0.13, '#eab308');
    grad.addColorStop(0.25, '#f97316');
    grad.addColorStop(0.38, '#ef4444');
    grad.addColorStop(0.50, '#7c3aed');

    ctx.beginPath();
    ctx.arc(cx, cy, R, Math.PI, 2 * Math.PI);
    ctx.lineWidth = 22;
    ctx.strokeStyle = grad;
    ctx.lineCap = 'round';
    ctx.stroke();

    // ── Overlay grey if no value ────────────────────────────────
    if (!value) {
      ctx.beginPath();
      ctx.arc(cx, cy, R, Math.PI, 2 * Math.PI);
      ctx.lineWidth = 22;
      ctx.strokeStyle = 'rgba(255,255,255,0.05)';
      ctx.stroke();
      // Label
      ctx.fillStyle = '#484f58';
      ctx.font = 'bold 14px Inter';
      ctx.textAlign = 'center';
      ctx.fillText('Run a prediction', cx, cy - R * 0.3);
      return;
    }

    // ── Filled arc ──────────────────────────────────────────────
    const fraction = Math.min(value / MAX_AQI, 1);
    const endAngle = Math.PI + fraction * Math.PI;

    ctx.beginPath();
    ctx.arc(cx, cy, R, Math.PI, endAngle);
    ctx.lineWidth = 22;
    ctx.strokeStyle = aqiColor(value);
    ctx.lineCap = 'round';
    ctx.stroke();

    // ── Needle ──────────────────────────────────────────────────
    const angle = Math.PI + fraction * Math.PI;
    const nx = cx + (R - 0) * Math.cos(angle);
    const ny = cy + (R - 0) * Math.sin(angle);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(nx, ny);
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.stroke();

    // ── Center dot ──────────────────────────────────────────────
    ctx.beginPath();
    ctx.arc(cx, cy, 6, 0, 2 * Math.PI);
    ctx.fillStyle = '#fff';
    ctx.fill();

    // ── Value text ──────────────────────────────────────────────
    ctx.fillStyle = aqiColor(value);
    ctx.font = 'bold 36px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(String(Math.round(value)), cx, cy - R * 0.25);

    ctx.fillStyle = '#8b949e';
    ctx.font = '12px Inter';
    ctx.fillText('AQI (avg)', cx, cy - R * 0.05);
  }, [value, loading]);

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
      <div style={{ fontWeight: 700, fontSize: 15, alignSelf: 'flex-start' }}>AQI Gauge</div>

      {loading
        ? <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <span className="spinner" />
          </div>
        : <canvas ref={canvasRef} width={280} height={160} style={{ maxWidth: '100%' }} />
      }

      {/* Per-model breakdown */}
      {result && (
        <div style={{ display: 'flex', gap: 10, width: '100%' }}>
          {[
            { label: 'Fuzzy', val: result.Fuzzy, color: '#00d2ff' },
            { label: 'NN',    val: result.NN,    color: '#a855f7' },
            { label: 'ANFIS', val: result.ANFIS, color: '#ff6b35' },
          ].map(m => (
            <div key={m.label} style={{
              flex: 1, background: 'var(--bg-base)',
              border: `1px solid ${m.color}44`,
              borderRadius: 8, padding: '8px',
              textAlign: 'center',
            }}>
              <div style={{ fontSize: 18, fontWeight: 700, fontFamily: 'JetBrains Mono', color: m.color }}>
                {Math.round(m.val)}
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 2 }}>{m.label}</div>
            </div>
          ))}
        </div>
      )}

      {value && (
        <div className="badge" style={{
          background: aqiColor(value) + '22',
          color: aqiColor(value),
          fontSize: 13, padding: '6px 16px',
        }}>
          {aqiCategory(value)}
        </div>
      )}
    </div>
  );
}
