import React, { useState, useEffect } from 'react';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, Filler, Tooltip, Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import axios from 'axios';
import { aqiColor, aqiCategory } from '../App';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Filler, Tooltip, Legend);

const CITIES = ['mumbai', 'delhi', 'bangalore', 'kolkata', 'chennai', 'hyderabad', 'pune', 'ahmedabad'];

export default function ForecastPanel({ api }) {
  const [city, setCity]         = useState('mumbai');
  const [forecast, setForecast] = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);

  const fetchForecast = async (c) => {
    setLoading(true);
    setError(null);
    try {
      const r = await axios.get(`${api}/forecast/${c}`);
      setForecast(r.data);
    } catch {
      setError('Could not fetch forecast. Ensure Flask server is running.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchForecast(city); }, [city]);

  const hours = forecast?.forecast || [];
  const labels = hours.map((_, i) => `+${i + 1}h`);

  const chartData = {
    labels,
    datasets: [{
      label: `AQI Forecast (${city})`,
      data:  hours,
      borderColor: '#22c55e',
      backgroundColor: 'rgba(34,197,94,0.08)',
      borderWidth: 2,
      fill: true,
      tension: 0.4,
      pointRadius: 4,
      pointHoverRadius: 7,
      pointBackgroundColor: hours.map(v => aqiColor(v)),
      pointBorderColor:     hours.map(v => aqiColor(v)),
    }],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        labels: { color: '#8b949e', font: { family: 'Inter', size: 12 }, boxWidth: 12 },
      },
      tooltip: {
        backgroundColor: '#161b22', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1,
        titleColor: '#e6edf3', bodyColor: '#8b949e', padding: 12,
        callbacks: {
          label: ctx => `AQI: ${ctx.parsed.y}`,
          afterLabel: ctx => `Category: ${aqiCategory(ctx.parsed.y)}`,
        },
      },
    },
    scales: {
      x: {
        grid: { color: 'rgba(255,255,255,0.04)' },
        ticks: { color: '#8b949e', font: { family: 'JetBrains Mono', size: 10 }, maxTicksLimit: 12 },
      },
      y: {
        grid: { color: 'rgba(255,255,255,0.04)' },
        ticks: { color: '#8b949e', font: { size: 11 } },
        beginAtZero: false,
      },
    },
  };

  const avg = hours.length ? Math.round(hours.reduce((a, b) => a + b, 0) / hours.length) : null;
  const max = hours.length ? Math.round(Math.max(...hours)) : null;
  const min = hours.length ? Math.round(Math.min(...hours)) : null;

  return (
    <div>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 4 }}>
        <div style={{ width: 10, height: 10, borderRadius: '50%', background: '#22c55e' }} />
        <div className="section-title" style={{ color: '#22c55e' }}>24-Hour AQI Forecast</div>
      </div>
      <div className="section-subtitle">
        MLP-based time-series model trained on hourly AQI data using lag features.
        Forecasts the next 24 hours of air quality for the selected city.
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 20 }}>
        <select
          value={city}
          onChange={e => setCity(e.target.value)}
          style={{
            background: 'var(--bg-surface)', border: '1px solid var(--border)',
            borderRadius: 8, padding: '9px 14px', color: 'var(--text-primary)',
            fontSize: 14, fontFamily: 'inherit', cursor: 'pointer',
          }}
        >
          {CITIES.map(c => (
            <option key={c} value={c}>{c.charAt(0).toUpperCase() + c.slice(1)}</option>
          ))}
        </select>
        <button
          className="btn btn-ghost"
          onClick={() => fetchForecast(city)}
          disabled={loading}
        >
          {loading ? <><span className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} /> Forecasting…</> : '🔄 Refresh'}
        </button>
      </div>

      {/* Stats row */}
      {hours.length > 0 && (
        <div style={{ display: 'flex', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
          {[
            { key: '24h Avg', val: avg, color: '#22c55e' },
            { key: 'Peak',    val: max, color: aqiColor(max) },
            { key: 'Lowest',  val: min, color: aqiColor(min) },
          ].map(s => (
            <div key={s.key} className="metric-chip" style={{ borderColor: s.color + '44' }}>
              <div className="value" style={{ color: s.color }}>{s.val}</div>
              <div className="key">{s.key} AQI — {aqiCategory(s.val)}</div>
            </div>
          ))}
        </div>
      )}

      {/* Chart */}
      <div className="card" style={{ marginBottom: 20 }}>
        {error
          ? <div style={{ color: '#fca5a5', fontSize: 13 }}>⚠️ {error}</div>
          : loading
            ? <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10 }}>
                <span className="spinner" /> <span style={{ color: 'var(--text-secondary)' }}>Generating forecast…</span>
              </div>
            : hours.length > 0
              ? <Line data={chartData} options={options} />
              : <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>No data available for this city in the dataset.</div>
        }
      </div>

      {/* Hourly breakdown table */}
      {hours.length > 0 && (
        <div className="card">
          <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 10 }}>🕐 Hourly Breakdown</div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(90px, 1fr))',
            gap: 8,
          }}>
            {hours.map((v, i) => (
              <div key={i} style={{
                background: aqiColor(v) + '15',
                border: `1px solid ${aqiColor(v)}44`,
                borderRadius: 8, padding: '8px 6px', textAlign: 'center',
              }}>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 3 }}>+{i + 1}h</div>
                <div style={{
                  fontFamily: 'JetBrains Mono', fontSize: 15, fontWeight: 700, color: aqiColor(v),
                }}>{Math.round(v)}</div>
                <div style={{ fontSize: 9, color: aqiColor(v), marginTop: 2 }}>{aqiCategory(v).split(' ')[0]}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
