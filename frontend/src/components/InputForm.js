import React from 'react';

const FIELDS = [
  { key: 'PM2.5', label: 'PM2.5',  unit: 'µg/m³', info: 'Fine particulate matter < 2.5 µm' },
  { key: 'PM10',  label: 'PM10',   unit: 'µg/m³', info: 'Coarse particulate matter < 10 µm' },
  { key: 'NO2',   label: 'NO2',    unit: 'ppb',   info: 'Nitrogen Dioxide' },
  { key: 'CO',    label: 'CO',     unit: 'ppm',   info: 'Carbon Monoxide' },
];

const CITIES = ['mumbai', 'delhi', 'bangalore', 'kolkata', 'chennai', 'hyderabad', 'pune', 'ahmedabad'];

export default function InputForm({ form, setForm, onPredict, onLive, liveCity, setLiveCity, loading }) {
  const handleChange = (key, val) => {
    const n = parseFloat(val);
    setForm(f => ({ ...f, [key]: isNaN(n) ? 0 : n }));
  };

  return (
    <div className="card">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 18 }}>
        <div>
          <div style={{ fontWeight: 700, fontSize: 15 }}>Pollutant Inputs</div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>
            Enter values manually or auto-fill with live data
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <select
            value={liveCity}
            onChange={e => setLiveCity(e.target.value)}
            style={{
              background: 'var(--bg-base)', border: '1px solid var(--border)',
              borderRadius: 7, padding: '7px 10px', color: 'var(--text-primary)',
              fontSize: 13, fontFamily: 'inherit', cursor: 'pointer',
            }}
          >
            {CITIES.map(c => (
              <option key={c} value={c}>{c.charAt(0).toUpperCase() + c.slice(1)}</option>
            ))}
          </select>
          <button className="btn btn-ghost" onClick={onLive} disabled={loading} style={{ fontSize: 13, padding: '7px 14px' }}>
            📡 Live
          </button>
        </div>
      </div>

      <div className="grid-2" style={{ marginBottom: 18 }}>
        {FIELDS.map(f => (
          <div key={f.key}>
            <div className="label" style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>{f.label}</span>
              <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>{f.unit}</span>
            </div>
            <input
              className="input-field"
              type="number"
              min="0"
              step="0.1"
              value={form[f.key]}
              onChange={e => handleChange(f.key, e.target.value)}
              placeholder={f.info}
            />
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3 }}>{f.info}</div>
          </div>
        ))}
      </div>

      <button
        className="btn btn-primary"
        style={{ width: '100%', justifyContent: 'center', padding: '12px' }}
        onClick={onPredict}
        disabled={loading}
      >
        {loading
          ? <><span className="spinner" /> Running Models…</>
          : '⚡ Run All Three Models'}
      </button>
    </div>
  );
}
