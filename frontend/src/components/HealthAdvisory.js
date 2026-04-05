import React from 'react';
import { aqiColor, aqiCategory } from '../App';

const ADVISORIES = [
  {
    max: 50,
    icon: '😊', label: 'Good',
    message: 'Air quality is satisfactory. Ideal for outdoor activities.',
    tips: ['Enjoy outdoor exercise', 'Open windows for ventilation', 'No restrictions needed'],
    color: '#22c55e',
  },
  {
    max: 100,
    icon: '😐', label: 'Moderate',
    message: 'Acceptable. Unusually sensitive individuals may experience symptoms.',
    tips: ['Sensitive groups: limit prolonged outdoor exertion', 'Others: normal activities OK'],
    color: '#eab308',
  },
  {
    max: 150,
    icon: '😷', label: 'Unhealthy for Sensitive Groups',
    message: 'Children, elderly, and those with respiratory conditions at risk.',
    tips: ['Wear N95 mask if outdoors', 'Keep windows closed', 'Use air purifier indoors'],
    color: '#f97316',
  },
  {
    max: 200,
    icon: '🚫', label: 'Unhealthy',
    message: 'Everyone may experience health effects.',
    tips: ['Avoid outdoor exercise', 'Wear mask outdoors', 'Keep children indoors', 'Consult doctor if wheezing'],
    color: '#ef4444',
  },
  {
    max: 300,
    icon: '☠️', label: 'Very Unhealthy',
    message: 'Health alert — serious effects for everyone.',
    tips: ['Stay indoors', 'Run air purifiers', 'Seal windows/doors', 'Seek medical attention if symptomatic'],
    color: '#a855f7',
  },
  {
    max: Infinity,
    icon: '💀', label: 'Hazardous',
    message: 'Emergency conditions. Entire population is affected.',
    tips: ['Do NOT go outside', 'Evacuate if possible', 'Emergency medical services on standby'],
    color: '#7c3aed',
  },
];

export default function HealthAdvisory({ aqi, explanation }) {
  if (!aqi) return null;
  const adv = ADVISORIES.find(a => aqi <= a.max) || ADVISORIES[ADVISORIES.length - 1];

  return (
    <div className="card fade-in" style={{ borderColor: adv.color + '44' }}>
      <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start', marginBottom: 14 }}>
        <div style={{ fontSize: 28 }}>{adv.icon}</div>
        <div>
          <div style={{ fontWeight: 700, fontSize: 15, color: adv.color }}>{adv.label}</div>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 3 }}>{adv.message}</div>
        </div>
      </div>

      <div style={{ marginBottom: 14 }}>
        {adv.tips.map((t, i) => (
          <div key={i} style={{
            display: 'flex', gap: 8, alignItems: 'flex-start',
            padding: '5px 0', borderBottom: i < adv.tips.length - 1 ? '1px solid var(--border)' : 'none',
            fontSize: 13, color: 'var(--text-primary)',
          }}>
            <span style={{ color: adv.color, marginTop: 1 }}>›</span>
            {t}
          </div>
        ))}
      </div>

      {/* Pollutant levels */}
      {explanation && (
        <div>
          <div className="label" style={{ marginBottom: 8 }}>Pollutant Levels</div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {Object.entries(explanation).map(([k, v]) => (
              <span key={k} className="badge" style={{
                background: v === 'High' ? '#ef444422' : v === 'Medium' ? '#f9731622' : '#22c55e22',
                color:      v === 'High' ? '#ef4444'   : v === 'Medium' ? '#f97316'   : '#22c55e',
              }}>
                {k}: {v}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
