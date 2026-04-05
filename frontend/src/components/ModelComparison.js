import React from 'react';
import {
  Chart as ChartJS, CategoryScale, LinearScale, BarElement,
  Tooltip, Legend, Title,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { aqiColor } from '../App';

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend, Title);

export default function ModelComparison({ result }) {
  if (!result) return null;

  const models = [
    { label: 'Fuzzy Logic', value: result.Fuzzy, color: '#00d2ff' },
    { label: 'Neural Net',  value: result.NN,    color: '#a855f7' },
    { label: 'ANFIS',       value: result.ANFIS, color: '#ff6b35' },
  ];

  const data = {
    labels: models.map(m => m.label),
    datasets: [{
      label: 'Predicted AQI',
      data:  models.map(m => Math.round(m.value)),
      backgroundColor: models.map(m => m.color + 'cc'),
      borderColor:     models.map(m => m.color),
      borderWidth: 2,
      borderRadius: 8,
      borderSkipped: false,
    }],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: ctx => `AQI: ${ctx.parsed.y}`,
          afterLabel: ctx => {
            const v = ctx.parsed.y;
            if (v <= 50)  return 'Category: Good';
            if (v <= 100) return 'Category: Moderate';
            if (v <= 150) return 'Category: Unhealthy (Sensitive)';
            if (v <= 200) return 'Category: Unhealthy';
            if (v <= 300) return 'Category: Very Unhealthy';
            return 'Category: Hazardous';
          },
        },
        backgroundColor: '#161b22',
        borderColor: 'rgba(255,255,255,0.1)',
        borderWidth: 1,
        titleColor: '#e6edf3',
        bodyColor: '#8b949e',
        padding: 12,
      },
    },
    scales: {
      x: {
        grid: { color: 'rgba(255,255,255,0.04)' },
        ticks: { color: '#8b949e', font: { family: 'Inter', size: 12 } },
      },
      y: {
        grid: { color: 'rgba(255,255,255,0.04)' },
        ticks: { color: '#8b949e', font: { family: 'JetBrains Mono', size: 11 } },
        beginAtZero: true,
      },
    },
  };

  return (
    <div className="card">
      <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 4 }}>Model Comparison</div>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 16 }}>
        All three soft computing models predict on the same input
      </div>
      <Bar data={data} options={options} />

      {/* Delta analysis */}
      <div style={{ marginTop: 16, padding: '12px', background: 'var(--bg-base)', borderRadius: 8 }}>
        <div className="label" style={{ marginBottom: 6 }}>Variance between models</div>
        <div style={{ fontFamily: 'JetBrains Mono', fontSize: 13, color: 'var(--text-secondary)' }}>
          Max Δ :{' '}
          <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
            {Math.round(Math.max(...models.map(m => m.value)) - Math.min(...models.map(m => m.value)))} AQI units
          </span>
          {' '} — measures how much the three approaches disagree.
        </div>
      </div>
    </div>
  );
}
