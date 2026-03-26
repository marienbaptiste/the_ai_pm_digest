export function render(container) {
  const cx = 400;
  const cy = 220;
  const r = 100;
  const ringWidth = 50;
  const innerR = r - ringWidth / 2;
  const outerR = r + ringWidth / 2;

  const segments = [
    { label: 'Developers', color: '#7EB8DA', growth: '+SDKs' },
    { label: 'Apps', color: '#C4A7E7', growth: '+plugins' },
    { label: 'Users', color: '#9CCFA4', growth: '+adoption' },
    { label: 'Feedback', color: '#F0B429', growth: '+insights' }
  ];

  let ringPaths = '';
  let ringLabels = '';
  let arrowsSvg = '';

  segments.forEach((seg, i) => {
    const startDeg = i * 90 - 90;
    const endDeg = startDeg + 90;
    const startA = startDeg * Math.PI / 180;
    const endA = endDeg * Math.PI / 180;

    const x1o = cx + outerR * Math.cos(startA);
    const y1o = cy + outerR * Math.sin(startA);
    const x2o = cx + outerR * Math.cos(endA);
    const y2o = cy + outerR * Math.sin(endA);
    const x1i = cx + innerR * Math.cos(endA);
    const y1i = cy + innerR * Math.sin(endA);
    const x2i = cx + innerR * Math.cos(startA);
    const y2i = cy + innerR * Math.sin(startA);

    ringPaths += `
      <path d="M${x1o},${y1o} A${outerR},${outerR} 0 0,1 ${x2o},${y2o}
               L${x1i},${y1i} A${innerR},${innerR} 0 0,0 ${x2i},${y2i} Z"
        fill="${seg.color}15" stroke="${seg.color}50" stroke-width="1"
        class="fw-seg" style="animation-delay: ${(0.2 + i * 0.2).toFixed(2)}s;"/>
    `;

    // Label on the ring arc
    const midA = (startDeg + 45) * Math.PI / 180;
    const lx = cx + r * Math.cos(midA);
    const ly = cy + r * Math.sin(midA);
    const rotDeg = startDeg + 45 + (i >= 2 ? 180 : 0);

    ringLabels += `
      <text x="${lx}" y="${ly}" text-anchor="middle" dominant-baseline="central"
        fill="${seg.color}" font-family="var(--font-heading)" font-size="11" font-weight="600"
        transform="rotate(${rotDeg}, ${lx}, ${ly})"
        class="fw-lbl" style="animation-delay: ${(0.5 + i * 0.2).toFixed(2)}s;">${seg.label}</text>
    `;

    // Arrow arc outside the ring
    const arrStart = (endDeg - 15) * Math.PI / 180;
    const arrEnd = (endDeg + 15) * Math.PI / 180;
    const arrR = outerR + 10;
    const ax1 = cx + arrR * Math.cos(arrStart);
    const ay1 = cy + arrR * Math.sin(arrStart);
    const ax2 = cx + arrR * Math.cos(arrEnd);
    const ay2 = cy + arrR * Math.sin(arrEnd);

    arrowsSvg += `
      <path d="M${ax1},${ay1} A${arrR},${arrR} 0 0,1 ${ax2},${ay2}"
        fill="none" stroke="${seg.color}" stroke-width="2" stroke-opacity="0.5"
        marker-end="url(#fw-a${i})"
        class="fw-arr" style="animation-delay: ${(1.2 + i * 0.15).toFixed(2)}s;"/>
      <defs>
        <marker id="fw-a${i}" markerWidth="8" markerHeight="8" refX="4" refY="4"
          orient="auto"><path d="M1,1 L7,4 L1,7" fill="none" stroke="${seg.color}" stroke-width="1.5"/></marker>
      </defs>
    `;
  });

  // External detail cards
  const cards = [
    { label: 'Developers', sub: 'Build on the platform', color: '#7EB8DA', growth: '+SDKs', x: 650, y: 58 },
    { label: 'Apps & Integrations', sub: 'Expand the ecosystem', color: '#C4A7E7', growth: '+plugins', x: 680, y: 370 },
    { label: 'Users', sub: 'Adopt & engage', color: '#9CCFA4', growth: '+adoption', x: 120, y: 370 },
    { label: 'Data & Feedback', sub: 'Improve the platform', color: '#F0B429', growth: '+insights', x: 135, y: 58 }
  ];

  // Connector anchor points on ring
  const anchors = [
    { x: cx + outerR, y: cy - outerR * 0.4 },
    { x: cx + outerR * 0.4, y: cy + outerR },
    { x: cx - outerR * 0.4, y: cy + outerR },
    { x: cx - outerR, y: cy - outerR * 0.4 }
  ];

  let cardsSvg = '';
  cards.forEach((c, i) => {
    const anchor = anchors[i];
    const delay = (1.6 + i * 0.2).toFixed(2);
    cardsSvg += `
      <g class="fw-card" style="animation-delay: ${delay}s;">
        <line x1="${anchor.x}" y1="${anchor.y}" x2="${c.x}" y2="${c.y + 8}"
          stroke="${c.color}" stroke-width="0.8" stroke-dasharray="3 3" opacity="0.3"/>
        <text x="${c.x}" y="${c.y}" text-anchor="middle"
          fill="${c.color}" font-family="var(--font-heading)" font-size="13" font-weight="700">${c.label}</text>
        <text x="${c.x}" y="${c.y + 16}" text-anchor="middle"
          fill="var(--text-dim)" font-family="var(--font-mono)" font-size="9">${c.sub}</text>
        <rect x="${c.x - 24}" y="${c.y + 22}" width="48" height="16" rx="8"
          fill="${c.color}12" stroke="${c.color}35" stroke-width="1"/>
        <text x="${c.x}" y="${c.y + 33}" text-anchor="middle"
          fill="${c.color}" font-family="var(--font-mono)" font-size="8">${c.growth}</text>
      </g>
    `;
  });

  container.innerHTML = `
    <div style="text-align: center; width: 100%;">
      <svg viewBox="0 0 800 440" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
        <style>
          .fw-seg { opacity: 0; animation: fwIn 0.7s ease forwards; }
          .fw-lbl { opacity: 0; animation: fwIn 0.5s ease forwards; }
          .fw-arr { opacity: 0; animation: fwIn 0.5s ease forwards; }
          .fw-card { opacity: 0; animation: fwIn 0.5s ease forwards; }
          .fw-ctr { opacity: 0; animation: fwIn 0.8s ease forwards; animation-delay: 0.1s; }
          @keyframes fwIn { to { opacity: 1; } }
          @keyframes fwPulse { 0%, 100% { opacity: 0.08; } 50% { opacity: 0.25; } }
        </style>

        <defs>
          <filter id="fwG"><feGaussianBlur in="SourceGraphic" stdDeviation="4"/></filter>
        </defs>

        <!-- Title -->
        <text x="${cx}" y="26" text-anchor="middle"
          fill="var(--text-primary)" font-family="var(--font-heading)" font-size="16" font-weight="700"
          class="fw-ctr">Platform Flywheel Strategy</text>

        <!-- Ring -->
        ${ringPaths}
        ${ringLabels}
        ${arrowsSvg}

        <!-- Center -->
        <circle cx="${cx}" cy="${cy}" r="48" fill="var(--bg-elevated)" stroke="var(--border-medium)" stroke-width="1.5" class="fw-ctr"/>
        <circle cx="${cx}" cy="${cy}" r="55" fill="none" stroke="#9CCFA4" stroke-width="1"
          style="animation: fwPulse 3s ease-in-out infinite;" filter="url(#fwG)"/>
        <g class="fw-ctr">
          <text x="${cx}" y="${cy - 10}" text-anchor="middle" dominant-baseline="central"
            fill="var(--text-primary)" font-family="var(--font-heading)" font-size="14" font-weight="700">Platform</text>
          <text x="${cx}" y="${cy + 8}" text-anchor="middle" dominant-baseline="central"
            fill="#9CCFA4" font-family="var(--font-heading)" font-size="14" font-weight="700">Flywheel</text>
          <text x="${cx}" y="${cy + 24}" text-anchor="middle" dominant-baseline="central"
            fill="var(--text-dim)" font-family="var(--font-mono)" font-size="8">self-reinforcing</text>
        </g>

        <!-- Detail cards -->
        ${cardsSvg}

        <!-- Subtitle -->
        <text x="${cx}" y="428" text-anchor="middle"
          fill="var(--text-dim)" font-family="var(--font-mono)" font-size="10"
          class="fw-card" style="animation-delay: 2.2s;">
          Each stage reinforces the next - compounding value over time
        </text>
      </svg>
      <p style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-dim); margin-top: var(--space-3);">
        Platform Flywheel - Self-reinforcing growth cycle
      </p>
    </div>
  `;
}
