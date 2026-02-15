export function render(container) {
  const cx = 375;
  const cy = 185;
  const r = 120;

  // Segment definitions: angle ranges (0 = top, clockwise)
  const segments = [
    { label: 'Developers', icon: '\u2328', color: '#7EB8DA', angle: -45, growth: '+SDKs' },
    { label: 'Apps &\nIntegrations', icon: '\u2699', color: '#C4A7E7', angle: 45, growth: '+plugins' },
    { label: 'Users', icon: '\u263A', color: '#9CCFA4', angle: 135, growth: '+adoption' },
    { label: 'Data &\nFeedback', icon: '\u2605', color: '#F0B429', angle: 225, growth: '+insights' }
  ];

  // Build arc segments
  let segmentsSvg = '';
  let labelsSvg = '';
  let arrowsSvg = '';
  let growthSvg = '';

  segments.forEach((seg, i) => {
    const startAngle = (i * 90 - 45) * Math.PI / 180;
    const endAngle = (i * 90 + 45) * Math.PI / 180;
    const midAngle = (i * 90) * Math.PI / 180;

    const innerR = r - 28;
    const outerR = r + 28;

    // Arc path for segment
    const x1o = cx + outerR * Math.sin(startAngle);
    const y1o = cy - outerR * Math.cos(startAngle);
    const x2o = cx + outerR * Math.sin(endAngle);
    const y2o = cy - outerR * Math.cos(endAngle);
    const x1i = cx + innerR * Math.sin(endAngle);
    const y1i = cy - innerR * Math.cos(endAngle);
    const x2i = cx + innerR * Math.sin(startAngle);
    const y2i = cy - innerR * Math.cos(startAngle);

    const delay = (0.3 + i * 0.4).toFixed(2);

    segmentsSvg += `
      <path d="M${x1o},${y1o} A${outerR},${outerR} 0 0,1 ${x2o},${y2o}
               L${x1i},${y1i} A${innerR},${innerR} 0 0,0 ${x2i},${y2i} Z"
        fill="${seg.color}20" stroke="${seg.color}" stroke-width="2"
        class="fw-segment" style="animation-delay: ${delay}s;"/>
    `;

    // Label position (on the ring)
    const labelR = r;
    const lx = cx + labelR * Math.sin(midAngle);
    const ly = cy - labelR * Math.cos(midAngle);

    // Icon position (slightly inward)
    const iconR = r - 8;
    const ix = cx + iconR * Math.sin(midAngle);
    const iy = cy - iconR * Math.cos(midAngle);

    // Text position (slightly outward)
    const textR = r + 6;
    const tx = cx + textR * Math.sin(midAngle);
    const ty = cy - textR * Math.cos(midAngle);

    const lines = seg.label.split('\n');

    labelsSvg += `
      <g class="fw-label" style="animation-delay: ${(0.5 + i * 0.4).toFixed(2)}s;">
        <text x="${ix}" y="${iy}" text-anchor="middle" dominant-baseline="central"
          fill="${seg.color}" font-size="18">${seg.icon}</text>
        ${lines.map((line, li) => `
          <text x="${tx}" y="${ty + (li - (lines.length - 1) / 2) * 13}" text-anchor="middle" dominant-baseline="central"
            fill="${seg.color}" font-family="var(--font-heading)" font-size="11" font-weight="600">${line}</text>
        `).join('')}
      </g>
    `;

    // Growth indicators (outside the ring)
    const growthR = r + 55;
    const gx = cx + growthR * Math.sin(midAngle);
    const gy = cy - growthR * Math.cos(midAngle);

    growthSvg += `
      <g class="fw-growth" style="animation-delay: ${(2.0 + i * 0.3).toFixed(2)}s;">
        <rect x="${gx - 24}" y="${gy - 9}" width="48" height="18" rx="9"
          fill="${seg.color}15" stroke="${seg.color}50" stroke-width="1"/>
        <text x="${gx}" y="${gy + 1}" text-anchor="middle" dominant-baseline="central"
          fill="${seg.color}" font-family="var(--font-mono)" font-size="8" opacity="0.8">${seg.growth}</text>
      </g>
    `;

    // Curved arrows between segments (at the outer edge)
    const arrowStartAngle = (i * 90 + 30) * Math.PI / 180;
    const arrowEndAngle = (i * 90 + 60) * Math.PI / 180;
    const arrowR = outerR + 10;

    const ax1 = cx + arrowR * Math.sin(arrowStartAngle);
    const ay1 = cy - arrowR * Math.cos(arrowStartAngle);
    const ax2 = cx + arrowR * Math.sin(arrowEndAngle);
    const ay2 = cy - arrowR * Math.cos(arrowEndAngle);

    // Arrowhead direction (tangent at end)
    const tangentAngle = arrowEndAngle + Math.PI / 2;
    const ahSize = 6;
    const ahx1 = ax2 - ahSize * Math.cos(tangentAngle - 0.4);
    const ahy1 = ay2 - ahSize * Math.sin(tangentAngle - 0.4);
    const ahx2 = ax2 - ahSize * Math.cos(tangentAngle + 0.4);
    const ahy2 = ay2 - ahSize * Math.sin(tangentAngle + 0.4);

    arrowsSvg += `
      <g class="fw-arrow" style="animation-delay: ${(1.8 + i * 0.2).toFixed(2)}s;">
        <path d="M${ax1},${ay1} A${arrowR},${arrowR} 0 0,1 ${ax2},${ay2}"
          fill="none" stroke="${seg.color}" stroke-width="2" stroke-opacity="0.6"
          marker-end="url(#fw-arrowhead-${i})"/>
        <defs>
          <marker id="fw-arrowhead-${i}" markerWidth="8" markerHeight="8" refX="4" refY="4"
            orient="auto" markerUnits="strokeWidth">
            <path d="M1,1 L7,4 L1,7" fill="none" stroke="${seg.color}" stroke-width="1.5"/>
          </marker>
        </defs>
      </g>
    `;
  });

  container.innerHTML = `
    <div style="text-align: center; width: 100%;">
      <svg viewBox="0 0 750 350" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
        <style>
          .fw-segment {
            opacity: 0;
            animation: fwFadeIn 0.8s ease forwards;
          }
          .fw-label {
            opacity: 0;
            animation: fwFadeIn 0.6s ease forwards;
          }
          .fw-arrow {
            opacity: 0;
            animation: fwFadeIn 0.5s ease forwards;
          }
          .fw-growth {
            opacity: 0;
            animation: fwPopIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
          }
          .fw-center-text {
            opacity: 0;
            animation: fwFadeIn 0.8s ease forwards;
            animation-delay: 0.1s;
          }
          .fw-ring-group {
            animation: fwSpin 30s linear infinite;
            animation-delay: 2.5s;
            transform-origin: ${cx}px ${cy}px;
          }
          .fw-title {
            opacity: 0;
            animation: fwFadeIn 0.6s ease forwards;
          }
          .fw-pulse-ring {
            animation: fwPulse 3s ease-in-out infinite;
            animation-delay: 2.5s;
          }
          .fw-dot {
            animation: fwDotPulse 2s ease-in-out infinite;
          }
          @keyframes fwFadeIn {
            to { opacity: 1; }
          }
          @keyframes fwPopIn {
            0% { opacity: 0; transform: scale(0.5); }
            100% { opacity: 1; transform: scale(1); }
          }
          @keyframes fwSpin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
          @keyframes fwPulse {
            0%, 100% { opacity: 0.15; }
            50% { opacity: 0.35; }
          }
          @keyframes fwDotPulse {
            0%, 100% { opacity: 0.4; }
            50% { opacity: 1; }
          }
        </style>

        <defs>
          <filter id="fwGlow">
            <feGaussianBlur in="SourceGraphic" stdDeviation="6"/>
          </filter>
        </defs>

        <!-- Title -->
        <text x="${cx}" y="28" text-anchor="middle" class="fw-title" style="animation-delay: 0s;"
          fill="var(--text-primary)" font-family="var(--font-heading)" font-size="16" font-weight="700">
          Platform Flywheel Strategy
        </text>

        <!-- Rotating ring group -->
        <g class="fw-ring-group">
          <!-- Segments -->
          ${segmentsSvg}

          <!-- Curved arrows -->
          ${arrowsSvg}

          <!-- Labels on ring -->
          ${labelsSvg}
        </g>

        <!-- Center content (does not rotate) -->
        <circle cx="${cx}" cy="${cy}" r="50" fill="var(--bg-elevated)" stroke="var(--border-medium)" stroke-width="1.5"
          class="fw-center-text"/>
        <circle cx="${cx}" cy="${cy}" r="55" fill="none" stroke="#9CCFA4" stroke-width="1" opacity="0.15"
          class="fw-pulse-ring" filter="url(#fwGlow)"/>

        <g class="fw-center-text">
          <text x="${cx}" y="${cy - 10}" text-anchor="middle" dominant-baseline="central"
            fill="var(--text-primary)" font-family="var(--font-heading)" font-size="13" font-weight="700">
            Platform
          </text>
          <text x="${cx}" y="${cy + 8}" text-anchor="middle" dominant-baseline="central"
            fill="#9CCFA4" font-family="var(--font-heading)" font-size="13" font-weight="700">
            Flywheel
          </text>
          <text x="${cx}" y="${cy + 25}" text-anchor="middle" dominant-baseline="central"
            fill="var(--text-dim)" font-family="var(--font-mono)" font-size="8">
            self-reinforcing
          </text>
        </g>

        <!-- Growth indicators (outside ring, non-rotating) -->
        ${growthSvg}

        <!-- Corner decorations: small flow dots -->
        <circle cx="${cx}" cy="${cy - r - 50}" r="3" fill="#7EB8DA" class="fw-dot" style="animation-delay: 0s;"/>
        <circle cx="${cx + r + 50}" cy="${cy}" r="3" fill="#C4A7E7" class="fw-dot" style="animation-delay: 0.5s;"/>
        <circle cx="${cx}" cy="${cy + r + 50}" r="3" fill="#9CCFA4" class="fw-dot" style="animation-delay: 1s;"/>
        <circle cx="${cx - r - 50}" cy="${cy}" r="3" fill="#F0B429" class="fw-dot" style="animation-delay: 1.5s;"/>

        <!-- Subtitle -->
        <text x="${cx}" y="335" text-anchor="middle" class="fw-title" style="animation-delay: 2.5s;"
          fill="var(--text-dim)" font-family="var(--font-mono)" font-size="10">
          Each stage reinforces the next \u2014 compounding value over time
        </text>
      </svg>
      <p style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-dim); margin-top: var(--space-3);">
        Platform Flywheel \u2014 Self-reinforcing growth cycle
      </p>
    </div>
  `;
}
