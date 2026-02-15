export function render(container) {
  const cx = 375;
  const cy = 175;

  // Layers from innermost to outermost
  const layers = [
    { label: 'Model', sublabel: 'Core AI', r: 35, color: '#a855f7', delay: 0.3, techniques: [] },
    { label: 'Alignment', sublabel: '', r: 70, color: '#3b82f6', delay: 0.8, techniques: ['RLHF', 'Constitutional AI'] },
    { label: 'Guardrails', sublabel: '', r: 108, color: '#00d4aa', delay: 1.3, techniques: ['Input Filters', 'Output Policy'] },
    { label: 'Red Teaming', sublabel: '', r: 143, color: '#f59e0b', delay: 1.8, techniques: ['Adversarial Tests', 'Jailbreak Probes'] },
    { label: 'Regulation', sublabel: '', r: 168, color: '#ef4444', delay: 2.3, techniques: ['EU AI Act', 'Governance'] }
  ];

  let ringsSvg = '';
  let labelsSvg = '';
  let techniquesSvg = '';

  layers.forEach((layer, i) => {
    const delay = layer.delay.toFixed(2);

    // Ring
    ringsSvg += `
      <circle cx="${cx}" cy="${cy}" r="${layer.r}" fill="${layer.color}08"
        stroke="${layer.color}" stroke-width="${i === 0 ? 2.5 : 1.5}" stroke-opacity="0.6"
        class="sl-ring" style="animation-delay: ${delay}s;"/>
    `;

    if (i === 0) {
      // Model core: label in center
      labelsSvg += `
        <g class="sl-label" style="animation-delay: ${(layer.delay + 0.2).toFixed(2)}s;">
          <text x="${cx}" y="${cy - 5}" text-anchor="middle" dominant-baseline="central"
            fill="${layer.color}" font-family="var(--font-heading)" font-size="13" font-weight="700">
            ${layer.label}
          </text>
          <text x="${cx}" y="${cy + 12}" text-anchor="middle" dominant-baseline="central"
            fill="var(--text-dim)" font-family="var(--font-mono)" font-size="9">
            ${layer.sublabel}
          </text>
        </g>
      `;
    } else {
      // Labels on top of ring
      const labelAngle = [-0.35, 0.25, -0.2, 0.15][i - 1];
      const lx = cx + layer.r * Math.cos(labelAngle);
      const ly = cy - layer.r * Math.sin(labelAngle) - 2;

      labelsSvg += `
        <g class="sl-label" style="animation-delay: ${(layer.delay + 0.2).toFixed(2)}s;">
          <rect x="${lx - 38}" y="${ly - 10}" width="76" height="18" rx="9"
            fill="var(--bg-elevated)" stroke="${layer.color}60" stroke-width="1"/>
          <text x="${lx}" y="${ly + 1}" text-anchor="middle" dominant-baseline="central"
            fill="${layer.color}" font-family="var(--font-heading)" font-size="10" font-weight="600">
            ${layer.label}
          </text>
        </g>
      `;

      // Technique labels
      layer.techniques.forEach((tech, ti) => {
        const techAngle = Math.PI + (ti === 0 ? -0.5 : 0.5) + (i * 0.15);
        const tr = layer.r - 3;
        const tx = cx + tr * Math.cos(techAngle);
        const ty = cy - tr * Math.sin(techAngle);

        techniquesSvg += `
          <g class="sl-technique" style="animation-delay: ${(layer.delay + 0.5 + ti * 0.15).toFixed(2)}s;">
            <text x="${tx}" y="${ty}" text-anchor="middle" dominant-baseline="central"
              fill="${layer.color}" font-family="var(--font-mono)" font-size="8" opacity="0.7">
              ${tech}
            </text>
          </g>
        `;
      });
    }
  });

  // Attack arrows from outside
  const attacks = [
    { fromX: 680, fromY: 60, angle: -2.2, label: 'Prompt Injection' },
    { fromX: 700, fromY: 200, angle: -3.0, label: 'Adversarial Input' },
    { fromX: 620, fromY: 310, angle: -2.6, label: 'Data Poisoning' }
  ];

  let attacksSvg = '';
  attacks.forEach((atk, i) => {
    // Arrow comes from outside and stops at the outermost ring
    const outerR = layers[layers.length - 1].r;
    const dx = cx - atk.fromX;
    const dy = cy - atk.fromY;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const stopX = atk.fromX + dx * (1 - outerR / dist) + (dx / dist) * 8;
    const stopY = atk.fromY + dy * (1 - outerR / dist) + (dy / dist) * 8;

    const midX = (atk.fromX + stopX) / 2;
    const midY = (atk.fromY + stopY) / 2;

    const delay = (3.0 + i * 0.4).toFixed(2);

    attacksSvg += `
      <g class="sl-attack" style="animation-delay: ${delay}s;">
        <!-- Arrow line -->
        <line x1="${atk.fromX}" y1="${atk.fromY}" x2="${stopX}" y2="${stopY}"
          stroke="#ef4444" stroke-width="2" stroke-dasharray="6,4" opacity="0.7"
          marker-end="url(#sl-attack-arrow)"/>
        <!-- Label -->
        <text x="${(atk.fromX + midX) / 2 + 5}" y="${(atk.fromY + midY) / 2 - 6}" text-anchor="middle"
          fill="#ef4444" font-family="var(--font-mono)" font-size="8" opacity="0.8">
          ${atk.label}
        </text>
        <!-- Blocked X mark -->
        <g transform="translate(${stopX}, ${stopY})">
          <circle r="10" fill="#ef444430" stroke="#ef4444" stroke-width="1.5"/>
          <line x1="-5" y1="-5" x2="5" y2="5" stroke="#ef4444" stroke-width="2" stroke-linecap="round"/>
          <line x1="5" y1="-5" x2="-5" y2="5" stroke="#ef4444" stroke-width="2" stroke-linecap="round"/>
        </g>
      </g>
    `;
  });

  container.innerHTML = `
    <div style="text-align: center; width: 100%;">
      <svg viewBox="0 0 750 350" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
        <style>
          .sl-ring {
            opacity: 0;
            animation: slExpand 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
          }
          .sl-label {
            opacity: 0;
            animation: slFadeIn 0.5s ease forwards;
          }
          .sl-technique {
            opacity: 0;
            animation: slFadeIn 0.4s ease forwards;
          }
          .sl-attack {
            opacity: 0;
            animation: slSlideIn 0.6s ease forwards;
          }
          .sl-title {
            opacity: 0;
            animation: slFadeIn 0.6s ease forwards;
          }
          .sl-core-glow {
            animation: slPulseGlow 2.5s ease-in-out infinite;
            animation-delay: 0.5s;
          }
          .sl-shield-icon {
            opacity: 0;
            animation: slFadeIn 0.5s ease forwards;
            animation-delay: 2.6s;
          }
          @keyframes slExpand {
            0% { opacity: 0; transform: scale(0.3); }
            100% { opacity: 1; transform: scale(1); }
          }
          @keyframes slFadeIn {
            to { opacity: 1; }
          }
          @keyframes slSlideIn {
            0% { opacity: 0; transform: translateX(30px); }
            100% { opacity: 1; transform: translateX(0); }
          }
          @keyframes slPulseGlow {
            0%, 100% { opacity: 0.2; }
            50% { opacity: 0.5; }
          }
        </style>

        <defs>
          <filter id="slGlow">
            <feGaussianBlur in="SourceGraphic" stdDeviation="8"/>
          </filter>
          <marker id="sl-attack-arrow" markerWidth="8" markerHeight="8" refX="6" refY="4"
            orient="auto" markerUnits="strokeWidth">
            <path d="M1,1 L7,4 L1,7" fill="none" stroke="#ef4444" stroke-width="1.5"/>
          </marker>
        </defs>

        <!-- Title -->
        <text x="${cx}" y="25" text-anchor="middle" class="sl-title" style="animation-delay: 0s;"
          fill="var(--text-primary)" font-family="var(--font-heading)" font-size="16" font-weight="700">
          AI Safety: Defense in Depth
        </text>

        <!-- Core glow -->
        <circle cx="${cx}" cy="${cy}" r="40" fill="#a855f7" opacity="0.2"
          class="sl-core-glow" filter="url(#slGlow)"/>

        <!-- Concentric rings (drawn outer to inner for correct stacking) -->
        ${ringsSvg}

        <!-- Labels -->
        ${labelsSvg}

        <!-- Technique sub-labels -->
        ${techniquesSvg}

        <!-- Attack arrows -->
        ${attacksSvg}

        <!-- Left-side legend -->
        <g class="sl-shield-icon">
          <text x="70" y="${cy - 40}" text-anchor="start"
            fill="var(--text-secondary)" font-family="var(--font-heading)" font-size="11" font-weight="600">
            Defense Layers:
          </text>
          ${layers.map((l, i) => `
            <circle cx="78" cy="${cy - 18 + i * 22}" r="5" fill="${l.color}40" stroke="${l.color}" stroke-width="1.5"/>
            <text x="90" y="${cy - 14 + i * 22}" fill="${l.color}" font-family="var(--font-mono)" font-size="9">${l.label}</text>
          `).join('')}
        </g>

        <!-- Bottom annotation -->
        <text x="${cx}" y="338" text-anchor="middle" class="sl-title" style="animation-delay: 3.5s;"
          fill="var(--text-dim)" font-family="var(--font-mono)" font-size="10">
          Multiple layers of protection \u2014 no single point of failure
        </text>
      </svg>
      <p style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-dim); margin-top: var(--space-3);">
        AI Safety Layers \u2014 Concentric defense model
      </p>
    </div>
  `;
}
