export function render(container) {
  const W = 750, H = 400;
  const cx = W / 2, cy = 195;

  // Five frontier pillars arranged in a horizontal layout
  const pillars = [
    { label: 'MoE', sub: 'Mixture of\nExperts', x: 95, color: '#9CCFA4', icon: 'M22 12H2M8 6h8M6 18h12' },
    { label: 'SSM', sub: 'State Space\nModels', x: 245, color: '#7EB8DA', icon: 'M4 20L8 4l4 10 4-6 4 16' },
    { label: 'Reasoning', sub: 'Test-Time\nCompute', x: 395, color: '#C4A7E7', icon: 'M12 2a10 10 0 100 20 10 10 0 000-20zm0 6v4l3 3' },
    { label: 'Agents', sub: 'Tool Use &\nMulti-Agent', x: 545, color: '#F0B429', icon: 'M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z' },
    { label: 'Frontier', sub: 'Video, World\nModels', x: 675, color: '#E8553A', icon: 'M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z' },
  ];

  // Connection arrows between pillars
  let connectionsSvg = '';
  for (let i = 0; i < pillars.length - 1; i++) {
    const from = pillars[i];
    const to = pillars[i + 1];
    const midX = (from.x + to.x) / 2;
    const delay = (1.5 + i * 0.2).toFixed(2);
    connectionsSvg += `
      <g class="ft-conn" style="animation-delay: ${delay}s;">
        <line x1="${from.x + 50}" y1="${cy}" x2="${to.x - 50}" y2="${cy}"
          stroke="var(--border-subtle)" stroke-width="1" stroke-dasharray="4,6"/>
        <polygon points="${to.x - 52},${cy - 4} ${to.x - 44},${cy} ${to.x - 52},${cy + 4}"
          fill="var(--text-dim)" opacity="0.5"/>
      </g>
    `;
  }

  // Build pillar nodes
  let pillarsSvg = '';
  pillars.forEach((p, i) => {
    const delay = (0.6 + i * 0.2).toFixed(2);
    const subLines = p.sub.split('\n');
    pillarsSvg += `
      <g class="ft-pillar" style="animation-delay: ${delay}s;">
        <!-- Glow -->
        <circle cx="${p.x}" cy="${cy}" r="48" fill="${p.color}" opacity="0.06"
          filter="url(#ftGlow)"/>
        <!-- Outer ring -->
        <circle cx="${p.x}" cy="${cy}" r="42" fill="none" stroke="${p.color}"
          stroke-width="1.5" stroke-opacity="0.3" stroke-dasharray="6,4"/>
        <!-- Inner circle -->
        <circle cx="${p.x}" cy="${cy}" r="34" fill="var(--bg-elevated)"
          stroke="${p.color}" stroke-width="2"/>
        <!-- Icon -->
        <g transform="translate(${p.x - 12}, ${cy - 18})">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none"
            stroke="${p.color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="${p.icon}"/>
          </svg>
        </g>
        <!-- Label -->
        <text x="${p.x}" y="${cy + 10}" text-anchor="middle" dominant-baseline="central"
          fill="${p.color}" font-family="var(--font-heading)" font-size="11" font-weight="700">
          ${p.label}
        </text>
        <!-- Sub-label -->
        ${subLines.map((line, li) => `
          <text x="${p.x}" y="${cy + 58 + li * 14}" text-anchor="middle"
            fill="var(--text-dim)" font-family="var(--font-mono)" font-size="9">
            ${line}
          </text>
        `).join('')}
      </g>
    `;
  });

  // Timeline bar at bottom showing evolution
  const timelineY = 340;
  const milestones = [
    { year: '2022', label: 'Chinchilla Laws', x: 75 },
    { year: '2023', label: 'GPT-4 / Gemini', x: 200 },
    { year: '2024', label: 'MoE + Agents', x: 375 },
    { year: '2025', label: 'Reasoning Era', x: 525 },
    { year: '2026', label: 'Frontier', x: 675 },
  ];

  let timelineSvg = `
    <g class="ft-timeline" style="animation-delay: 2.0s;">
      <line x1="55" y1="${timelineY}" x2="695" y2="${timelineY}"
        stroke="var(--border-subtle)" stroke-width="1.5"/>
      ${milestones.map((m, i) => `
        <g style="animation-delay: ${(2.2 + i * 0.15).toFixed(2)}s;" class="ft-milestone">
          <circle cx="${m.x}" cy="${timelineY}" r="4" fill="${pillars[i].color}" stroke="var(--bg-surface)" stroke-width="2"/>
          <text x="${m.x}" y="${timelineY - 12}" text-anchor="middle"
            fill="${pillars[i].color}" font-family="var(--font-mono)" font-size="10" font-weight="600">
            ${m.year}
          </text>
          <text x="${m.x}" y="${timelineY + 16}" text-anchor="middle"
            fill="var(--text-dim)" font-family="var(--font-mono)" font-size="8">
            ${m.label}
          </text>
        </g>
      `).join('')}
    </g>
  `;

  container.innerHTML = `
    <div style="text-align: center; width: 100%;">
      <svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
        <style>
          .ft-title { opacity: 0; animation: ftFadeIn 0.6s ease forwards; }
          .ft-pillar { opacity: 0; animation: ftScaleIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) forwards; }
          .ft-conn { opacity: 0; animation: ftFadeIn 0.5s ease forwards; }
          .ft-timeline { opacity: 0; animation: ftFadeIn 0.6s ease forwards; }
          .ft-milestone { opacity: 0; animation: ftScaleIn 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) forwards; }

          @keyframes ftFadeIn { to { opacity: 1; } }
          @keyframes ftScaleIn {
            0% { opacity: 0; transform: scale(0.5); }
            100% { opacity: 1; transform: scale(1); }
          }
        </style>

        <defs>
          <filter id="ftGlow">
            <feGaussianBlur in="SourceGraphic" stdDeviation="8"/>
          </filter>
        </defs>

        <!-- Title -->
        <text x="${cx}" y="28" text-anchor="middle" class="ft-title" style="animation-delay: 0s;"
          fill="var(--text-dim)" font-family="var(--font-mono)" font-size="11" letter-spacing="2">
          Frontier AI Landscape \u2014 2022\u20132026
        </text>

        <!-- Subtitle -->
        <text x="${cx}" y="50" text-anchor="middle" class="ft-title" style="animation-delay: 0.2s;"
          fill="var(--text-secondary)" font-family="var(--font-heading)" font-size="14" font-weight="600">
          Five Paradigm Shifts Defining Next-Generation AI
        </text>

        <!-- Connection arrows -->
        ${connectionsSvg}

        <!-- Pillar nodes -->
        ${pillarsSvg}

        <!-- Timeline -->
        ${timelineSvg}

        <!-- Bottom annotation -->
        <text x="${cx}" y="${H - 10}" text-anchor="middle" class="ft-title" style="animation-delay: 3.0s;"
          fill="var(--text-dim)" font-family="var(--font-mono)" font-size="9">
          Each shift compounds on the last \u2014 modern frontier models combine all five
        </text>
      </svg>
      <p style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-dim); margin-top: var(--space-3);">
        Frontier AI Architecture Landscape
      </p>
    </div>
  `;
}
