export function render(container) {
  const cx = 375;
  const cy = 190;
  const innerOrbitR = 90;
  const outerOrbitR = 150;

  const products = [
    { label: 'Android', angle: 0 },
    { label: 'iOS', angle: 90 },
    { label: 'Web', angle: 180 },
    { label: 'API', angle: 270 }
  ];

  const capabilities = [
    { label: 'Multimodal', angle: 0, speed: 45 },
    { label: 'Long Context', angle: 72, speed: 55 },
    { label: 'Code Gen', angle: 144, speed: 40 },
    { label: 'Reasoning', angle: 216, speed: 50 },
    { label: 'Vision', angle: 288, speed: 60 }
  ];

  // Build product nodes (inner orbit)
  let productsSvg = '';
  products.forEach((p, i) => {
    const rad = p.angle * Math.PI / 180;
    const px = cx + innerOrbitR * Math.cos(rad);
    const py = cy + innerOrbitR * Math.sin(rad);
    const delay = (1.0 + i * 0.25).toFixed(2);

    productsSvg += `
      <g class="ge-product" style="animation-delay: ${delay}s;">
        <!-- Orbital line to core -->
        <line x1="${cx}" y1="${cy}" x2="${px}" y2="${py}"
          stroke="#7EB8DA" stroke-width="1" stroke-opacity="0.25" stroke-dasharray="3,5"/>
        <!-- Product box -->
        <rect x="${px - 30}" y="${py - 14}" width="60" height="28" rx="8"
          fill="var(--bg-elevated)" stroke="#7EB8DA" stroke-width="1.5"/>
        <rect x="${px - 30}" y="${py - 14}" width="60" height="28" rx="8"
          fill="#7EB8DA10"/>
        <text x="${px}" y="${py + 1}" text-anchor="middle" dominant-baseline="central"
          fill="#7EB8DA" font-family="var(--font-heading)" font-size="11" font-weight="600">
          ${p.label}
        </text>
      </g>
    `;
  });

  // Build capability dots (outer orbit) - each has its own orbit animation
  let capabilitiesSvg = '';
  capabilities.forEach((c, i) => {
    const delay = (2.0 + i * 0.2).toFixed(2);
    const dur = c.speed;
    const startAngle = c.angle;

    capabilitiesSvg += `
      <g class="ge-capability" style="animation-delay: ${delay}s;">
        <g style="animation: geOrbit${i} ${dur}s linear infinite; animation-delay: 2.8s; transform-origin: ${cx}px ${cy}px;">
          <!-- Capability dot -->
          <circle cx="${cx + outerOrbitR * Math.cos(startAngle * Math.PI / 180)}"
                  cy="${cy + outerOrbitR * Math.sin(startAngle * Math.PI / 180)}"
                  r="6" fill="#9CCFA430" stroke="#9CCFA4" stroke-width="1.5"/>
          <circle cx="${cx + outerOrbitR * Math.cos(startAngle * Math.PI / 180)}"
                  cy="${cy + outerOrbitR * Math.sin(startAngle * Math.PI / 180)}"
                  r="3" fill="#9CCFA4"/>
          <!-- Label -->
          <text x="${cx + (outerOrbitR + 16) * Math.cos(startAngle * Math.PI / 180)}"
                y="${cy + (outerOrbitR + 16) * Math.sin(startAngle * Math.PI / 180)}"
                text-anchor="middle" dominant-baseline="central"
                fill="#9CCFA4" font-family="var(--font-mono)" font-size="9" opacity="0.85">
            ${c.label}
          </text>
        </g>
      </g>
    `;
  });

  // Generate unique orbit keyframes for each capability
  const orbitKeyframes = capabilities.map((c, i) => `
    @keyframes geOrbit${i} {
      from { transform: rotate(0deg); }
      to { transform: rotate(360deg); }
    }
  `).join('');

  container.innerHTML = `
    <div style="text-align: center; width: 100%;">
      <svg viewBox="0 0 750 380" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
        <style>
          .ge-title {
            opacity: 0;
            animation: geFadeIn 0.6s ease forwards;
          }
          .ge-core {
            opacity: 0;
            animation: geCoreIn 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
            animation-delay: 0.3s;
          }
          .ge-core-glow {
            animation: gePulseGlow 3s ease-in-out infinite;
            animation-delay: 0.5s;
          }
          .ge-orbit-ring {
            opacity: 0;
            animation: geFadeIn 0.6s ease forwards;
          }
          .ge-product {
            opacity: 0;
            animation: geFadeScale 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
          }
          .ge-capability {
            opacity: 0;
            animation: geFadeIn 0.5s ease forwards;
          }
          .ge-connector {
            opacity: 0;
            animation: geFadeIn 0.5s ease forwards;
          }
          .ge-badge {
            opacity: 0;
            animation: geFadeScale 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
          }
          .ge-sparkle {
            animation: geSparkle 2s ease-in-out infinite;
          }
          ${orbitKeyframes}
          @keyframes geFadeIn {
            to { opacity: 1; }
          }
          @keyframes geCoreIn {
            0% { opacity: 0; transform: scale(0.3); }
            100% { opacity: 1; transform: scale(1); }
          }
          @keyframes geFadeScale {
            0% { opacity: 0; transform: scale(0.5); }
            100% { opacity: 1; transform: scale(1); }
          }
          @keyframes gePulseGlow {
            0%, 100% { opacity: 0.2; }
            50% { opacity: 0.5; }
          }
          @keyframes geSparkle {
            0%, 100% { opacity: 0.2; transform: scale(0.8); }
            50% { opacity: 0.8; transform: scale(1.2); }
          }
        </style>

        <defs>
          <filter id="geGlow">
            <feGaussianBlur in="SourceGraphic" stdDeviation="10"/>
          </filter>
          <filter id="geGlowSmall">
            <feGaussianBlur in="SourceGraphic" stdDeviation="4"/>
          </filter>
          <radialGradient id="geCoreGrad" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stop-color="#C4A7E7" stop-opacity="0.3"/>
            <stop offset="100%" stop-color="#C4A7E7" stop-opacity="0.05"/>
          </radialGradient>
        </defs>

        <!-- Header -->
        <text x="${cx}" y="28" text-anchor="middle" class="ge-title" style="animation-delay: 0s;"
          fill="var(--text-dim)" font-family="var(--font-mono)" font-size="11" letter-spacing="2">
          Google DeepMind
        </text>

        <!-- Orbit rings (subtle guides) -->
        <circle cx="${cx}" cy="${cy}" r="${innerOrbitR}" fill="none" stroke="var(--border-subtle)"
          stroke-width="0.8" stroke-dasharray="6,8"
          class="ge-orbit-ring" style="animation-delay: 0.6s;"/>
        <circle cx="${cx}" cy="${cy}" r="${outerOrbitR}" fill="none" stroke="var(--border-subtle)"
          stroke-width="0.5" stroke-dasharray="4,10"
          class="ge-orbit-ring" style="animation-delay: 0.8s;"/>

        <!-- Core Gemini node -->
        <g class="ge-core">
          <!-- Outer glow -->
          <circle cx="${cx}" cy="${cy}" r="55" fill="#C4A7E7" opacity="0.2" filter="url(#geGlow)"
            class="ge-core-glow"/>
          <!-- Core circle -->
          <circle cx="${cx}" cy="${cy}" r="42" fill="url(#geCoreGrad)" stroke="#C4A7E7" stroke-width="2"/>
          <circle cx="${cx}" cy="${cy}" r="42" fill="var(--bg-elevated)" opacity="0.7"/>
          <!-- Inner accent ring -->
          <circle cx="${cx}" cy="${cy}" r="35" fill="none" stroke="#C4A7E7" stroke-width="0.8" stroke-opacity="0.4"/>
          <!-- Gemini text -->
          <text x="${cx}" y="${cy - 6}" text-anchor="middle" dominant-baseline="central"
            fill="#C4A7E7" font-family="var(--font-heading)" font-size="18" font-weight="700">
            Gemini
          </text>
          <text x="${cx}" y="${cy + 13}" text-anchor="middle" dominant-baseline="central"
            fill="var(--text-dim)" font-family="var(--font-mono)" font-size="8">
            Foundation Model
          </text>
        </g>

        <!-- Decorative sparkles around core -->
        <circle cx="${cx - 50}" cy="${cy - 45}" r="2" fill="#C4A7E7" class="ge-sparkle" style="animation-delay: 0s;"/>
        <circle cx="${cx + 48}" cy="${cy - 38}" r="1.5" fill="#C4A7E7" class="ge-sparkle" style="animation-delay: 0.7s;"/>
        <circle cx="${cx + 52}" cy="${cy + 40}" r="2" fill="#C4A7E7" class="ge-sparkle" style="animation-delay: 1.4s;"/>
        <circle cx="${cx - 45}" cy="${cy + 42}" r="1.5" fill="#C4A7E7" class="ge-sparkle" style="animation-delay: 2.1s;"/>

        <!-- Product nodes (inner orbit) -->
        ${productsSvg}

        <!-- Capability dots (outer orbit, rotating) -->
        ${capabilitiesSvg}

        <!-- Model variants badges -->
        <g class="ge-badge" style="animation-delay: 2.8s;">
          <rect x="${cx - 90}" y="${cy + outerOrbitR + 30}" width="52" height="20" rx="10"
            fill="#C4A7E720" stroke="#C4A7E760" stroke-width="1"/>
          <text x="${cx - 64}" y="${cy + outerOrbitR + 41}" text-anchor="middle" dominant-baseline="central"
            fill="#C4A7E7" font-family="var(--font-mono)" font-size="8" font-weight="600">Ultra</text>
        </g>
        <g class="ge-badge" style="animation-delay: 3.0s;">
          <rect x="${cx - 26}" y="${cy + outerOrbitR + 30}" width="52" height="20" rx="10"
            fill="#7EB8DA20" stroke="#7EB8DA60" stroke-width="1"/>
          <text x="${cx}" y="${cy + outerOrbitR + 41}" text-anchor="middle" dominant-baseline="central"
            fill="#7EB8DA" font-family="var(--font-mono)" font-size="8" font-weight="600">Pro</text>
        </g>
        <g class="ge-badge" style="animation-delay: 3.2s;">
          <rect x="${cx + 38}" y="${cy + outerOrbitR + 30}" width="52" height="20" rx="10"
            fill="#9CCFA420" stroke="#9CCFA460" stroke-width="1"/>
          <text x="${cx + 64}" y="${cy + outerOrbitR + 41}" text-anchor="middle" dominant-baseline="central"
            fill="#9CCFA4" font-family="var(--font-mono)" font-size="8" font-weight="600">Flash</text>
        </g>

        <!-- Labels for rings -->
        <g class="ge-connector" style="animation-delay: 1.5s;">
          <text x="${cx + innerOrbitR + 40}" y="${cy - innerOrbitR + 10}" text-anchor="start"
            fill="var(--text-dim)" font-family="var(--font-mono)" font-size="8" opacity="0.6">
            Products
          </text>
          <line x1="${cx + innerOrbitR + 38}" y1="${cy - innerOrbitR + 12}" x2="${cx + innerOrbitR + 5}" y2="${cy - innerOrbitR + 20}"
            stroke="var(--border-subtle)" stroke-width="0.5"/>
        </g>
        <g class="ge-connector" style="animation-delay: 2.2s;">
          <text x="${cx - outerOrbitR - 55}" y="${cy - outerOrbitR + 20}" text-anchor="start"
            fill="var(--text-dim)" font-family="var(--font-mono)" font-size="8" opacity="0.6">
            Capabilities
          </text>
          <line x1="${cx - outerOrbitR - 10}" y1="${cy - outerOrbitR + 22}" x2="${cx - outerOrbitR + 10}" y2="${cy - outerOrbitR + 30}"
            stroke="var(--border-subtle)" stroke-width="0.5"/>
        </g>

        <!-- Bottom text -->
        <text x="${cx}" y="372" text-anchor="middle" class="ge-title" style="animation-delay: 3.5s;"
          fill="var(--text-dim)" font-family="var(--font-mono)" font-size="10">
          Products orbit the core \u2014 Capabilities expand the frontier
        </text>
      </svg>
      <p style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-dim); margin-top: var(--space-3);">
        Gemini Ecosystem \u2014 Foundation model product architecture
      </p>
    </div>
  `;
}
