export function render(container) {
  const cx = 375;
  const cy = 180;
  const innerR = 95;
  const outerR = 155;

  const innerStakeholders = [
    { label: 'Engineering', icon: '\u2699', angle: -90 },
    { label: 'Design/UX', icon: '\u270E', angle: 30 },
    { label: 'Research', icon: '\uD83D\uDD2C', angle: 150 }
  ];

  const outerStakeholders = [
    { label: 'Legal', icon: '\u2696', angle: -110 },
    { label: 'Marketing', icon: '\uD83D\uDCE2', angle: -50 },
    { label: 'Executives', icon: '\u2605', angle: 10 },
    { label: 'Data Science', icon: '\uD83D\uDCC8', angle: 75 },
    { label: 'Support', icon: '\uD83C\uDFA7', angle: 140 },
    { label: 'Sales', icon: '\uD83E\uDD1D', angle: 200 }
  ];

  // Build connection lines (inner = solid thick, outer = dashed thin)
  let connectionsSvg = '';
  let nodesSvg = '';

  // Inner stakeholder connections and nodes
  innerStakeholders.forEach((s, i) => {
    const rad = s.angle * Math.PI / 180;
    const sx = cx + innerR * Math.cos(rad);
    const sy = cy + innerR * Math.sin(rad);
    const delay = (0.8 + i * 0.2).toFixed(2);
    const lineDelay = (0.6 + i * 0.15).toFixed(2);

    // Connection line (thick, animated dash flow)
    connectionsSvg += `
      <line x1="${cx}" y1="${cy}" x2="${sx}" y2="${sy}"
        stroke="#3b82f6" stroke-width="2.5" stroke-opacity="0.5"
        class="sm-line-inner" style="animation-delay: ${lineDelay}s;"/>
      <line x1="${cx}" y1="${cy}" x2="${sx}" y2="${sy}"
        stroke="#3b82f6" stroke-width="2.5" stroke-dasharray="4,6"
        class="sm-flow-line" style="animation-delay: 1.8s;"/>
    `;

    // Node
    nodesSvg += `
      <g class="sm-inner-node" style="animation-delay: ${delay}s;">
        <circle cx="${sx}" cy="${sy}" r="28" fill="var(--bg-elevated)" stroke="#3b82f6" stroke-width="2"/>
        <circle cx="${sx}" cy="${sy}" r="28" fill="#3b82f620"/>
        <text x="${sx}" y="${sy - 5}" text-anchor="middle" dominant-baseline="central"
          fill="#3b82f6" font-size="14">${s.icon}</text>
        <text x="${sx}" y="${sy + 12}" text-anchor="middle" dominant-baseline="central"
          fill="var(--text-secondary)" font-family="var(--font-heading)" font-size="9" font-weight="600">
          ${s.label}
        </text>
      </g>
    `;
  });

  // Outer stakeholder connections and nodes
  outerStakeholders.forEach((s, i) => {
    const rad = s.angle * Math.PI / 180;
    const sx = cx + outerR * Math.cos(rad);
    const sy = cy + outerR * Math.sin(rad);
    const delay = (1.4 + i * 0.15).toFixed(2);
    const lineDelay = (1.2 + i * 0.1).toFixed(2);

    // Connection line (thin dashed)
    connectionsSvg += `
      <line x1="${cx}" y1="${cy}" x2="${sx}" y2="${sy}"
        stroke="#f59e0b" stroke-width="1.2" stroke-dasharray="5,5" stroke-opacity="0.35"
        class="sm-line-outer" style="animation-delay: ${lineDelay}s;"/>
      <line x1="${cx}" y1="${cy}" x2="${sx}" y2="${sy}"
        stroke="#f59e0b" stroke-width="1.2" stroke-dasharray="3,7"
        class="sm-flow-line-outer" style="animation-delay: 2.2s;"/>
    `;

    // Node (smaller)
    nodesSvg += `
      <g class="sm-outer-node" style="animation-delay: ${delay}s;">
        <circle cx="${sx}" cy="${sy}" r="22" fill="var(--bg-elevated)" stroke="#f59e0b" stroke-width="1.5" stroke-opacity="0.7"/>
        <circle cx="${sx}" cy="${sy}" r="22" fill="#f59e0b10"/>
        <text x="${sx}" y="${sy - 4}" text-anchor="middle" dominant-baseline="central"
          fill="#f59e0b" font-size="12">${s.icon}</text>
        <text x="${sx}" y="${sy + 11}" text-anchor="middle" dominant-baseline="central"
          fill="var(--text-dim)" font-family="var(--font-heading)" font-size="8" font-weight="600">
          ${s.label}
        </text>
      </g>
    `;
  });

  container.innerHTML = `
    <div style="text-align: center; width: 100%;">
      <svg viewBox="0 0 750 350" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
        <style>
          .sm-center {
            opacity: 0;
            animation: smFadeScale 0.7s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
            animation-delay: 0.2s;
          }
          .sm-center-glow {
            animation: smPulse 3s ease-in-out infinite;
            animation-delay: 0.5s;
          }
          .sm-line-inner {
            opacity: 0;
            animation: smFadeIn 0.5s ease forwards;
          }
          .sm-line-outer {
            opacity: 0;
            animation: smFadeIn 0.5s ease forwards;
          }
          .sm-flow-line {
            opacity: 0;
            stroke-opacity: 0.4;
            animation: smFlowIn 0.5s ease forwards, smDashFlow 2s linear infinite;
            animation-delay: 1.8s, 1.8s;
          }
          .sm-flow-line-outer {
            opacity: 0;
            stroke-opacity: 0.25;
            animation: smFlowIn 0.5s ease forwards, smDashFlow 3s linear infinite;
            animation-delay: 2.2s, 2.2s;
          }
          .sm-inner-node {
            opacity: 0;
            animation: smFadeScale 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
          }
          .sm-inner-node:hover circle {
            transform-origin: center;
            filter: brightness(1.2);
          }
          .sm-outer-node {
            opacity: 0;
            animation: smFadeScale 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
          }
          .sm-outer-node:hover circle {
            transform-origin: center;
            filter: brightness(1.2);
          }
          .sm-title {
            opacity: 0;
            animation: smFadeIn 0.6s ease forwards;
          }
          .sm-ring-guide {
            opacity: 0;
            animation: smFadeIn 0.5s ease forwards;
          }
          .sm-legend {
            opacity: 0;
            animation: smFadeIn 0.5s ease forwards;
            animation-delay: 2.5s;
          }
          @keyframes smFadeIn {
            to { opacity: 1; }
          }
          @keyframes smFadeScale {
            0% { opacity: 0; transform: scale(0.4); }
            100% { opacity: 1; transform: scale(1); }
          }
          @keyframes smPulse {
            0%, 100% { opacity: 0.15; }
            50% { opacity: 0.4; }
          }
          @keyframes smDashFlow {
            0% { stroke-dashoffset: 0; }
            100% { stroke-dashoffset: -20; }
          }
          @keyframes smFlowIn {
            to { opacity: 1; }
          }
        </style>

        <defs>
          <filter id="smGlow">
            <feGaussianBlur in="SourceGraphic" stdDeviation="8"/>
          </filter>
        </defs>

        <!-- Title -->
        <text x="${cx}" y="25" text-anchor="middle" class="sm-title" style="animation-delay: 0s;"
          fill="var(--text-primary)" font-family="var(--font-heading)" font-size="16" font-weight="700">
          Cross-Functional Stakeholder Map
        </text>

        <!-- Guide rings (subtle) -->
        <circle cx="${cx}" cy="${cy}" r="${innerR}" fill="none" stroke="var(--border-subtle)" stroke-width="0.5"
          stroke-dasharray="4,8" class="sm-ring-guide" style="animation-delay: 0.5s;"/>
        <circle cx="${cx}" cy="${cy}" r="${outerR}" fill="none" stroke="var(--border-subtle)" stroke-width="0.5"
          stroke-dasharray="4,8" class="sm-ring-guide" style="animation-delay: 1.0s;"/>

        <!-- Connection lines (drawn behind nodes) -->
        ${connectionsSvg}

        <!-- Center PM node -->
        <g class="sm-center">
          <circle cx="${cx}" cy="${cy}" r="38" fill="#00d4aa" opacity="0.15" filter="url(#smGlow)"
            class="sm-center-glow"/>
          <circle cx="${cx}" cy="${cy}" r="32" fill="var(--bg-elevated)" stroke="#00d4aa" stroke-width="2.5"/>
          <circle cx="${cx}" cy="${cy}" r="32" fill="#00d4aa15"/>
          <text x="${cx}" y="${cy - 6}" text-anchor="middle" dominant-baseline="central"
            fill="#00d4aa" font-family="var(--font-heading)" font-size="16" font-weight="700">
            PM
          </text>
          <text x="${cx}" y="${cy + 11}" text-anchor="middle" dominant-baseline="central"
            fill="var(--text-dim)" font-family="var(--font-mono)" font-size="8">
            Product Lead
          </text>
        </g>

        <!-- Stakeholder nodes -->
        ${nodesSvg}

        <!-- Legend -->
        <g class="sm-legend">
          <line x1="55" y1="330" x2="75" y2="330" stroke="#3b82f6" stroke-width="2.5"/>
          <text x="82" y="334" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="9">Core Team (daily)</text>

          <line x1="230" y1="330" x2="250" y2="330" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="4,4"/>
          <text x="257" y="334" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="9">Extended (weekly)</text>

          <circle cx="420" cy="330" r="5" fill="#00d4aa30" stroke="#00d4aa" stroke-width="1.5"/>
          <text x="432" y="334" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="9">PM (hub)</text>

          <text x="530" y="334" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="9">
            \u2192 Animated dashes = communication flow
          </text>
        </g>
      </svg>
      <p style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-dim); margin-top: var(--space-3);">
        Stakeholder Map \u2014 PM as the cross-functional hub
      </p>
    </div>
  `;
}
