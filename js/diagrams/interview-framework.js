export function render(container) {
  const steps = [
    { letter: 'C', word: 'Comprehend', desc: 'Understand the question and context', color: '#7EB8DA' },
    { letter: 'I', word: 'Identify', desc: 'Identify the user and customer', color: '#7EB8DA' },
    { letter: 'R', word: 'Report', desc: 'Report customer needs and pain points', color: '#C4A7E7' },
    { letter: 'C', word: 'Cut', desc: 'Cut through prioritization', color: '#C4A7E7' },
    { letter: 'L', word: 'List', desc: 'List possible solutions', color: '#9CCFA4' },
    { letter: 'E', word: 'Evaluate', desc: 'Evaluate trade-offs systematically', color: '#9CCFA4' },
    { letter: 'S', word: 'Summarize', desc: 'Summarize your recommendation', color: '#F0B429' }
  ];

  const startY = 55;
  const barHeight = 36;
  const barGap = 6;
  const barX = 120;
  const maxBarWidth = 530;
  const lineX = 95;

  let stepsSvg = '';
  let arrowsSvg = '';
  let connectorSvg = '';

  // Vertical connector line on the left
  const lineTopY = startY + barHeight / 2;
  const lineBottomY = startY + (steps.length - 1) * (barHeight + barGap) + barHeight / 2;

  connectorSvg = `
    <line x1="${lineX}" y1="${lineTopY}" x2="${lineX}" y2="${lineBottomY}"
      stroke="var(--border-medium)" stroke-width="2" stroke-linecap="round"
      class="if-connector"/>
  `;

  steps.forEach((step, i) => {
    const y = startY + i * (barHeight + barGap);
    const delay = (0.3 + i * 0.2).toFixed(2);
    // Progressive bar width (slightly narrower as you go down for visual taper, then widens at end)
    const widthFactor = [1.0, 0.95, 0.9, 0.85, 0.88, 0.92, 1.0][i];
    const barW = maxBarWidth * widthFactor;

    // Step number dot on connector line
    connectorSvg += `
      <g class="if-dot" style="animation-delay: ${delay}s;">
        <circle cx="${lineX}" cy="${y + barHeight / 2}" r="8" fill="var(--bg-elevated)"
          stroke="${step.color}" stroke-width="2"/>
        <text x="${lineX}" y="${y + barHeight / 2 + 1}" text-anchor="middle" dominant-baseline="central"
          fill="${step.color}" font-family="var(--font-mono)" font-size="8" font-weight="700">
          ${i + 1}
        </text>
      </g>
    `;

    // Bar
    stepsSvg += `
      <g class="if-bar" style="animation-delay: ${delay}s;">
        <!-- Background bar -->
        <rect x="${barX}" y="${y}" width="${barW}" height="${barHeight}" rx="6"
          fill="${step.color}10" stroke="${step.color}40" stroke-width="1"/>

        <!-- Letter circle -->
        <circle cx="${barX + 22}" cy="${y + barHeight / 2}" r="14"
          fill="${step.color}20" stroke="${step.color}" stroke-width="1.5"/>
        <text x="${barX + 22}" y="${y + barHeight / 2 + 1}" text-anchor="middle" dominant-baseline="central"
          fill="${step.color}" font-family="var(--font-heading)" font-size="15" font-weight="700">
          ${step.letter}
        </text>

        <!-- Word -->
        <text x="${barX + 50}" y="${y + barHeight / 2 - 3}" dominant-baseline="central"
          fill="${step.color}" font-family="var(--font-heading)" font-size="13" font-weight="600">
          ${step.word}
        </text>

        <!-- Description -->
        <text x="${barX + 50}" y="${y + barHeight / 2 + 12}" dominant-baseline="central"
          fill="var(--text-dim)" font-family="var(--font-mono)" font-size="9">
          ${step.desc}
        </text>

        <!-- Right-side indicator bar (visual weight) -->
        <rect x="${barX + barW - 4}" y="${y + 6}" width="3" height="${barHeight - 12}" rx="1.5"
          fill="${step.color}" opacity="0.4"/>
      </g>
    `;

    // Arrow between steps
    if (i < steps.length - 1) {
      const arrowY = y + barHeight + barGap / 2;
      arrowsSvg += `
        <g class="if-arrow" style="animation-delay: ${(0.5 + i * 0.2).toFixed(2)}s;">
          <path d="M${lineX - 3},${arrowY - 1} L${lineX},${arrowY + 2} L${lineX + 3},${arrowY - 1}"
            fill="none" stroke="var(--text-dim)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.4"/>
        </g>
      `;
    }
  });

  container.innerHTML = `
    <div style="text-align: center; width: 100%;">
      <svg viewBox="0 0 750 370" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
        <style>
          .if-title {
            opacity: 0;
            animation: ifFadeIn 0.6s ease forwards;
          }
          .if-connector {
            stroke-dasharray: ${lineBottomY - lineTopY};
            stroke-dashoffset: ${lineBottomY - lineTopY};
            animation: ifDrawLine 1.5s ease forwards;
            animation-delay: 0.2s;
          }
          .if-dot {
            opacity: 0;
            animation: ifPopIn 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
          }
          .if-bar {
            opacity: 0;
            animation: ifSlideIn 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94) forwards;
          }
          .if-arrow {
            opacity: 0;
            animation: ifFadeIn 0.3s ease forwards;
          }
          .if-subtitle {
            opacity: 0;
            animation: ifFadeIn 0.6s ease forwards;
          }
          .if-badge {
            opacity: 0;
            animation: ifPopIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
          }
          .if-accent-line {
            stroke-dasharray: 80;
            stroke-dashoffset: 80;
            animation: ifDrawShort 0.8s ease forwards;
            animation-delay: 0.1s;
          }
          @keyframes ifFadeIn {
            to { opacity: 1; }
          }
          @keyframes ifSlideIn {
            0% { opacity: 0; transform: translateX(-40px); }
            100% { opacity: 1; transform: translateX(0); }
          }
          @keyframes ifPopIn {
            0% { opacity: 0; transform: scale(0.4); }
            100% { opacity: 1; transform: scale(1); }
          }
          @keyframes ifDrawLine {
            to { stroke-dashoffset: 0; }
          }
          @keyframes ifDrawShort {
            to { stroke-dashoffset: 0; }
          }
        </style>

        <!-- Title area -->
        <g class="if-title" style="animation-delay: 0s;">
          <text x="375" y="22" text-anchor="middle"
            fill="var(--text-primary)" font-family="var(--font-heading)" font-size="16" font-weight="700">
            CIRCLES Framework
          </text>
        </g>

        <!-- Accent line under title -->
        <line x1="300" y1="32" x2="450" y2="32" stroke="#9CCFA4" stroke-width="2"
          stroke-linecap="round" class="if-accent-line"/>

        <!-- "For PM Interviews" badge -->
        <g class="if-badge" style="animation-delay: 0.15s;">
          <rect x="325" y="37" width="100" height="16" rx="8"
            fill="#9CCFA415" stroke="#9CCFA440" stroke-width="1"/>
          <text x="375" y="47" text-anchor="middle" dominant-baseline="central"
            fill="#9CCFA4" font-family="var(--font-mono)" font-size="8">PM Interviews</text>
        </g>

        <!-- Vertical connector -->
        ${connectorSvg}

        <!-- Step bars -->
        ${stepsSvg}

        <!-- Inter-step arrows -->
        ${arrowsSvg}

        <!-- Color legend at bottom left -->
        <g class="if-subtitle" style="animation-delay: 2.0s;">
          <circle cx="115" cy="350" r="4" fill="#7EB8DA"/>
          <text x="125" y="354" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="8">Define</text>
          <circle cx="175" cy="350" r="4" fill="#C4A7E7"/>
          <text x="185" y="354" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="8">Analyze</text>
          <circle cx="240" cy="350" r="4" fill="#9CCFA4"/>
          <text x="250" y="354" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="8">Solve</text>
          <circle cx="300" cy="350" r="4" fill="#F0B429"/>
          <text x="310" y="354" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="8">Conclude</text>
        </g>

        <!-- Bottom subtitle -->
        <g class="if-subtitle" style="animation-delay: 2.2s;">
          <text x="550" y="354" text-anchor="middle"
            fill="var(--text-dim)" font-family="var(--font-mono)" font-size="9">
            Structured approach to product questions
          </text>
        </g>
      </svg>
      <p style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-dim); margin-top: var(--space-3);">
        CIRCLES Method \u2014 7-step framework for PM interview answers
      </p>
    </div>
  `;
}
