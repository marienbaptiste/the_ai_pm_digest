export function render(container) {
  const milestones = [
    { year: 1950, label: 'Turing Test', color: '#3b82f6', above: true },
    { year: 1957, label: 'Perceptron', color: '#3b82f6', above: false },
    { year: 1986, label: 'Backprop', color: '#a855f7', above: true },
    { year: 1997, label: 'Deep Blue', color: '#a855f7', above: false },
    { year: 2012, label: 'AlexNet', color: '#f59e0b', above: true },
    { year: 2017, label: 'Transformer', color: '#00d4aa', above: false },
    { year: 2022, label: 'ChatGPT', color: '#ef4444', above: true }
  ];

  const lineY = 190;
  const startX = 65;
  const endX = 695;
  const totalSpan = milestones[milestones.length - 1].year - milestones[0].year;

  function xFor(year) {
    return startX + ((year - milestones[0].year) / totalSpan) * (endX - startX);
  }

  let milestoneSvg = '';
  milestones.forEach((m, i) => {
    const x = xFor(m.year);
    const stemTop = m.above ? lineY - 55 : lineY + 55;
    const labelY = m.above ? lineY - 65 : lineY + 75;
    const yearY = m.above ? lineY - 80 : lineY + 90;
    const delay = (0.5 + i * 0.25).toFixed(2);

    milestoneSvg += `
      <g class="tl-milestone" style="animation-delay: ${delay}s;">
        <!-- Stem -->
        <line x1="${x}" y1="${lineY}" x2="${x}" y2="${stemTop}"
          stroke="${m.color}" stroke-width="2" opacity="0.6"/>
        <!-- Node circle -->
        <circle cx="${x}" cy="${lineY}" r="7" fill="${m.color}30" stroke="${m.color}" stroke-width="2"/>
        <circle cx="${x}" cy="${lineY}" r="3" fill="${m.color}"/>
        <!-- Label -->
        <text x="${x}" y="${labelY}" text-anchor="middle"
          fill="${m.color}" font-family="var(--font-heading)" font-size="12" font-weight="600">
          ${m.label}
        </text>
        <!-- Year -->
        <text x="${x}" y="${yearY}" text-anchor="middle"
          fill="var(--text-dim)" font-family="var(--font-mono)" font-size="10">
          ${m.year}
        </text>
      </g>
    `;
  });

  // Era brackets
  const eras = [
    { start: 1950, end: 1957, label: 'Foundations', color: '#3b82f6' },
    { start: 1986, end: 1997, label: 'Revival', color: '#a855f7' },
    { start: 2012, end: 2022, label: 'Deep Learning', color: '#00d4aa' }
  ];

  let eraSvg = '';
  eras.forEach((era, i) => {
    const x1 = xFor(era.start);
    const x2 = xFor(era.end);
    const delay = (2.5 + i * 0.3).toFixed(2);
    eraSvg += `
      <g class="tl-era" style="animation-delay: ${delay}s;">
        <rect x="${x1 - 5}" y="${lineY - 3}" width="${x2 - x1 + 10}" height="6" rx="3"
          fill="${era.color}" opacity="0.15"/>
      </g>
    `;
  });

  container.innerHTML = `
    <div style="text-align: center; width: 100%;">
      <svg viewBox="0 0 750 350" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
        <style>
          .tl-milestone {
            animation: tlFadeIn 0.7s ease forwards;
            opacity: 0;
          }
          .tl-era {
            animation: tlFadeIn 0.8s ease forwards;
            opacity: 0;
          }
          .tl-line {
            stroke-dasharray: 650;
            stroke-dashoffset: 650;
            animation: tlDrawLine 1.5s ease forwards;
          }
          .tl-glow {
            animation: tlGlowTravel 4s linear infinite 2s;
          }
          .tl-title {
            animation: tlFadeIn 0.5s ease forwards;
            opacity: 0;
          }
          @keyframes tlFadeIn {
            to { opacity: 1; }
          }
          @keyframes tlDrawLine {
            to { stroke-dashoffset: 0; }
          }
          @keyframes tlGlowTravel {
            0% { offset-distance: 0%; opacity: 0; }
            5% { opacity: 1; }
            95% { opacity: 1; }
            100% { offset-distance: 100%; opacity: 0; }
          }
          @keyframes tlPulse {
            0%, 100% { opacity: 0.3; r: 5; }
            50% { opacity: 0.7; r: 10; }
          }
        </style>

        <!-- Title -->
        <text x="375" y="30" text-anchor="middle" class="tl-title"
          fill="var(--text-primary)" font-family="var(--font-heading)" font-size="16" font-weight="700">
          Key Milestones in Artificial Intelligence
        </text>

        <!-- Main timeline line -->
        <line x1="${startX - 15}" y1="${lineY}" x2="${endX + 15}" y2="${lineY}"
          stroke="var(--border-medium)" stroke-width="2.5" stroke-linecap="round"
          class="tl-line"/>

        <!-- Animated glow that travels along the line -->
        <path id="tl-path" d="M${startX - 15},${lineY} L${endX + 15},${lineY}" fill="none" stroke="none"/>
        <circle r="6" fill="#00d4aa" opacity="0" class="tl-glow">
          <animateMotion dur="4s" repeatCount="indefinite" begin="2s"
            path="M${startX - 15},0 L${endX + 15},0"/>
          <animate attributeName="opacity" values="0;0.8;0.8;0" dur="4s" repeatCount="indefinite" begin="2s"/>
        </circle>
        <!-- Glow halo -->
        <circle r="14" fill="#00d4aa" opacity="0" filter="url(#tlBlur)">
          <animateMotion dur="4s" repeatCount="indefinite" begin="2s"
            path="M${startX - 15},${lineY} L${endX + 15},${lineY}"/>
          <animate attributeName="opacity" values="0;0.3;0.3;0" dur="4s" repeatCount="indefinite" begin="2s"/>
        </circle>

        <defs>
          <filter id="tlBlur">
            <feGaussianBlur in="SourceGraphic" stdDeviation="4"/>
          </filter>
        </defs>

        <!-- Era background bars -->
        ${eraSvg}

        <!-- Start/end arrows -->
        <polygon points="${startX - 25},${lineY} ${startX - 15},${lineY - 6} ${startX - 15},${lineY + 6}"
          fill="var(--border-medium)" class="tl-milestone" style="animation-delay: 0.2s;"/>
        <polygon points="${endX + 25},${lineY} ${endX + 15},${lineY - 6} ${endX + 15},${lineY + 6}"
          fill="var(--border-medium)" class="tl-milestone" style="animation-delay: 0.2s;"/>

        <!-- Milestones -->
        ${milestoneSvg}

        <!-- Legend at bottom -->
        <g class="tl-era" style="animation-delay: 3s;">
          <circle cx="200" cy="325" r="5" fill="#3b82f6"/>
          <text x="212" y="329" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="10">Foundations</text>
          <circle cx="320" cy="325" r="5" fill="#a855f7"/>
          <text x="332" y="329" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="10">Revival</text>
          <circle cx="420" cy="325" r="5" fill="#00d4aa"/>
          <text x="432" y="329" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="10">Deep Learning</text>
          <circle cx="545" cy="325" r="5" fill="#ef4444"/>
          <text x="557" y="329" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="10">Generative AI</text>
        </g>
      </svg>
      <p style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-dim); margin-top: var(--space-3);">
        AI Timeline \u2014 From Turing to ChatGPT
      </p>
    </div>
  `;
}
