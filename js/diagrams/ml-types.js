export function render(container) {
  // Supervised Learning data points
  const supervisedPoints = [
    { x: 30, y: 25, cls: 0 }, { x: 45, y: 35, cls: 0 }, { x: 35, y: 50, cls: 0 },
    { x: 55, y: 30, cls: 0 }, { x: 40, y: 45, cls: 0 },
    { x: 120, y: 100, cls: 1 }, { x: 135, y: 115, cls: 1 }, { x: 110, y: 120, cls: 1 },
    { x: 145, y: 105, cls: 1 }, { x: 125, y: 130, cls: 1 }
  ];

  let supPointsSvg = '';
  supervisedPoints.forEach((p, i) => {
    const color = p.cls === 0 ? '#60a5fa' : '#f59e0b';
    const shape = p.cls === 0 ? 'circle' : 'rect';
    const delay = (0.8 + i * 0.08).toFixed(2);
    if (shape === 'circle') {
      supPointsSvg += `<circle cx="${p.x}" cy="${p.y}" r="5" fill="${color}" class="ml-point" style="animation-delay: ${delay}s;"/>`;
    } else {
      supPointsSvg += `<rect x="${p.x - 4}" y="${p.y - 4}" width="8" height="8" rx="1" fill="${color}" class="ml-point" style="animation-delay: ${delay}s;"/>`;
    }
  });

  // Unsupervised clustering points
  const clusters = [
    { cx: 50, cy: 45, r: 30, color: '#c084fc', points: [
      { x: 40, y: 35 }, { x: 55, y: 40 }, { x: 45, y: 55 }, { x: 60, y: 50 }, { x: 38, y: 48 }
    ]},
    { cx: 130, cy: 110, r: 28, color: '#f0abfc', points: [
      { x: 120, y: 100 }, { x: 135, y: 115 }, { x: 140, y: 105 }, { x: 125, y: 120 }
    ]},
    { cx: 50, cy: 120, r: 25, color: '#e879f9', points: [
      { x: 40, y: 115 }, { x: 55, y: 125 }, { x: 48, y: 110 }, { x: 60, y: 118 }
    ]}
  ];

  let unsupSvg = '';
  clusters.forEach((cl, ci) => {
    const delay = (1.5 + ci * 0.3).toFixed(2);
    cl.points.forEach((p, pi) => {
      const pDelay = (0.8 + ci * 0.2 + pi * 0.08).toFixed(2);
      unsupSvg += `<circle cx="${p.x}" cy="${p.y}" r="5" fill="${cl.color}" class="ml-point" style="animation-delay: ${pDelay}s;"/>`;
    });
    unsupSvg += `<circle cx="${cl.cx}" cy="${cl.cy}" r="${cl.r}" fill="none"
      stroke="${cl.color}" stroke-width="1.5" stroke-dasharray="4 3" opacity="0"
      class="ml-cluster" style="animation-delay: ${delay}s;"/>`;
  });

  container.innerHTML = `
    <div style="text-align: center; width: 100%;">
      <svg viewBox="0 0 750 350" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
        <style>
          .ml-panel {
            animation: mlPanelIn 0.7s ease forwards;
            opacity: 0;
          }
          .ml-point {
            animation: mlPointPop 0.4s ease forwards;
            opacity: 0;
            transform-origin: center;
          }
          .ml-cluster {
            animation: mlClusterIn 0.8s ease forwards;
            opacity: 0;
          }
          .ml-label {
            animation: mlFadeIn 0.5s ease forwards;
            opacity: 0;
          }
          .ml-boundary {
            stroke-dasharray: 200;
            stroke-dashoffset: 200;
            animation: mlDrawLine 1.5s ease forwards 1.5s;
          }
          .ml-arrow {
            stroke-dasharray: 6;
            animation: mlFlowArrow 1.2s linear infinite;
          }
          .ml-rl-pulse {
            animation: mlRlPulse 2s ease-in-out infinite;
          }
          @keyframes mlPanelIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
          }
          @keyframes mlFadeIn {
            to { opacity: 1; }
          }
          @keyframes mlPointPop {
            0% { opacity: 0; transform: scale(0); }
            70% { transform: scale(1.3); }
            100% { opacity: 1; transform: scale(1); }
          }
          @keyframes mlClusterIn {
            to { opacity: 0.7; }
          }
          @keyframes mlDrawLine {
            to { stroke-dashoffset: 0; }
          }
          @keyframes mlFlowArrow {
            to { stroke-dashoffset: -12; }
          }
          @keyframes mlRlPulse {
            0%, 100% { opacity: 0.7; }
            50% { opacity: 1; }
          }
          @keyframes mlRewardGlow {
            0%, 100% { fill: #00d4aa; filter: none; }
            50% { fill: #00ffcc; filter: url(#mlGlow); }
          }
        </style>

        <defs>
          <filter id="mlGlow">
            <feGaussianBlur in="SourceGraphic" stdDeviation="3"/>
          </filter>
        </defs>

        <!-- ===== Panel 1: Supervised ===== -->
        <g class="ml-panel" style="animation-delay: 0s;">
          <rect x="15" y="40" width="230" height="260" rx="12"
            fill="#3b82f608" stroke="#3b82f6" stroke-width="1.5" opacity="0.8"/>
          <text x="130" y="30" text-anchor="middle"
            fill="#3b82f6" font-family="var(--font-heading)" font-size="14" font-weight="700">
            Supervised Learning
          </text>
        </g>

        <!-- Axis -->
        <g class="ml-label" style="animation-delay: 0.3s;">
          <line x1="35" y1="55" x2="35" y2="275" stroke="var(--border-subtle)" stroke-width="1"/>
          <line x1="35" y1="275" x2="225" y2="275" stroke="var(--border-subtle)" stroke-width="1"/>
          <text x="130" y="295" text-anchor="middle" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="9">Features (X)</text>
        </g>

        <!-- Data points with labels -->
        <g transform="translate(35, 55) scale(1.15, 1.15)">
          ${supPointsSvg}
          <!-- Decision boundary -->
          <line x1="10" y1="150" x2="155" y2="10"
            stroke="#3b82f6" stroke-width="2" class="ml-boundary"/>
        </g>

        <!-- X->Y labels -->
        <g class="ml-label" style="animation-delay: 1.8s;">
          <text x="75" y="85" fill="#60a5fa" font-family="var(--font-mono)" font-size="10" font-weight="600">Class A</text>
          <text x="155" y="210" fill="#f59e0b" font-family="var(--font-mono)" font-size="10" font-weight="600">Class B</text>
          <text x="60" y="258" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="9">X \u2192 Y (labeled)</text>
        </g>

        <!-- ===== Panel 2: Unsupervised ===== -->
        <g class="ml-panel" style="animation-delay: 0.3s;">
          <rect x="260" y="40" width="230" height="260" rx="12"
            fill="#a855f708" stroke="#a855f7" stroke-width="1.5" opacity="0.8"/>
          <text x="375" y="30" text-anchor="middle"
            fill="#a855f7" font-family="var(--font-heading)" font-size="14" font-weight="700">
            Unsupervised Learning
          </text>
        </g>

        <!-- Axis -->
        <g class="ml-label" style="animation-delay: 0.6s;">
          <line x1="280" y1="55" x2="280" y2="275" stroke="var(--border-subtle)" stroke-width="1"/>
          <line x1="280" y1="275" x2="470" y2="275" stroke="var(--border-subtle)" stroke-width="1"/>
          <text x="375" y="295" text-anchor="middle" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="9">Unlabeled Data</text>
        </g>

        <!-- Clustered points -->
        <g transform="translate(280, 55) scale(1.15, 1.15)">
          ${unsupSvg}
        </g>

        <g class="ml-label" style="animation-delay: 2.2s;">
          <text x="305" y="258" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="9">Discover structure</text>
        </g>

        <!-- ===== Panel 3: Reinforcement ===== -->
        <g class="ml-panel" style="animation-delay: 0.6s;">
          <rect x="505" y="40" width="230" height="260" rx="12"
            fill="#00d4aa08" stroke="#00d4aa" stroke-width="1.5" opacity="0.8"/>
          <text x="620" y="30" text-anchor="middle"
            fill="#00d4aa" font-family="var(--font-heading)" font-size="14" font-weight="700">
            Reinforcement Learning
          </text>
        </g>

        <!-- Agent -->
        <g class="ml-label" style="animation-delay: 1s;">
          <rect x="580" y="85" width="80" height="40" rx="8"
            fill="#00d4aa20" stroke="#00d4aa" stroke-width="2"/>
          <text x="620" y="110" text-anchor="middle"
            fill="#00d4aa" font-family="var(--font-heading)" font-size="12" font-weight="600">Agent</text>
        </g>

        <!-- Environment -->
        <g class="ml-label" style="animation-delay: 1.2s;">
          <rect x="565" y="195" width="110" height="45" rx="8"
            fill="#f59e0b15" stroke="#f59e0b" stroke-width="1.5"/>
          <text x="620" y="222" text-anchor="middle"
            fill="#f59e0b" font-family="var(--font-heading)" font-size="12" font-weight="600">Environment</text>
        </g>

        <!-- Action arrow (Agent -> Env, right side) -->
        <g class="ml-label" style="animation-delay: 1.5s;">
          <path d="M660,115 C690,115 690,200 660,200" fill="none"
            stroke="#00d4aa" stroke-width="2" class="ml-arrow"
            marker-end="url(#mlArrowHead)"/>
          <text x="700" y="160" text-anchor="middle"
            fill="#00d4aa" font-family="var(--font-mono)" font-size="9" font-weight="600">Action</text>
        </g>

        <!-- Reward arrow (Env -> Agent, left side) -->
        <g class="ml-label" style="animation-delay: 1.8s;">
          <path d="M580,200 C545,200 545,115 580,115" fill="none"
            stroke="#f59e0b" stroke-width="2" class="ml-arrow"
            marker-end="url(#mlArrowHeadAmber)"/>
          <text x="530" y="148" text-anchor="middle"
            fill="#f59e0b" font-family="var(--font-mono)" font-size="9" font-weight="600">Reward</text>
          <text x="530" y="162" text-anchor="middle"
            fill="#f59e0b" font-family="var(--font-mono)" font-size="9">+ State</text>
        </g>

        <!-- RL pulse on agent -->
        <circle cx="620" cy="105" r="12" fill="none" stroke="#00d4aa" stroke-width="1" opacity="0"
          class="ml-rl-pulse">
          <animate attributeName="r" values="12;22;12" dur="2s" repeatCount="indefinite" begin="2s"/>
          <animate attributeName="opacity" values="0.5;0;0.5" dur="2s" repeatCount="indefinite" begin="2s"/>
        </circle>

        <g class="ml-label" style="animation-delay: 2.2s;">
          <text x="556" y="258" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="9">Trial and error loop</text>
        </g>

        <!-- Arrowhead markers -->
        <defs>
          <marker id="mlArrowHead" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#00d4aa"/>
          </marker>
          <marker id="mlArrowHeadAmber" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#f59e0b"/>
          </marker>
        </defs>

        <!-- Bottom summary -->
        <g class="ml-label" style="animation-delay: 2.5s;">
          <text x="130" y="325" text-anchor="middle" fill="#3b82f6" font-family="var(--font-mono)" font-size="10">
            "Learn from labels"
          </text>
          <text x="375" y="325" text-anchor="middle" fill="#a855f7" font-family="var(--font-mono)" font-size="10">
            "Find patterns"
          </text>
          <text x="620" y="325" text-anchor="middle" fill="#00d4aa" font-family="var(--font-mono)" font-size="10">
            "Learn by doing"
          </text>
        </g>
      </svg>
      <p style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-dim); margin-top: var(--space-3);">
        Three Paradigms of Machine Learning
      </p>
    </div>
  `;
}
