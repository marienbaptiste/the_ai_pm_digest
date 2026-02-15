export function render(container) {
  container.innerHTML = `
    <div style="text-align: center; width: 100%;">
      <svg viewBox="0 0 600 360" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
        <style>
          .tl-box { animation: fadeIn 0.5s ease forwards; opacity: 0; }
          .tl-path { stroke-dasharray: 6; animation: dashFlow 1.2s linear infinite; }
          .tl-label { font-family: var(--font-mono); font-size: 10px; fill: var(--text-dim); }
          .tl-title { font-family: var(--font-heading); font-size: 12px; font-weight: 600; }
          @keyframes fadeIn { to { opacity: 1; } }
          @keyframes dashFlow { to { stroke-dashoffset: -12; } }
          @keyframes orbiting {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        </style>
        <defs>
          <marker id="tl-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="var(--accent-primary)"/>
          </marker>
        </defs>

        <!-- Center cycle indicator -->
        <circle cx="300" cy="180" r="100" fill="none" stroke="var(--border-subtle)" stroke-width="1" stroke-dasharray="4"/>

        <!-- Step 1: Data Batch -->
        <g class="tl-box" style="animation-delay: 0s;">
          <rect x="235" y="30" width="130" height="55" rx="10" fill="#3b82f615" stroke="#3b82f6" stroke-width="2"/>
          <text x="300" y="55" text-anchor="middle" class="tl-title" fill="#3b82f6">Data Batch</text>
          <text x="300" y="72" text-anchor="middle" class="tl-label">Mini-batch sampling</text>
        </g>

        <!-- Arrow 1: Data to Forward -->
        <path d="M365,65 Q420,65 420,115" fill="none" stroke="var(--accent-primary)" stroke-width="1.5" class="tl-path" marker-end="url(#tl-arrow)"/>

        <!-- Step 2: Forward Pass -->
        <g class="tl-box" style="animation-delay: 0.2s;">
          <rect x="370" y="120" width="130" height="55" rx="10" fill="#a855f715" stroke="#a855f7" stroke-width="2"/>
          <text x="435" y="145" text-anchor="middle" class="tl-title" fill="#a855f7">Forward Pass</text>
          <text x="435" y="162" text-anchor="middle" class="tl-label">y\u0302 = f(x; \u03B8)</text>
        </g>

        <!-- Arrow 2: Forward to Loss -->
        <path d="M435,175 Q435,220 390,240" fill="none" stroke="var(--accent-primary)" stroke-width="1.5" class="tl-path" marker-end="url(#tl-arrow)"/>

        <!-- Step 3: Loss Calculation -->
        <g class="tl-box" style="animation-delay: 0.4s;">
          <rect x="255" y="240" width="130" height="55" rx="10" fill="#ef444415" stroke="#ef4444" stroke-width="2"/>
          <text x="320" y="265" text-anchor="middle" class="tl-title" fill="#ef4444">Loss Function</text>
          <text x="320" y="282" text-anchor="middle" class="tl-label">L(y, y\u0302) = error</text>
        </g>

        <!-- Arrow 3: Loss to Backward -->
        <path d="M255,265 Q200,265 175,220" fill="none" stroke="var(--accent-primary)" stroke-width="1.5" class="tl-path" marker-end="url(#tl-arrow)"/>

        <!-- Step 4: Backward Pass -->
        <g class="tl-box" style="animation-delay: 0.6s;">
          <rect x="95" y="150" width="130" height="55" rx="10" fill="#f59e0b15" stroke="#f59e0b" stroke-width="2"/>
          <text x="160" y="175" text-anchor="middle" class="tl-title" fill="#f59e0b">Backward Pass</text>
          <text x="160" y="192" text-anchor="middle" class="tl-label">\u2207L / \u2207\u03B8 (gradients)</text>
        </g>

        <!-- Arrow 4: Backward to Update -->
        <path d="M160,150 Q160,100 205,80" fill="none" stroke="var(--accent-primary)" stroke-width="1.5" class="tl-path" marker-end="url(#tl-arrow)"/>

        <!-- Update weights label in center -->
        <g class="tl-box" style="animation-delay: 0.8s;">
          <circle cx="300" cy="180" r="35" fill="#00d4aa15" stroke="#00d4aa" stroke-width="2"/>
          <text x="300" y="176" text-anchor="middle" class="tl-title" fill="#00d4aa">Update</text>
          <text x="300" y="192" text-anchor="middle" class="tl-label">\u03B8 \u2192 \u03B8 - \u03B1\u2207L</text>
        </g>

        <!-- Orbiting particle -->
        <g style="transform-origin: 300px 180px; animation: orbiting 4s linear infinite;">
          <circle cx="300" cy="75" r="4" fill="var(--accent-primary)">
            <animate attributeName="opacity" values="1;0.4;1" dur="4s" repeatCount="indefinite"/>
          </circle>
        </g>

        <!-- Epoch counter -->
        <g class="tl-box" style="animation-delay: 1s;">
          <rect x="15" y="30" width="90" height="40" rx="8" fill="var(--bg-surface)" stroke="var(--border-subtle)" stroke-width="1"/>
          <text x="60" y="48" text-anchor="middle" class="tl-label" fill="var(--accent-primary)">Epoch 1..N</text>
          <text x="60" y="62" text-anchor="middle" class="tl-label">Repeat until</text>
        </g>

        <!-- Loss curve mini -->
        <g class="tl-box" style="animation-delay: 1.2s;">
          <rect x="460" y="270" width="120" height="75" rx="8" fill="var(--bg-surface)" stroke="var(--border-subtle)" stroke-width="1"/>
          <text x="520" y="290" text-anchor="middle" class="tl-label" fill="var(--accent-red)">Loss Curve</text>
          <polyline points="475,330 490,320 505,310 515,308 525,306 535,305 545,304 555,304 565,303"
            fill="none" stroke="var(--accent-red)" stroke-width="1.5" stroke-linecap="round"/>
          <line x1="475" y1="333" x2="570" y2="333" stroke="var(--border-subtle)" stroke-width="0.5"/>
          <line x1="475" y1="295" x2="475" y2="333" stroke="var(--border-subtle)" stroke-width="0.5"/>
        </g>
      </svg>
      <p style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-dim); margin-top: var(--space-3);">
        Machine Learning Training Loop \u2014 Data \u2192 Forward \u2192 Loss \u2192 Backward \u2192 Update
      </p>
    </div>
  `;
}
