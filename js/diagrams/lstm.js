export function render(container) {
  container.innerHTML = `
    <div style="text-align: center; width: 100%;">
      <svg viewBox="0 0 750 400" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
        <style>
          .lstm-box {
            animation: lstmFadeIn 0.6s ease forwards;
            opacity: 0;
          }
          .lstm-label {
            animation: lstmFadeIn 0.5s ease forwards;
            opacity: 0;
          }
          .lstm-flow {
            stroke-dasharray: 8;
            animation: lstmFlowAnim 1.5s linear infinite;
          }
          .lstm-cell-state {
            stroke-dasharray: 300;
            stroke-dashoffset: 300;
            animation: lstmDrawLine 1.5s ease forwards 0.5s;
          }
          .lstm-gate-glow {
            animation: lstmGateGlow 2.5s ease-in-out infinite;
          }
          .lstm-pulse {
            animation: lstmPulse 3s ease-in-out infinite;
          }
          .lstm-data-flow {
            animation: lstmDataMove 2s ease-in-out infinite;
          }
          @keyframes lstmFadeIn {
            to { opacity: 1; }
          }
          @keyframes lstmFlowAnim {
            to { stroke-dashoffset: -16; }
          }
          @keyframes lstmDrawLine {
            to { stroke-dashoffset: 0; }
          }
          @keyframes lstmGateGlow {
            0%, 100% { filter: none; opacity: 0.85; }
            50% { filter: url(#lstmGlow); opacity: 1; }
          }
          @keyframes lstmPulse {
            0%, 100% { opacity: 0.4; }
            50% { opacity: 0.9; }
          }
          @keyframes lstmDataMove {
            0% { opacity: 0; }
            20% { opacity: 1; }
            80% { opacity: 1; }
            100% { opacity: 0; }
          }
          @keyframes lstmForgetActivate {
            0%, 30% { fill: #EB6F9230; }
            50% { fill: #EB6F9280; }
            70%, 100% { fill: #EB6F9230; }
          }
          @keyframes lstmInputActivate {
            0%, 35% { fill: #7EB8DA30; }
            55% { fill: #7EB8DA80; }
            75%, 100% { fill: #7EB8DA30; }
          }
          @keyframes lstmOutputActivate {
            0%, 50% { fill: #9CCFA430; }
            70% { fill: #9CCFA480; }
            90%, 100% { fill: #9CCFA430; }
          }
        </style>

        <defs>
          <filter id="lstmGlow">
            <feGaussianBlur in="SourceGraphic" stdDeviation="3"/>
          </filter>
          <marker id="lstmArrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--text-dim)"/>
          </marker>
          <marker id="lstmArrowRed" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#EB6F92"/>
          </marker>
          <marker id="lstmArrowBlue" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#7EB8DA"/>
          </marker>
          <marker id="lstmArrowTeal" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#9CCFA4"/>
          </marker>
          <marker id="lstmArrowAmber" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#F0B429"/>
          </marker>
        </defs>

        <!-- Title -->
        <text x="375" y="28" text-anchor="middle"
          fill="var(--text-primary)" font-family="var(--font-heading)" font-size="16" font-weight="700"
          class="lstm-label" style="animation-delay: 0s;">
          LSTM Cell \u2014 Gate Mechanisms
        </text>

        <!-- ===== Cell State Line (top horizontal) ===== -->
        <line x1="60" y1="85" x2="690" y2="85" stroke="#F0B429" stroke-width="3"
          class="lstm-cell-state" marker-end="url(#lstmArrowAmber)"/>
        <text x="375" y="70" text-anchor="middle" class="lstm-label" style="animation-delay: 0.8s;"
          fill="#F0B429" font-family="var(--font-heading)" font-size="13" font-weight="600">
          Cell State (C\u209C)
        </text>

        <!-- Animated pulse along cell state -->
        <circle r="5" fill="#F0B429" opacity="0">
          <animate attributeName="cx" values="80;670" dur="3s" repeatCount="indefinite" begin="1.5s"/>
          <animate attributeName="cy" values="85;85" dur="3s" repeatCount="indefinite" begin="1.5s"/>
          <animate attributeName="opacity" values="0;0.9;0.9;0" dur="3s" repeatCount="indefinite" begin="1.5s"/>
        </circle>
        <circle r="10" fill="#F0B429" opacity="0" filter="url(#lstmGlow)">
          <animate attributeName="cx" values="80;670" dur="3s" repeatCount="indefinite" begin="1.5s"/>
          <animate attributeName="cy" values="85;85" dur="3s" repeatCount="indefinite" begin="1.5s"/>
          <animate attributeName="opacity" values="0;0.3;0.3;0" dur="3s" repeatCount="indefinite" begin="1.5s"/>
        </circle>

        <!-- C(t-1) label -->
        <text x="40" y="80" text-anchor="middle" class="lstm-label" style="animation-delay: 0.6s;"
          fill="#F0B429" font-family="var(--font-mono)" font-size="10">C\u209C\u208B\u2081</text>

        <!-- C(t) label -->
        <text x="710" y="80" text-anchor="middle" class="lstm-label" style="animation-delay: 0.6s;"
          fill="#F0B429" font-family="var(--font-mono)" font-size="10">C\u209C</text>

        <!-- ===== Main cell body ===== -->
        <rect x="100" y="110" width="550" height="180" rx="16"
          fill="var(--bg-elevated)" stroke="var(--border-medium)" stroke-width="1.5"
          class="lstm-box" style="animation-delay: 0.2s;"/>

        <!-- ===== Forget Gate ===== -->
        <g class="lstm-box" style="animation-delay: 0.5s;">
          <rect x="140" y="145" width="100" height="70" rx="10"
            stroke="#EB6F92" stroke-width="2">
            <animate attributeName="fill" values="#EB6F9220;#EB6F9260;#EB6F9220" dur="3s" repeatCount="indefinite" begin="2s"/>
          </rect>
          <text x="190" y="172" text-anchor="middle"
            fill="#EB6F92" font-family="var(--font-heading)" font-size="12" font-weight="700">Forget Gate</text>
          <text x="190" y="200" text-anchor="middle"
            fill="#EB6F92" font-family="var(--font-mono)" font-size="18" font-weight="700">\u03C3</text>
        </g>

        <!-- Forget gate output up to cell state -->
        <line x1="190" y1="145" x2="190" y2="85" stroke="#EB6F92" stroke-width="1.5"
          class="lstm-flow" marker-end="url(#lstmArrowRed)"/>

        <!-- X (multiply) at forget gate junction -->
        <g class="lstm-box" style="animation-delay: 1s;">
          <circle cx="190" cy="85" r="12" fill="var(--bg-surface)" stroke="#EB6F92" stroke-width="1.5"/>
          <text x="190" y="90" text-anchor="middle"
            fill="#EB6F92" font-family="var(--font-mono)" font-size="14" font-weight="700">\u00D7</text>
        </g>

        <!-- ===== Input Gate ===== -->
        <g class="lstm-box" style="animation-delay: 0.8s;">
          <rect x="300" y="145" width="100" height="70" rx="10"
            stroke="#7EB8DA" stroke-width="2">
            <animate attributeName="fill" values="#7EB8DA20;#7EB8DA60;#7EB8DA20" dur="3s" repeatCount="indefinite" begin="2.5s"/>
          </rect>
          <text x="350" y="172" text-anchor="middle"
            fill="#7EB8DA" font-family="var(--font-heading)" font-size="12" font-weight="700">Input Gate</text>
          <text x="335" y="202" text-anchor="middle"
            fill="#7EB8DA" font-family="var(--font-mono)" font-size="16" font-weight="700">\u03C3</text>
          <text x="365" y="202" text-anchor="middle"
            fill="#7EB8DA" font-family="var(--font-mono)" font-size="12" font-weight="600">tanh</text>
        </g>

        <!-- Input gate output up to cell state -->
        <line x1="350" y1="145" x2="350" y2="85" stroke="#7EB8DA" stroke-width="1.5"
          class="lstm-flow" marker-end="url(#lstmArrowBlue)"/>

        <!-- + (add) at input gate junction -->
        <g class="lstm-box" style="animation-delay: 1.2s;">
          <circle cx="350" cy="85" r="12" fill="var(--bg-surface)" stroke="#7EB8DA" stroke-width="1.5"/>
          <text x="350" y="91" text-anchor="middle"
            fill="#7EB8DA" font-family="var(--font-mono)" font-size="16" font-weight="700">+</text>
        </g>

        <!-- ===== Output Gate ===== -->
        <g class="lstm-box" style="animation-delay: 1.1s;">
          <rect x="460" y="145" width="100" height="70" rx="10"
            stroke="#9CCFA4" stroke-width="2">
            <animate attributeName="fill" values="#9CCFA420;#9CCFA460;#9CCFA420" dur="3s" repeatCount="indefinite" begin="3s"/>
          </rect>
          <text x="510" y="172" text-anchor="middle"
            fill="#9CCFA4" font-family="var(--font-heading)" font-size="12" font-weight="700">Output Gate</text>
          <text x="495" y="202" text-anchor="middle"
            fill="#9CCFA4" font-family="var(--font-mono)" font-size="16" font-weight="700">\u03C3</text>
          <text x="525" y="202" text-anchor="middle"
            fill="#9CCFA4" font-family="var(--font-mono)" font-size="12" font-weight="600">tanh</text>
        </g>

        <!-- Output gate connection up -->
        <line x1="510" y1="145" x2="510" y2="85" stroke="#9CCFA4" stroke-width="1" stroke-dasharray="3"/>

        <!-- X (multiply) at output gate junction -->
        <g class="lstm-box" style="animation-delay: 1.4s;">
          <circle cx="510" cy="85" r="12" fill="var(--bg-surface)" stroke="#9CCFA4" stroke-width="1.5"/>
          <text x="510" y="90" text-anchor="middle"
            fill="#9CCFA4" font-family="var(--font-mono)" font-size="14" font-weight="700">\u00D7</text>
        </g>

        <!-- Output arrow going right and up -->
        <path d="M510,85 L510,55 L690,55" fill="none" stroke="#9CCFA4" stroke-width="2"
          class="lstm-flow" marker-end="url(#lstmArrowTeal)"/>
        <text x="620" y="48" text-anchor="middle" class="lstm-label" style="animation-delay: 1.6s;"
          fill="#9CCFA4" font-family="var(--font-heading)" font-size="12" font-weight="600">
          h\u209C (output)
        </text>

        <!-- ===== Input from below ===== -->
        <!-- x(t) input -->
        <line x1="280" y1="350" x2="280" y2="290" stroke="var(--text-dim)" stroke-width="1.5"
          class="lstm-flow" marker-end="url(#lstmArrow)"/>
        <text x="280" y="370" text-anchor="middle" class="lstm-label" style="animation-delay: 1.5s;"
          fill="var(--text-secondary)" font-family="var(--font-heading)" font-size="12" font-weight="600">
          x\u209C (input)
        </text>

        <!-- h(t-1) input -->
        <line x1="400" y1="350" x2="400" y2="290" stroke="var(--text-dim)" stroke-width="1.5"
          class="lstm-flow" marker-end="url(#lstmArrow)"/>
        <text x="400" y="370" text-anchor="middle" class="lstm-label" style="animation-delay: 1.5s;"
          fill="var(--text-secondary)" font-family="var(--font-heading)" font-size="12" font-weight="600">
          h\u209C\u208B\u2081 (prev)
        </text>

        <!-- Internal arrows from input to gates -->
        <g class="lstm-label" style="animation-delay: 1.6s;">
          <!-- Horizontal bus line inside cell -->
          <line x1="140" y1="265" x2="560" y2="265" stroke="var(--border-medium)" stroke-width="1.5"/>

          <!-- Input bus connected from x_t and h_{t-1} -->
          <line x1="280" y1="290" x2="280" y2="265" stroke="var(--text-dim)" stroke-width="1"/>
          <line x1="400" y1="290" x2="400" y2="265" stroke="var(--text-dim)" stroke-width="1"/>

          <!-- From bus to each gate -->
          <line x1="190" y1="265" x2="190" y2="215" stroke="#EB6F92" stroke-width="1" stroke-dasharray="3"
            marker-end="url(#lstmArrowRed)"/>
          <line x1="350" y1="265" x2="350" y2="215" stroke="#7EB8DA" stroke-width="1" stroke-dasharray="3"
            marker-end="url(#lstmArrowBlue)"/>
          <line x1="510" y1="265" x2="510" y2="215" stroke="#9CCFA4" stroke-width="1" stroke-dasharray="3"
            marker-end="url(#lstmArrowTeal)"/>
        </g>

        <!-- Data flow particles -->
        <circle r="3" fill="#EB6F92" opacity="0">
          <animate attributeName="cx" values="190;190" dur="2s" repeatCount="indefinite" begin="2s"/>
          <animate attributeName="cy" values="265;145" dur="2s" repeatCount="indefinite" begin="2s"/>
          <animate attributeName="opacity" values="0;1;1;0" dur="2s" repeatCount="indefinite" begin="2s"/>
        </circle>
        <circle r="3" fill="#7EB8DA" opacity="0">
          <animate attributeName="cx" values="350;350" dur="2s" repeatCount="indefinite" begin="2.4s"/>
          <animate attributeName="cy" values="265;145" dur="2s" repeatCount="indefinite" begin="2.4s"/>
          <animate attributeName="opacity" values="0;1;1;0" dur="2s" repeatCount="indefinite" begin="2.4s"/>
        </circle>
        <circle r="3" fill="#9CCFA4" opacity="0">
          <animate attributeName="cx" values="510;510" dur="2s" repeatCount="indefinite" begin="2.8s"/>
          <animate attributeName="cy" values="265;145" dur="2s" repeatCount="indefinite" begin="2.8s"/>
          <animate attributeName="opacity" values="0;1;1;0" dur="2s" repeatCount="indefinite" begin="2.8s"/>
        </circle>

        <!-- Legend -->
        <g class="lstm-label" style="animation-delay: 2s;">
          <text x="80" y="395" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="9">
            \u03C3 = sigmoid (0\u20131 gate)
          </text>
          <text x="250" y="395" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="9">
            tanh = candidate values (\u22121 to 1)
          </text>
          <text x="450" y="395" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="9">
            \u00D7 = element-wise multiply
          </text>
          <text x="630" y="395" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="9">
            + = element-wise add
          </text>
        </g>
      </svg>
      <p style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-dim); margin-top: var(--space-3);">
        LSTM Cell \u2014 Forget, Input, and Output Gate Mechanisms
      </p>
    </div>
  `;
}
