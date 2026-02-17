export function render(container) {
  container.innerHTML = `
    <div style="text-align: center; width: 100%;">
      <svg viewBox="0 0 750 400" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
        <style>
          .pl-fade { animation: plFadeIn 0.6s ease forwards; opacity: 0; }
          .pl-stage { animation: plSlideUp 0.7s cubic-bezier(0.34, 1.56, 0.64, 1) forwards; opacity: 0; transform: translateY(20px); }
          .pl-gate { animation: plFadeIn 0.5s ease forwards; opacity: 0; }
          .pl-arrow { stroke-dasharray: 6 3; animation: plFlow 1.2s linear infinite; }
          .pl-activity { animation: plFadeIn 0.5s ease forwards; opacity: 0; }
          .pl-label { font-family: var(--font-mono); font-size: 10px; fill: var(--text-dim); }
          .pl-title { font-family: var(--font-heading); font-size: 11px; font-weight: 600; }
          .pl-icon { font-size: 18px; }
          @keyframes plFadeIn { to { opacity: 1; } }
          @keyframes plSlideUp { to { opacity: 1; transform: translateY(0); } }
          @keyframes plFlow { to { stroke-dashoffset: -18; } }
          @keyframes plGateSpin { from { transform: rotate(45deg); } to { transform: rotate(405deg); } }
        </style>

        <defs>
          <marker id="pl-arrowhead" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
            <polygon points="0 0, 7 2.5, 0 5" fill="var(--border-medium)"/>
          </marker>
          <linearGradient id="pl-risk-grad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stop-color="#EB6F92" stop-opacity="0.6"/>
            <stop offset="40%" stop-color="#F0B429" stop-opacity="0.4"/>
            <stop offset="100%" stop-color="#9CCFA4" stop-opacity="0.6"/>
          </linearGradient>
        </defs>

        <!-- Risk <-> Certainty gradient bar (top) -->
        <g class="pl-fade" style="animation-delay: 3.0s;">
          <rect x="75" y="30" width="600" height="8" rx="4" fill="url(#pl-risk-grad)"/>
          <text x="75" y="25" font-family="var(--font-heading)" font-size="10" fill="#EB6F92" font-weight="600">\u2190 High Risk</text>
          <text x="675" y="25" text-anchor="end" font-family="var(--font-heading)" font-size="10" fill="#9CCFA4" font-weight="600">High Certainty \u2192</text>
          <text x="375" y="25" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="var(--text-dim)">Risk \u2194 Certainty</text>
        </g>

        <!-- ========== Stage 1: Research ========== -->
        <g class="pl-stage" style="animation-delay: 0.2s;">
          <rect x="45" y="70" width="110" height="100" rx="14" fill="#C4A7E712" stroke="#C4A7E7" stroke-width="2"/>
          <!-- Beaker icon -->
          <g transform="translate(82, 82)">
            <path d="M-6,-8 L-6,0 L-12,12 Q-12,16 -8,16 L8,16 Q12,16 12,12 L6,0 L6,-8" fill="#C4A7E720" stroke="#C4A7E7" stroke-width="1.5" stroke-linejoin="round"/>
            <line x1="-6" y1="-8" x2="6" y2="-8" stroke="#C4A7E7" stroke-width="1.5" stroke-linecap="round"/>
            <circle cx="-2" cy="8" r="2" fill="#C4A7E7" opacity="0.6"/>
            <circle cx="4" cy="11" r="1.5" fill="#C4A7E7" opacity="0.4"/>
          </g>
          <text x="100" y="120" text-anchor="middle" class="pl-title" fill="#C4A7E7">Research</text>
          <text x="100" y="135" text-anchor="middle" class="pl-label" fill="#C4A7E7">Explore &amp; validate</text>
          <!-- Step number -->
          <circle cx="55" cy="80" r="10" fill="#C4A7E7"/>
          <text x="55" y="84" text-anchor="middle" font-family="var(--font-mono)" font-size="9" fill="#fff" font-weight="700">1</text>
        </g>

        <!-- Arrow 1 to Arrow 2 -->
        <line x1="155" y1="120" x2="200" y2="120" stroke="var(--border-medium)" stroke-width="1.5" class="pl-arrow" marker-end="url(#pl-arrowhead)"/>

        <!-- Gate 1 diamond (on timeline) -->
        <g class="pl-gate" style="animation-delay: 0.7s;">
          <rect x="172.5" y="115" width="10" height="10" rx="1" fill="#F0B429" stroke="#F0B42980" stroke-width="1.5" transform="rotate(45, 177.5, 120)">
            <animateTransform attributeName="transform" type="rotate" values="45 177.5 120; 405 177.5 120; 45 177.5 120" dur="4s" repeatCount="indefinite"/>
          </rect>
        </g>

        <!-- ========== Stage 2: Prototype ========== -->
        <g class="pl-stage" style="animation-delay: 0.6s;">
          <rect x="200" y="70" width="110" height="100" rx="14" fill="#7EB8DA12" stroke="#7EB8DA" stroke-width="2"/>
          <!-- Flask icon -->
          <g transform="translate(238, 82)">
            <path d="M-4,-10 L-4,-2 L-12,12 Q-12,16 -8,16 L8,16 Q12,16 12,12 L4,-2 L4,-10" fill="#7EB8DA20" stroke="#7EB8DA" stroke-width="1.5" stroke-linejoin="round"/>
            <line x1="-5" y1="-10" x2="5" y2="-10" stroke="#7EB8DA" stroke-width="1.5" stroke-linecap="round"/>
            <line x1="-8" y1="6" x2="8" y2="6" stroke="#7EB8DA" stroke-width="1" opacity="0.5"/>
          </g>
          <text x="255" y="120" text-anchor="middle" class="pl-title" fill="#7EB8DA">Prototype</text>
          <text x="255" y="135" text-anchor="middle" class="pl-label" fill="#7EB8DA">Build &amp; test</text>
          <circle cx="210" cy="80" r="10" fill="#7EB8DA"/>
          <text x="210" y="84" text-anchor="middle" font-family="var(--font-mono)" font-size="9" fill="#fff" font-weight="700">2</text>
        </g>

        <!-- Arrow 2 to Arrow 3 -->
        <line x1="310" y1="120" x2="355" y2="120" stroke="var(--border-medium)" stroke-width="1.5" class="pl-arrow" marker-end="url(#pl-arrowhead)"/>

        <!-- Gate 2 diamond (on timeline) -->
        <g class="pl-gate" style="animation-delay: 1.1s;">
          <rect x="327.5" y="115" width="10" height="10" rx="1" fill="#F0B429" stroke="#F0B42980" stroke-width="1.5" transform="rotate(45, 332.5, 120)">
            <animateTransform attributeName="transform" type="rotate" values="45 332.5 120; 405 332.5 120; 45 332.5 120" dur="4s" repeatCount="indefinite"/>
          </rect>
        </g>

        <!-- ========== Stage 3: MVP ========== -->
        <g class="pl-stage" style="animation-delay: 1.0s;">
          <rect x="355" y="70" width="110" height="100" rx="14" fill="#9CCFA412" stroke="#9CCFA4" stroke-width="2"/>
          <!-- Rocket icon -->
          <g transform="translate(393, 82)">
            <path d="M0,-12 Q-6,-6 -6,4 L-3,8 L-3,12 L3,12 L3,8 L6,4 Q6,-6 0,-12Z" fill="#9CCFA420" stroke="#9CCFA4" stroke-width="1.5"/>
            <circle cx="0" cy="0" r="2.5" fill="#9CCFA4" opacity="0.6"/>
            <!-- Flames -->
            <path d="M-2,12 L0,18 L2,12" fill="#F0B429" opacity="0.7">
              <animate attributeName="opacity" values="0.4;0.8;0.4" dur="0.5s" repeatCount="indefinite"/>
            </path>
          </g>
          <text x="410" y="120" text-anchor="middle" class="pl-title" fill="#9CCFA4">MVP</text>
          <text x="410" y="135" text-anchor="middle" class="pl-label" fill="#9CCFA4">Launch &amp; learn</text>
          <circle cx="365" cy="80" r="10" fill="#9CCFA4"/>
          <text x="365" y="84" text-anchor="middle" font-family="var(--font-mono)" font-size="9" fill="#0C0A09" font-weight="700">3</text>
        </g>

        <!-- Arrow 3 to Arrow 4 -->
        <line x1="465" y1="120" x2="510" y2="120" stroke="var(--border-medium)" stroke-width="1.5" class="pl-arrow" marker-end="url(#pl-arrowhead)"/>

        <!-- Gate 3 diamond (on timeline) -->
        <g class="pl-gate" style="animation-delay: 1.5s;">
          <rect x="482.5" y="115" width="10" height="10" rx="1" fill="#F0B429" stroke="#F0B42980" stroke-width="1.5" transform="rotate(45, 487.5, 120)">
            <animateTransform attributeName="transform" type="rotate" values="45 487.5 120; 405 487.5 120; 45 487.5 120" dur="4s" repeatCount="indefinite"/>
          </rect>
        </g>

        <!-- ========== Stage 4: Scale ========== -->
        <g class="pl-stage" style="animation-delay: 1.4s;">
          <rect x="510" y="70" width="110" height="100" rx="14" fill="#F0B42912" stroke="#F0B429" stroke-width="2"/>
          <!-- Chart icon -->
          <g transform="translate(548, 84)">
            <rect x="-10" y="6" width="6" height="10" rx="1" fill="#F0B429" opacity="0.5"/>
            <rect x="-2" y="0" width="6" height="16" rx="1" fill="#F0B429" opacity="0.7"/>
            <rect x="6" y="-6" width="6" height="22" rx="1" fill="#F0B429"/>
            <polyline points="-7,4 -1,-2 7,-8" fill="none" stroke="#F0B429" stroke-width="1.5" stroke-linecap="round"/>
          </g>
          <text x="565" y="120" text-anchor="middle" class="pl-title" fill="#F0B429">Scale</text>
          <text x="565" y="135" text-anchor="middle" class="pl-label" fill="#F0B429">Grow &amp; optimize</text>
          <circle cx="520" cy="80" r="10" fill="#F0B429"/>
          <text x="520" y="84" text-anchor="middle" font-family="var(--font-mono)" font-size="9" fill="#0C0A09" font-weight="700">4</text>
        </g>

        <!-- Arrow 4 to Arrow 5 -->
        <line x1="620" y1="120" x2="660" y2="120" stroke="var(--border-medium)" stroke-width="1.5" class="pl-arrow" marker-end="url(#pl-arrowhead)"/>

        <!-- Gate 4 diamond (on timeline) -->
        <g class="pl-gate" style="animation-delay: 1.9s;">
          <rect x="635" y="115" width="10" height="10" rx="1" fill="#F0B429" stroke="#F0B42980" stroke-width="1.5" transform="rotate(45, 640, 120)">
            <animateTransform attributeName="transform" type="rotate" values="45 640 120; 405 640 120; 45 640 120" dur="4s" repeatCount="indefinite"/>
          </rect>
        </g>

        <!-- ========== Stage 5: Mature ========== -->
        <g class="pl-stage" style="animation-delay: 1.8s;">
          <rect x="660" y="70" width="75" height="100" rx="14" fill="#9CCFA412" stroke="#9CCFA4" stroke-width="2"/>
          <!-- Checkmark icon -->
          <g transform="translate(685, 90)">
            <circle cx="0" cy="0" r="10" fill="#9CCFA420" stroke="#9CCFA4" stroke-width="1.5"/>
            <polyline points="-5,0 -1,5 7,-5" fill="none" stroke="#9CCFA4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </g>
          <text x="697" y="120" text-anchor="middle" class="pl-title" fill="#9CCFA4">Mature</text>
          <text x="697" y="135" text-anchor="middle" class="pl-label" fill="#9CCFA4">Maintain</text>
          <circle cx="670" cy="80" r="10" fill="#9CCFA4"/>
          <text x="670" y="84" text-anchor="middle" font-family="var(--font-mono)" font-size="9" fill="#0C0A09" font-weight="700">5</text>
        </g>

        <!-- ========== Activities Below Each Stage ========== -->

        <!-- Research activities -->
        <g class="pl-activity" style="animation-delay: 2.2s;">
          <rect x="45" y="185" width="110" height="80" rx="8" fill="#0C0A0980" stroke="var(--border-subtle)" stroke-width="0.8"/>
          <text x="100" y="202" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#C4A7E7">\u2022 Literature review</text>
          <text x="100" y="216" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#C4A7E7">\u2022 Feasibility study</text>
          <text x="100" y="230" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#C4A7E7">\u2022 Data assessment</text>
          <text x="100" y="244" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#C4A7E7">\u2022 Risk analysis</text>
        </g>

        <!-- Prototype activities -->
        <g class="pl-activity" style="animation-delay: 2.4s;">
          <rect x="200" y="185" width="110" height="80" rx="8" fill="#0C0A0980" stroke="var(--border-subtle)" stroke-width="0.8"/>
          <text x="255" y="202" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#7EB8DA">\u2022 Model training</text>
          <text x="255" y="216" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#7EB8DA">\u2022 A/B experiments</text>
          <text x="255" y="230" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#7EB8DA">\u2022 Eval framework</text>
          <text x="255" y="244" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#7EB8DA">\u2022 User testing</text>
        </g>

        <!-- MVP activities -->
        <g class="pl-activity" style="animation-delay: 2.6s;">
          <rect x="355" y="185" width="110" height="80" rx="8" fill="#0C0A0980" stroke="var(--border-subtle)" stroke-width="0.8"/>
          <text x="410" y="202" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#9CCFA4">\u2022 Ship to users</text>
          <text x="410" y="216" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#9CCFA4">\u2022 Gather feedback</text>
          <text x="410" y="230" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#9CCFA4">\u2022 Monitor metrics</text>
          <text x="410" y="244" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#9CCFA4">\u2022 Iterate fast</text>
        </g>

        <!-- Scale activities -->
        <g class="pl-activity" style="animation-delay: 2.8s;">
          <rect x="510" y="185" width="110" height="80" rx="8" fill="#0C0A0980" stroke="var(--border-subtle)" stroke-width="0.8"/>
          <text x="565" y="202" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#F0B429">\u2022 Infra scaling</text>
          <text x="565" y="216" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#F0B429">\u2022 Cost optimization</text>
          <text x="565" y="230" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#F0B429">\u2022 Edge cases</text>
          <text x="565" y="244" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#F0B429">\u2022 Team growth</text>
        </g>

        <!-- Mature activities -->
        <g class="pl-activity" style="animation-delay: 3.0s;">
          <rect x="650" y="185" width="92" height="80" rx="8" fill="#0C0A0980" stroke="var(--border-subtle)" stroke-width="0.8"/>
          <text x="696" y="202" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#9CCFA4">\u2022 Maintenance</text>
          <text x="696" y="216" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#9CCFA4">\u2022 Model refresh</text>
          <text x="696" y="230" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#9CCFA4">\u2022 Compliance</text>
          <text x="696" y="244" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#9CCFA4">\u2022 Deprecation</text>
        </g>

        <!-- Connecting dashed lines from stages to activities -->
        <g class="pl-fade" style="animation-delay: 2.0s;" opacity="0.3">
          <line x1="100" y1="170" x2="100" y2="185" stroke="var(--text-dim)" stroke-width="1" stroke-dasharray="2"/>
          <line x1="255" y1="170" x2="255" y2="185" stroke="var(--text-dim)" stroke-width="1" stroke-dasharray="2"/>
          <line x1="410" y1="170" x2="410" y2="185" stroke="var(--text-dim)" stroke-width="1" stroke-dasharray="2"/>
          <line x1="565" y1="170" x2="565" y2="185" stroke="var(--text-dim)" stroke-width="1" stroke-dasharray="2"/>
          <line x1="697" y1="170" x2="697" y2="185" stroke="var(--text-dim)" stroke-width="1" stroke-dasharray="2"/>
        </g>

        <!-- Gate legend -->
        <g class="pl-fade" style="animation-delay: 3.3s;">
          <rect x="270" y="290" width="210" height="50" rx="8" fill="#0C0A0990" stroke="var(--border-subtle)" stroke-width="1"/>
          <rect x="288" y="308" width="10" height="10" rx="1" fill="#F0B429" stroke="#F0B42980" stroke-width="1.5" transform="rotate(45, 293, 313)"/>
          <text x="310" y="318" font-family="var(--font-mono)" font-size="9" fill="var(--text-secondary)">= Milestone gate (Go/No-Go)</text>
          <text x="375" y="334" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="var(--text-dim)">Each gate requires stakeholder approval</text>
        </g>

        <!-- Title -->
        <g class="pl-fade" style="animation-delay: 0s;">
          <text x="375" y="378" text-anchor="middle" font-family="var(--font-heading)" font-size="14" fill="var(--text-primary)" font-weight="700">AI Product Lifecycle</text>
          <text x="375" y="395" text-anchor="middle" font-family="var(--font-mono)" font-size="9" fill="var(--text-dim)">From Research to Production</text>
        </g>

      </svg>
      <p style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-dim); margin-top: var(--space-3);">
        AI Product Lifecycle \u2014 Stage-Gate Framework from Research to Maturity
      </p>
    </div>
  `;
}
