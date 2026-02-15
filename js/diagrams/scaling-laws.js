export function render(container) {
  container.innerHTML = `
    <div style="text-align: center; width: 100%;">
      <svg viewBox="0 0 750 420" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
        <style>
          .sl-fade { animation: slFadeIn 0.6s ease forwards; opacity: 0; }
          .sl-label { font-family: var(--font-mono); font-size: 10px; fill: var(--text-dim); }
          .sl-title { font-family: var(--font-heading); font-size: 13px; font-weight: 600; }
          .sl-axis { font-family: var(--font-mono); font-size: 9px; fill: var(--text-dim); }
          .sl-curve { stroke-dasharray: 500; stroke-dashoffset: 500; animation: slDraw 2.5s ease forwards; }
          .sl-dot { animation: slPopIn 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) forwards; opacity: 0; transform-origin: center; transform: scale(0); }
          .sl-emerge { animation: slEmerge 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) forwards; opacity: 0; transform: scale(0); transform-origin: center; }
          .sl-threshold { stroke-dasharray: 6 4; animation: slFadeIn 0.8s ease forwards; opacity: 0; }
          @keyframes slFadeIn { to { opacity: 1; } }
          @keyframes slDraw { to { stroke-dashoffset: 0; } }
          @keyframes slPopIn { to { opacity: 1; transform: scale(1); } }
          @keyframes slEmerge { to { opacity: 1; transform: scale(1); } }
          @keyframes slPulse { 0%, 100% { opacity: 0.7; } 50% { opacity: 1; } }
        </style>

        <defs>
          <linearGradient id="sl-curve-grad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stop-color="#9CCFA4"/>
            <stop offset="100%" stop-color="#7EB8DA"/>
          </linearGradient>
          <filter id="sl-glow">
            <feGaussianBlur stdDeviation="3" result="glow"/>
            <feMerge><feMergeNode in="glow"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>

        <!-- Chart background -->
        <g class="sl-fade" style="animation-delay: 0s;">
          <rect x="90" y="30" width="530" height="260" rx="4" fill="#0C0A0990" stroke="var(--border-subtle)" stroke-width="1"/>
        </g>

        <!-- Grid lines -->
        <g class="sl-fade" style="animation-delay: 0.1s;" opacity="0.15">
          <line x1="90" y1="95" x2="620" y2="95" stroke="var(--text-dim)" stroke-width="0.5"/>
          <line x1="90" y1="160" x2="620" y2="160" stroke="var(--text-dim)" stroke-width="0.5"/>
          <line x1="90" y1="225" x2="620" y2="225" stroke="var(--text-dim)" stroke-width="0.5"/>
          <line x1="200" y1="30" x2="200" y2="290" stroke="var(--text-dim)" stroke-width="0.5"/>
          <line x1="310" y1="30" x2="310" y2="290" stroke="var(--text-dim)" stroke-width="0.5"/>
          <line x1="420" y1="30" x2="420" y2="290" stroke="var(--text-dim)" stroke-width="0.5"/>
          <line x1="530" y1="30" x2="530" y2="290" stroke="var(--text-dim)" stroke-width="0.5"/>
        </g>

        <!-- Y-axis label -->
        <g class="sl-fade" style="animation-delay: 0.2s;">
          <text x="30" y="165" text-anchor="middle" class="sl-title" fill="var(--text-secondary)" transform="rotate(-90, 30, 165)">Performance (Loss)</text>
          <text x="75" y="55" class="sl-axis">High</text>
          <text x="75" y="280" class="sl-axis">Low</text>
        </g>

        <!-- X-axis label -->
        <g class="sl-fade" style="animation-delay: 0.2s;">
          <text x="355" y="310" text-anchor="middle" class="sl-title" fill="var(--text-secondary)">Compute (FLOPs) \u2192</text>
          <text x="120" y="302" class="sl-axis">10\u00B9</text>
          <text x="230" y="302" class="sl-axis">10\u00B9\u00B2</text>
          <text x="340" y="302" class="sl-axis">10\u00B9\u2075</text>
          <text x="450" y="302" class="sl-axis">10\u00B9\u2078</text>
          <text x="560" y="302" class="sl-axis">10\u00B2\u00B9</text>
        </g>

        <!-- Main scaling law curve (power law) -->
        <path d="M 110 60 C 160 75, 200 110, 250 140 C 300 170, 350 195, 400 215 C 450 230, 500 242, 560 252 C 590 256, 610 258, 620 260"
          fill="none" stroke="url(#sl-curve-grad)" stroke-width="3" class="sl-curve" filter="url(#sl-glow)"/>

        <!-- Glow underlay for curve -->
        <path d="M 110 60 C 160 75, 200 110, 250 140 C 300 170, 350 195, 400 215 C 450 230, 500 242, 560 252 C 590 256, 610 258, 620 260"
          fill="none" stroke="#9CCFA4" stroke-width="8" opacity="0.1" class="sl-curve"/>

        <!-- Model size dots on curve -->
        <g class="sl-dot" style="animation-delay: 2.6s;">
          <circle cx="155" cy="82" r="5" fill="#9CCFA4"/>
          <text x="155" y="75" text-anchor="middle" font-family="var(--font-heading)" font-size="11" fill="#9CCFA4" font-weight="600">1B</text>
        </g>

        <g class="sl-dot" style="animation-delay: 2.9s;">
          <circle cx="250" cy="140" r="5" fill="#9CCFA4"/>
          <text x="250" y="133" text-anchor="middle" font-family="var(--font-heading)" font-size="11" fill="#9CCFA4" font-weight="600">7B</text>
        </g>

        <g class="sl-dot" style="animation-delay: 3.2s;">
          <circle cx="380" cy="208" r="5" fill="#9CCFA4"/>
          <text x="380" y="201" text-anchor="middle" font-family="var(--font-heading)" font-size="11" fill="#9CCFA4" font-weight="600">70B</text>
        </g>

        <g class="sl-dot" style="animation-delay: 3.5s;">
          <circle cx="470" cy="236" r="6" fill="#7EB8DA"/>
          <text x="470" y="229" text-anchor="middle" font-family="var(--font-heading)" font-size="11" fill="#7EB8DA" font-weight="600">175B</text>
        </g>

        <g class="sl-dot" style="animation-delay: 3.8s;">
          <circle cx="565" cy="253" r="7" fill="#7EB8DA"/>
          <text x="565" y="246" text-anchor="middle" font-family="var(--font-heading)" font-size="11" fill="#7EB8DA" font-weight="600">540B</text>
        </g>

        <!-- Emergent abilities threshold line -->
        <line x1="380" y1="30" x2="380" y2="290" stroke="#EB6F92" stroke-width="1.5" class="sl-threshold" style="animation-delay: 4.0s;"/>
        <g class="sl-fade" style="animation-delay: 4.2s;">
          <rect x="383" y="35" width="120" height="20" rx="4" fill="#EB6F9220"/>
          <text x="443" y="49" text-anchor="middle" font-family="var(--font-heading)" font-size="10" fill="#EB6F92" font-weight="600">Emergent Threshold</text>
        </g>

        <!-- Emergent Abilities Section (below chart) -->
        <g class="sl-fade" style="animation-delay: 4.5s;">
          <text x="355" y="340" text-anchor="middle" class="sl-title" fill="#F0B429">Emergent Abilities</text>
          <text x="355" y="355" text-anchor="middle" class="sl-label">Capabilities that suddenly appear at scale</text>
        </g>

        <!-- Reasoning ability -->
        <g class="sl-emerge" style="animation-delay: 4.8s;">
          <rect x="130" y="365" width="110" height="40" rx="8" fill="#F0B42915" stroke="#F0B429" stroke-width="1.5"/>
          <text x="151" y="383" font-size="14">&#129504;</text>
          <text x="172" y="388" font-family="var(--font-heading)" font-size="11" fill="#F0B429" font-weight="600">Reasoning</text>
          <text x="185" y="400" text-anchor="middle" class="sl-label" fill="#F0B429">Chain-of-thought</text>
        </g>

        <!-- Coding ability -->
        <g class="sl-emerge" style="animation-delay: 5.1s;">
          <rect x="260" y="365" width="90" height="40" rx="8" fill="#F0B42915" stroke="#F0B429" stroke-width="1.5"/>
          <text x="278" y="383" font-size="14">&#128187;</text>
          <text x="303" y="388" font-family="var(--font-heading)" font-size="11" fill="#F0B429" font-weight="600">Coding</text>
          <text x="305" y="400" text-anchor="middle" class="sl-label" fill="#F0B429">Program synth</text>
        </g>

        <!-- Math ability -->
        <g class="sl-emerge" style="animation-delay: 5.4s;">
          <rect x="370" y="365" width="90" height="40" rx="8" fill="#F0B42915" stroke="#F0B429" stroke-width="1.5"/>
          <text x="388" y="383" font-size="14">&#128202;</text>
          <text x="417" y="388" font-family="var(--font-heading)" font-size="11" fill="#F0B429" font-weight="600">Math</text>
          <text x="415" y="400" text-anchor="middle" class="sl-label" fill="#F0B429">Multi-step</text>
        </g>

        <!-- Translation ability -->
        <g class="sl-emerge" style="animation-delay: 5.7s;">
          <rect x="480" y="365" width="120" height="40" rx="8" fill="#F0B42915" stroke="#F0B429" stroke-width="1.5"/>
          <text x="498" y="383" font-size="14">&#127760;</text>
          <text x="534" y="388" font-family="var(--font-heading)" font-size="11" fill="#F0B429" font-weight="600">Translation</text>
          <text x="540" y="400" text-anchor="middle" class="sl-label" fill="#F0B429">Zero-shot</text>
        </g>

        <!-- Connecting arrows from threshold to abilities -->
        <g class="sl-fade" style="animation-delay: 4.6s;">
          <line x1="380" y1="290" x2="380" y2="340" stroke="#EB6F92" stroke-width="1" stroke-dasharray="3 3" opacity="0.5"/>
          <polygon points="376,338 380,345 384,338" fill="#EB6F92" opacity="0.5"/>
        </g>

        <!-- Legend -->
        <g class="sl-fade" style="animation-delay: 0.5s;">
          <rect x="635" y="60" width="105" height="80" rx="6" fill="#0C0A0990" stroke="var(--border-subtle)" stroke-width="1"/>
          <line x1="645" y1="80" x2="665" y2="80" stroke="#9CCFA4" stroke-width="2"/>
          <text x="670" y="84" class="sl-label" fill="var(--text-secondary)">Scaling law</text>
          <line x1="645" y1="100" x2="665" y2="100" stroke="#EB6F92" stroke-width="1.5" stroke-dasharray="4 3"/>
          <text x="670" y="104" class="sl-label" fill="var(--text-secondary)">Threshold</text>
          <circle cx="655" cy="120" r="4" fill="#7EB8DA"/>
          <text x="670" y="124" class="sl-label" fill="var(--text-secondary)">Model size</text>
        </g>

      </svg>
      <p style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-dim); margin-top: var(--space-3);">
        Neural Scaling Laws \u2014 Power-Law Performance &amp; Emergent Abilities
      </p>
    </div>
  `;
}
