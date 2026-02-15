export function render(container) {
  container.innerHTML = `
    <div style="text-align: center; width: 100%;">
      <svg viewBox="0 0 750 400" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
        <style>
          .rlhf-box { animation: rlhfFadeIn 0.7s ease forwards; opacity: 0; }
          .rlhf-arrow { stroke-dasharray: 8 4; animation: rlhfFlow 1.2s linear infinite; }
          .rlhf-label { font-family: var(--font-mono); font-size: 10px; fill: var(--text-dim); }
          .rlhf-title { font-family: var(--font-heading); font-size: 13px; font-weight: 600; }
          .rlhf-step { font-family: var(--font-mono); font-size: 9px; font-weight: 700; }
          .rlhf-icon { animation: rlhfFadeIn 0.5s ease forwards; opacity: 0; }
          .rlhf-responses { animation: rlhfFadeIn 0.4s ease forwards; opacity: 0; }
          @keyframes rlhfFadeIn { to { opacity: 1; } }
          @keyframes rlhfFlow { to { stroke-dashoffset: -24; } }
          @keyframes rlhfPulse { 0%, 100% { opacity: 0.6; } 50% { opacity: 1; } }
          @keyframes rlhfGlow { 0%, 100% { filter: drop-shadow(0 0 2px currentColor); } 50% { filter: drop-shadow(0 0 8px currentColor); } }
        </style>

        <defs>
          <marker id="rlhf-arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="var(--border-medium)"/>
          </marker>
          <marker id="rlhf-arrowhead-teal" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#00d4aa"/>
          </marker>
        </defs>

        <!-- Central cycle label -->
        <g class="rlhf-box" style="animation-delay: 2.5s;">
          <text x="375" y="205" text-anchor="middle" font-family="var(--font-heading)" font-size="11" fill="var(--text-dim)" opacity="0.7">RLHF</text>
          <text x="375" y="220" text-anchor="middle" font-family="var(--font-heading)" font-size="11" fill="var(--text-dim)" opacity="0.7">Cycle</text>
        </g>

        <!-- Step 1: Base LLM (top-left) -->
        <g class="rlhf-box" style="animation-delay: 0s;">
          <rect x="90" y="50" width="180" height="80" rx="12" fill="#a855f715" stroke="#a855f7" stroke-width="2"/>
          <!-- Brain icon -->
          <g transform="translate(130, 70)">
            <ellipse cx="12" cy="10" rx="11" ry="12" fill="none" stroke="#a855f7" stroke-width="1.5"/>
            <path d="M7 10 Q12 2 17 10" fill="none" stroke="#a855f7" stroke-width="1.2"/>
            <path d="M7 14 Q12 6 17 14" fill="none" stroke="#a855f7" stroke-width="1.2"/>
            <line x1="12" y1="0" x2="12" y2="22" stroke="#a855f7" stroke-width="1.2"/>
          </g>
          <text x="195" y="85" text-anchor="middle" class="rlhf-title" fill="#a855f7">Base LLM</text>
          <text x="195" y="105" text-anchor="middle" class="rlhf-label">Generates responses</text>
          <!-- Step badge -->
          <circle cx="100" cy="60" r="11" fill="#a855f7"/>
          <text x="100" y="64" text-anchor="middle" class="rlhf-step" fill="#fff">1</text>
        </g>

        <!-- Responses bubbles from LLM -->
        <g class="rlhf-responses" style="animation-delay: 0.5s;">
          <rect x="300" y="42" width="80" height="22" rx="6" fill="#a855f720" stroke="#a855f7" stroke-width="1"/>
          <text x="340" y="57" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#a855f7">Response A</text>
          <rect x="300" y="68" width="80" height="22" rx="6" fill="#a855f720" stroke="#a855f7" stroke-width="1"/>
          <text x="340" y="83" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#a855f7">Response B</text>
          <rect x="300" y="94" width="80" height="22" rx="6" fill="#a855f720" stroke="#a855f7" stroke-width="1"/>
          <text x="340" y="109" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#a855f7">Response C</text>
        </g>

        <!-- Arrow: LLM -> Responses -->
        <line x1="270" y1="90" x2="296" y2="80" stroke="var(--border-medium)" stroke-width="1.5" class="rlhf-arrow" marker-end="url(#rlhf-arrowhead)"/>

        <!-- Step 2: Human Evaluator (top-right) -->
        <g class="rlhf-box" style="animation-delay: 0.6s;">
          <rect x="470" y="50" width="190" height="80" rx="12" fill="#f59e0b15" stroke="#f59e0b" stroke-width="2"/>
          <!-- Human silhouette icon -->
          <g transform="translate(500, 65)">
            <circle cx="12" cy="6" r="6" fill="none" stroke="#f59e0b" stroke-width="1.5"/>
            <path d="M0 24 Q0 14 12 14 Q24 14 24 24" fill="none" stroke="#f59e0b" stroke-width="1.5"/>
          </g>
          <text x="580" y="85" text-anchor="middle" class="rlhf-title" fill="#f59e0b">Human Evaluator</text>
          <text x="580" y="105" text-anchor="middle" class="rlhf-label">Ranks preferences</text>
          <!-- Step badge -->
          <circle cx="480" cy="60" r="11" fill="#f59e0b"/>
          <text x="480" y="64" text-anchor="middle" class="rlhf-step" fill="#fff">2</text>
        </g>

        <!-- Ranking indicators -->
        <g class="rlhf-box" style="animation-delay: 1.0s;">
          <text x="632" y="74" font-family="var(--font-mono)" font-size="14" fill="#22c55e">&#9733;</text>
          <text x="645" y="74" font-family="var(--font-mono)" font-size="14" fill="#22c55e">&#9733;</text>
          <text x="658" y="74" font-family="var(--font-mono)" font-size="14" fill="#22c55e">&#9733;</text>
          <text x="632" y="100" font-family="var(--font-mono)" font-size="10" fill="var(--text-dim)">A &gt; C &gt; B</text>
        </g>

        <!-- Arrow: Responses -> Human -->
        <line x1="384" y1="80" x2="466" y2="80" stroke="var(--border-medium)" stroke-width="1.5" class="rlhf-arrow" marker-end="url(#rlhf-arrowhead)"/>

        <!-- Step 3: Reward Model (bottom-right) -->
        <g class="rlhf-box" style="animation-delay: 1.2s;">
          <rect x="470" y="260" width="190" height="80" rx="12" fill="#3b82f615" stroke="#3b82f6" stroke-width="2"/>
          <!-- Chart icon -->
          <g transform="translate(500, 275)">
            <rect x="0" y="14" width="6" height="10" rx="1" fill="#3b82f6" opacity="0.6"/>
            <rect x="8" y="8" width="6" height="16" rx="1" fill="#3b82f6" opacity="0.8"/>
            <rect x="16" y="2" width="6" height="22" rx="1" fill="#3b82f6"/>
          </g>
          <text x="577" y="295" text-anchor="middle" class="rlhf-title" fill="#3b82f6">Reward Model</text>
          <text x="577" y="315" text-anchor="middle" class="rlhf-label">Learns human preferences</text>
          <!-- Step badge -->
          <circle cx="480" cy="270" r="11" fill="#3b82f6"/>
          <text x="480" y="274" text-anchor="middle" class="rlhf-step" fill="#fff">3</text>
        </g>

        <!-- Arrow: Human -> Reward Model (right side, going down) -->
        <path d="M 660 130 L 660 270 L 660 290" fill="none" stroke="var(--border-medium)" stroke-width="1.5" class="rlhf-arrow" marker-end="url(#rlhf-arrowhead)"/>
        <text x="672" y="210" class="rlhf-label" fill="var(--text-dim)" font-size="9" transform="rotate(90, 672, 210)">Preference data</text>

        <!-- Step 4: PPO/Policy Update (bottom-left) -->
        <g class="rlhf-box" style="animation-delay: 1.8s;">
          <rect x="90" y="260" width="180" height="80" rx="12" fill="#00d4aa15" stroke="#00d4aa" stroke-width="2"/>
          <!-- Gear/update icon -->
          <g transform="translate(120, 278)">
            <circle cx="12" cy="12" r="10" fill="none" stroke="#00d4aa" stroke-width="1.5"/>
            <circle cx="12" cy="12" r="4" fill="none" stroke="#00d4aa" stroke-width="1.5"/>
            <line x1="12" y1="0" x2="12" y2="4" stroke="#00d4aa" stroke-width="2"/>
            <line x1="12" y1="20" x2="12" y2="24" stroke="#00d4aa" stroke-width="2"/>
            <line x1="0" y1="12" x2="4" y2="12" stroke="#00d4aa" stroke-width="2"/>
            <line x1="20" y1="12" x2="24" y2="12" stroke="#00d4aa" stroke-width="2"/>
          </g>
          <text x="200" y="295" text-anchor="middle" class="rlhf-title" fill="#00d4aa">PPO / Policy Update</text>
          <text x="200" y="315" text-anchor="middle" class="rlhf-label">Optimizes LLM policy</text>
          <!-- Step badge -->
          <circle cx="100" cy="270" r="11" fill="#00d4aa"/>
          <text x="100" y="274" text-anchor="middle" class="rlhf-step" fill="#111827">4</text>
        </g>

        <!-- Arrow: Reward Model -> PPO (bottom, going left) -->
        <line x1="470" y1="300" x2="274" y2="300" stroke="var(--border-medium)" stroke-width="1.5" class="rlhf-arrow" marker-end="url(#rlhf-arrowhead)"/>
        <text x="375" y="290" text-anchor="middle" class="rlhf-label" fill="var(--text-dim)">Reward signal</text>

        <!-- Arrow: PPO -> LLM (left side, going up) - THE CYCLE! -->
        <path d="M 90 280 L 50 280 L 50 90 L 86 90" fill="none" stroke="#00d4aa" stroke-width="2" class="rlhf-arrow" marker-end="url(#rlhf-arrowhead-teal)"/>
        <text x="40" y="195" class="rlhf-label" fill="#00d4aa" font-size="9" transform="rotate(-90, 40, 195)">Updated weights</text>

        <!-- Iteration indicator -->
        <g class="rlhf-box" style="animation-delay: 2.8s;">
          <rect x="290" y="350" width="170" height="36" rx="8" fill="#00d4aa10" stroke="#00d4aa" stroke-width="1" stroke-dasharray="4"/>
          <text x="375" y="365" text-anchor="middle" font-family="var(--font-mono)" font-size="9" fill="#00d4aa">
            <animate attributeName="opacity" values="0.5;1;0.5" dur="2s" repeatCount="indefinite"/>
            &#8635; Iterative refinement loop
          </text>
          <text x="375" y="380" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="var(--text-dim)">Repeat until alignment converges</text>
        </g>

        <!-- Reward score flowing -->
        <g class="rlhf-box" style="animation-delay: 2.0s;">
          <rect x="380" y="315" width="56" height="18" rx="4" fill="#3b82f630"/>
          <text x="408" y="328" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#3b82f6">r = 0.87</text>
        </g>

      </svg>
      <p style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-dim); margin-top: var(--space-3);">
        RLHF Pipeline \u2014 Aligning Language Models with Human Preferences
      </p>
    </div>
  `;
}
