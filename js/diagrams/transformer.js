export function render(container) {
  container.innerHTML = `
    <div style="text-align: center; width: 100%;">
      <svg viewBox="0 0 720 420" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
        <style>
          .tf-box { animation: fadeSlideUp 0.6s ease forwards; opacity: 0; }
          .tf-arrow { stroke-dasharray: 8; animation: drawArrow 1.5s linear infinite; }
          .tf-label { font-family: var(--font-mono); font-size: 10px; fill: var(--text-dim); }
          .tf-title { font-family: var(--font-heading); font-size: 13px; fill: var(--text-primary); font-weight: 600; }
          .tf-flow { animation: flowPulse 2s ease-in-out infinite; }
          @keyframes fadeSlideUp { to { opacity: 1; transform: translateY(0); } }
          @keyframes drawArrow { to { stroke-dashoffset: -16; } }
        </style>

        <!-- Input embedding -->
        <g class="tf-box" style="animation-delay: 0s;">
          <rect x="20" y="340" width="160" height="50" rx="8" fill="#3b82f620" stroke="#3b82f6" stroke-width="1.5"/>
          <text x="100" y="370" text-anchor="middle" class="tf-title" fill="#3b82f6">Input Embedding</text>
        </g>

        <!-- Positional encoding -->
        <g class="tf-box" style="animation-delay: 0.1s;">
          <rect x="200" y="340" width="140" height="50" rx="8" fill="#a855f720" stroke="#a855f7" stroke-width="1.5"/>
          <text x="270" y="362" text-anchor="middle" class="tf-title" fill="#a855f7">Positional</text>
          <text x="270" y="378" text-anchor="middle" class="tf-title" fill="#a855f7">Encoding</text>
        </g>

        <!-- Arrow down from embedding -->
        <line x1="100" y1="340" x2="100" y2="305" stroke="var(--border-medium)" stroke-width="1.5" class="tf-arrow"/>
        <line x1="270" y1="340" x2="270" y2="305" stroke="var(--border-medium)" stroke-width="1.5" class="tf-arrow"/>

        <!-- Add -->
        <g class="tf-box" style="animation-delay: 0.2s;">
          <circle cx="185" cy="310" r="15" fill="#f59e0b20" stroke="#f59e0b" stroke-width="1.5"/>
          <text x="185" y="314" text-anchor="middle" fill="#f59e0b" font-size="16" font-weight="700">+</text>
        </g>

        <!-- Multi-Head Attention block -->
        <g class="tf-box" style="animation-delay: 0.3s;">
          <rect x="60" y="200" width="250" height="70" rx="10" fill="#00d4aa15" stroke="#00d4aa" stroke-width="2"/>
          <text x="185" y="228" text-anchor="middle" class="tf-title" fill="#00d4aa">Multi-Head Attention</text>
          <text x="185" y="248" text-anchor="middle" class="tf-label">Q, K, V Projections \u2192 Scaled Dot-Product</text>
        </g>

        <!-- Q K V labels -->
        <g class="tf-box" style="animation-delay: 0.35s;">
          <rect x="80" y="258" width="36" height="22" rx="4" fill="#3b82f640" stroke="none"/>
          <text x="98" y="273" text-anchor="middle" fill="#3b82f6" font-size="10" font-weight="600">Q</text>
          <rect x="126" y="258" width="36" height="22" rx="4" fill="#a855f740" stroke="none"/>
          <text x="144" y="273" text-anchor="middle" fill="#a855f7" font-size="10" font-weight="600">K</text>
          <rect x="172" y="258" width="36" height="22" rx="4" fill="#ec489940" stroke="none"/>
          <text x="190" y="273" text-anchor="middle" fill="#ec4899" font-size="10" font-weight="600">V</text>
        </g>

        <line x1="185" y1="295" x2="185" y2="270" stroke="var(--border-medium)" stroke-width="1.5" class="tf-arrow"/>

        <!-- Add & Norm -->
        <g class="tf-box" style="animation-delay: 0.4s;">
          <rect x="340" y="210" width="120" height="45" rx="8" fill="#f59e0b20" stroke="#f59e0b" stroke-width="1.5"/>
          <text x="400" y="237" text-anchor="middle" class="tf-title" fill="#f59e0b">Add & Norm</text>
        </g>

        <line x1="310" y1="235" x2="340" y2="235" stroke="var(--border-medium)" stroke-width="1.5" class="tf-arrow"/>

        <!-- Feed Forward -->
        <g class="tf-box" style="animation-delay: 0.5s;">
          <rect x="340" y="130" width="120" height="55" rx="10" fill="#3b82f615" stroke="#3b82f6" stroke-width="2"/>
          <text x="400" y="155" text-anchor="middle" class="tf-title" fill="#3b82f6">Feed Forward</text>
          <text x="400" y="172" text-anchor="middle" class="tf-label">Linear \u2192 ReLU \u2192 Linear</text>
        </g>

        <line x1="400" y1="210" x2="400" y2="185" stroke="var(--border-medium)" stroke-width="1.5" class="tf-arrow"/>

        <!-- Add & Norm 2 -->
        <g class="tf-box" style="animation-delay: 0.6s;">
          <rect x="500" y="140" width="120" height="45" rx="8" fill="#f59e0b20" stroke="#f59e0b" stroke-width="1.5"/>
          <text x="560" y="167" text-anchor="middle" class="tf-title" fill="#f59e0b">Add & Norm</text>
        </g>

        <line x1="460" y1="157" x2="500" y2="157" stroke="var(--border-medium)" stroke-width="1.5" class="tf-arrow"/>

        <!-- Output -->
        <g class="tf-box" style="animation-delay: 0.7s;">
          <rect x="490" y="50" width="160" height="60" rx="10" fill="#00d4aa15" stroke="#00d4aa" stroke-width="2"/>
          <text x="570" y="76" text-anchor="middle" class="tf-title" fill="#00d4aa">Output Probabilities</text>
          <text x="570" y="96" text-anchor="middle" class="tf-label">Softmax over vocabulary</text>
        </g>

        <line x1="560" y1="140" x2="560" y2="110" stroke="var(--border-medium)" stroke-width="1.5" class="tf-arrow"/>

        <!-- Nx label -->
        <g class="tf-box" style="animation-delay: 0.8s;">
          <rect x="20" y="130" width="30" height="145" rx="6" fill="none" stroke="var(--text-dim)" stroke-width="1" stroke-dasharray="4"/>
          <text x="35" y="210" text-anchor="middle" fill="var(--text-dim)" font-size="12" font-weight="600" transform="rotate(-90, 35, 210)">N\u00D7</text>
        </g>

        <!-- Attention visualization mini -->
        <g class="tf-box" style="animation-delay: 0.9s;">
          <rect x="500" y="240" width="180" height="140" rx="10" fill="#11182790" stroke="var(--border-subtle)" stroke-width="1"/>
          <text x="590" y="262" text-anchor="middle" class="tf-title" fill="var(--accent-primary)">Attention Weights</text>
          <!-- Mini heatmap -->
          ${[0,1,2,3].map(r => [0,1,2,3].map(c => {
            const opacity = (0.2 + Math.random() * 0.8).toFixed(2);
            const delay = (1 + r * 0.1 + c * 0.1).toFixed(2);
            return `<rect x="${520 + c * 38}" y="${275 + r * 22}" width="32" height="18" rx="3"
              fill="var(--accent-primary)" opacity="${opacity}"
              class="tf-box" style="animation-delay: ${delay}s;">
              <animate attributeName="opacity" values="${opacity};${(parseFloat(opacity)*0.5).toFixed(2)};${opacity}" dur="3s" repeatCount="indefinite" begin="${delay}s"/>
            </rect>`;
          }).join('')).join('')}
          <text x="590" y="372" text-anchor="middle" class="tf-label">Scaled Dot-Product Scores</text>
        </g>
      </svg>
      <p style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-dim); margin-top: var(--space-3);">
        Transformer Architecture \u2014 Self-Attention Flow
      </p>
    </div>
  `;
}
