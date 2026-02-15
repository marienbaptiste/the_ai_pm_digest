export function render(container) {
  container.innerHTML = `
    <div style="text-align: center; width: 100%;">
      <svg viewBox="0 0 750 340" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
        <style>
          .rag-box { animation: fadeSlideUp 0.5s ease forwards; opacity: 0; }
          .rag-arrow { stroke-dasharray: 6; animation: dashFlow 1s linear infinite; }
          .rag-label { font-family: var(--font-mono); font-size: 9px; fill: var(--text-dim); }
          .rag-title { font-family: var(--font-heading); font-size: 12px; font-weight: 600; }
          @keyframes fadeSlideUp { to { opacity: 1; } }
          @keyframes dashFlow { to { stroke-dashoffset: -12; } }
        </style>
        <defs>
          <marker id="rag-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="var(--accent-primary)"/>
          </marker>
          <marker id="rag-arrow-blue" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="var(--accent-blue)"/>
          </marker>
        </defs>

        <!-- Step 1: User Query -->
        <g class="rag-box" style="animation-delay: 0s;">
          <rect x="20" y="120" width="120" height="70" rx="10" fill="#3b82f615" stroke="#3b82f6" stroke-width="2"/>
          <text x="80" y="148" text-anchor="middle" class="rag-title" fill="#3b82f6">User Query</text>
          <text x="80" y="168" text-anchor="middle" class="rag-label">"How does RAG work?"</text>
        </g>

        <!-- Arrow 1 -->
        <line x1="140" y1="155" x2="175" y2="155" stroke="var(--accent-blue)" stroke-width="1.5" class="rag-arrow" marker-end="url(#rag-arrow-blue)"
          style="animation-delay: 0.2s;"/>

        <!-- Step 2: Embedding Model -->
        <g class="rag-box" style="animation-delay: 0.2s;">
          <rect x="180" y="110" width="120" height="90" rx="10" fill="#a855f715" stroke="#a855f7" stroke-width="2"/>
          <text x="240" y="142" text-anchor="middle" class="rag-title" fill="#a855f7">Embedding</text>
          <text x="240" y="158" text-anchor="middle" class="rag-title" fill="#a855f7">Model</text>
          <text x="240" y="180" text-anchor="middle" class="rag-label">[0.23, -0.41, 0.87...]</text>
        </g>

        <!-- Arrow 2 -->
        <line x1="300" y1="155" x2="335" y2="155" stroke="var(--accent-primary)" stroke-width="1.5" class="rag-arrow" marker-end="url(#rag-arrow)"
          style="animation-delay: 0.4s;"/>

        <!-- Step 3: Vector DB -->
        <g class="rag-box" style="animation-delay: 0.4s;">
          <rect x="340" y="90" width="130" height="130" rx="10" fill="#f59e0b15" stroke="#f59e0b" stroke-width="2"/>
          <text x="405" y="118" text-anchor="middle" class="rag-title" fill="#f59e0b">Vector Database</text>

          <!-- Mini vector dots -->
          ${[0,1,2,3,4,5,6,7].map(i => {
            const cx = 365 + (i % 4) * 25;
            const cy = 140 + Math.floor(i / 4) * 30;
            const isMatch = i === 1 || i === 5;
            return `<circle cx="${cx}" cy="${cy}" r="6" fill="${isMatch ? '#00d4aa' : '#f59e0b30'}" stroke="${isMatch ? '#00d4aa' : '#f59e0b50'}" stroke-width="1">
              ${isMatch ? `<animate attributeName="r" values="6;8;6" dur="1.5s" repeatCount="indefinite" begin="${(0.6 + i * 0.1).toFixed(1)}s"/>` : ''}
            </circle>`;
          }).join('')}

          <text x="405" y="200" text-anchor="middle" class="rag-label">Similarity Search</text>
        </g>

        <!-- Arrow 3 -->
        <line x1="470" y1="155" x2="505" y2="155" stroke="var(--accent-primary)" stroke-width="1.5" class="rag-arrow" marker-end="url(#rag-arrow)"
          style="animation-delay: 0.6s;"/>

        <!-- Step 4: Retrieved Context -->
        <g class="rag-box" style="animation-delay: 0.6s;">
          <rect x="510" y="100" width="100" height="110" rx="10" fill="#00d4aa15" stroke="#00d4aa" stroke-width="2"/>
          <text x="560" y="128" text-anchor="middle" class="rag-title" fill="#00d4aa">Retrieved</text>
          <text x="560" y="144" text-anchor="middle" class="rag-title" fill="#00d4aa">Context</text>

          <!-- Mini doc icons -->
          <rect x="530" y="155" width="60" height="12" rx="2" fill="#00d4aa30" stroke="none"/>
          <rect x="530" y="172" width="50" height="12" rx="2" fill="#00d4aa20" stroke="none"/>
          <rect x="530" y="189" width="55" height="12" rx="2" fill="#00d4aa20" stroke="none"/>
        </g>

        <!-- Arrow 4 (down to LLM) -->
        <line x1="560" y1="210" x2="560" y2="240" stroke="var(--accent-primary)" stroke-width="1.5" class="rag-arrow" marker-end="url(#rag-arrow)"
          style="animation-delay: 0.8s;"/>

        <!-- Query also goes to LLM -->
        <path d="M80,190 L80,280 L400,280" fill="none" stroke="var(--accent-blue)" stroke-width="1.5" class="rag-arrow" marker-end="url(#rag-arrow-blue)"
          style="animation-delay: 0.3s;"/>
        <text x="230" y="275" text-anchor="middle" class="rag-label" style="animation: fadeSlideUp 0.5s ease 0.5s forwards; opacity: 0;">Original query passed through</text>

        <!-- Step 5: LLM -->
        <g class="rag-box" style="animation-delay: 0.8s;">
          <rect x="400" y="250" width="220" height="70" rx="12" fill="#ec489915" stroke="#ec4899" stroke-width="2"/>
          <text x="510" y="278" text-anchor="middle" class="rag-title" fill="#ec4899">Large Language Model</text>
          <text x="510" y="298" text-anchor="middle" class="rag-label">Query + Context \u2192 Grounded Response</text>
        </g>

        <!-- Arrow 5 -->
        <line x1="620" y1="285" x2="660" y2="285" stroke="var(--accent-primary)" stroke-width="1.5" class="rag-arrow" marker-end="url(#rag-arrow)"
          style="animation-delay: 1s;"/>

        <!-- Step 6: Response -->
        <g class="rag-box" style="animation-delay: 1s;">
          <rect x="665" y="260" width="70" height="50" rx="10" fill="#00d4aa15" stroke="#00d4aa" stroke-width="2"/>
          <text x="700" y="285" text-anchor="middle" font-size="22">\u{1F4AC}</text>
          <text x="700" y="300" text-anchor="middle" class="rag-label" fill="#00d4aa">Answer</text>
        </g>

        <!-- Title -->
        <text x="375" y="35" text-anchor="middle" fill="var(--text-secondary)" font-family="var(--font-heading)" font-size="14" font-weight="600">
          Retrieval-Augmented Generation Pipeline
        </text>
        <text x="375" y="55" text-anchor="middle" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="10">
          Ground LLM responses in retrieved factual data
        </text>
      </svg>
      <p style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-dim); margin-top: var(--space-3);">
        RAG Pipeline \u2014 Query \u2192 Embed \u2192 Retrieve \u2192 Generate
      </p>
    </div>
  `;
}
