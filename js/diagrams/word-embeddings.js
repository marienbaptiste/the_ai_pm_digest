export function render(container) {
  // Word positions in 2D embedding space (within a ~600x300 coordinate area)
  const words = {
    animals: {
      color: '#3b82f6',
      label: 'Animals',
      items: [
        { word: 'cat', x: 120, y: 80 },
        { word: 'dog', x: 155, y: 110 },
        { word: 'fish', x: 100, y: 130 }
      ]
    },
    food: {
      color: '#f59e0b',
      label: 'Food',
      items: [
        { word: 'pizza', x: 480, y: 90 },
        { word: 'burger', x: 520, y: 120 },
        { word: 'sushi', x: 460, y: 140 }
      ]
    },
    tech: {
      color: '#00d4aa',
      label: 'Tech',
      items: [
        { word: 'code', x: 310, y: 230 },
        { word: 'data', x: 350, y: 260 },
        { word: 'GPU', x: 280, y: 260 }
      ]
    }
  };

  // Analogy points: king - man + woman = queen
  const analogy = {
    king:   { x: 530, y: 220 },
    man:    { x: 580, y: 270 },
    woman:  { x: 460, y: 270 },
    queen:  { x: 410, y: 220 }
  };

  // Build grid lines
  let gridLines = '';
  for (let x = 50; x <= 650; x += 50) {
    gridLines += `<line x1="${x}" y1="40" x2="${x}" y2="310" stroke="var(--border-subtle)" stroke-width="0.5" opacity="0.3"/>`;
  }
  for (let y = 40; y <= 310; y += 45) {
    gridLines += `<line x1="50" y1="${y}" x2="650" y2="${y}" stroke="var(--border-subtle)" stroke-width="0.5" opacity="0.3"/>`;
  }

  // Build word dots and labels
  let wordDots = '';
  let similarityLines = '';
  let dotIndex = 0;

  Object.values(words).forEach((cluster) => {
    const items = cluster.items;
    const color = cluster.color;

    // Similarity lines within cluster
    for (let i = 0; i < items.length; i++) {
      for (let j = i + 1; j < items.length; j++) {
        const delay = (2.5 + dotIndex * 0.15).toFixed(2);
        similarityLines += `
          <line x1="${items[i].x}" y1="${items[i].y}" x2="${items[j].x}" y2="${items[j].y}"
            stroke="${color}" stroke-width="1" stroke-dasharray="4 3"
            class="we-sim-line" style="animation-delay: ${delay}s;"/>
        `;
      }
    }

    // Word dots
    items.forEach((item, idx) => {
      const delay = (0.8 + dotIndex * 0.15).toFixed(2);
      wordDots += `
        <g class="we-dot" style="animation-delay: ${delay}s;">
          <circle cx="${item.x}" cy="${item.y}" r="18" fill="${color}15" stroke="none"/>
          <circle cx="${item.x}" cy="${item.y}" r="6" fill="${color}" stroke="${color}40" stroke-width="3"/>
          <text x="${item.x}" y="${item.y - 14}" text-anchor="middle"
            fill="${color}" font-family="var(--font-mono)" font-size="11" font-weight="600">
            ${item.word}
          </text>
        </g>
      `;
      dotIndex++;
    });
  });

  // Cluster labels
  let clusterLabels = '';
  const clusterPositions = [
    { label: 'Animals', x: 125, y: 160, color: '#3b82f6' },
    { label: 'Food', x: 490, y: 168, color: '#f59e0b' },
    { label: 'Tech', x: 315, y: 290, color: '#00d4aa' }
  ];
  clusterPositions.forEach((cl, i) => {
    const delay = (2.8 + i * 0.2).toFixed(2);
    clusterLabels += `
      <text x="${cl.x}" y="${cl.y}" text-anchor="middle" class="we-cluster-label" style="animation-delay: ${delay}s;"
        fill="${cl.color}" font-family="var(--font-heading)" font-size="10" font-weight="600" opacity="0.6">
        \u25CB ${cl.label}
      </text>
    `;
  });

  // Analogy visualization: king - man + woman = queen
  const a = analogy;

  container.innerHTML = `
    <div style="text-align: center; width: 100%;">
      <svg viewBox="0 0 700 380" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
        <style>
          .we-dot {
            animation: weDotIn 0.5s ease forwards;
            opacity: 0;
          }
          .we-sim-line {
            stroke-dasharray: 100;
            stroke-dashoffset: 100;
            animation: weDrawLine 0.8s ease forwards;
            opacity: 0;
          }
          .we-cluster-label {
            animation: weFadeIn 0.5s ease forwards;
            opacity: 0;
          }
          .we-analogy {
            animation: weFadeIn 0.7s ease forwards;
            opacity: 0;
          }
          .we-analogy-arrow {
            stroke-dasharray: 200;
            stroke-dashoffset: 200;
            animation: weDrawLine 1s ease forwards;
          }
          .we-title {
            animation: weFadeIn 0.5s ease forwards;
            opacity: 0;
          }
          .we-grid {
            animation: weFadeIn 1s ease forwards;
            opacity: 0;
          }
          .we-axis-label {
            animation: weFadeIn 0.5s ease forwards 0.3s;
            opacity: 0;
          }
          @keyframes weDotIn {
            0% { opacity: 0; transform: scale(0); }
            60% { transform: scale(1.2); }
            100% { opacity: 1; transform: scale(1); }
          }
          @keyframes weDrawLine {
            to { stroke-dashoffset: 0; opacity: 0.6; }
          }
          @keyframes weFadeIn {
            to { opacity: 1; }
          }
          @keyframes weAnalogyPulse {
            0%, 100% { opacity: 0.7; }
            50% { opacity: 1; }
          }
          @keyframes weStarPulse {
            0%, 100% { r: 6; opacity: 0.8; }
            50% { r: 9; opacity: 1; }
          }
        </style>

        <defs>
          <marker id="weArrowPurple" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#a855f7"/>
          </marker>
          <marker id="weArrowRed" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#ef4444"/>
          </marker>
          <marker id="weArrowGreen" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#22c55e"/>
          </marker>
          <filter id="weGlow">
            <feGaussianBlur in="SourceGraphic" stdDeviation="3"/>
          </filter>
        </defs>

        <!-- Title -->
        <text x="350" y="25" text-anchor="middle" class="we-title" style="animation-delay: 0s;"
          fill="var(--text-primary)" font-family="var(--font-heading)" font-size="15" font-weight="700">
          Word Embedding Space (2D Projection)
        </text>

        <!-- Grid background -->
        <g class="we-grid" style="animation-delay: 0.1s;">
          ${gridLines}
        </g>

        <!-- Axes -->
        <g class="we-grid" style="animation-delay: 0.2s;">
          <line x1="50" y1="310" x2="650" y2="310" stroke="var(--border-medium)" stroke-width="1.5"/>
          <line x1="50" y1="310" x2="50" y2="40" stroke="var(--border-medium)" stroke-width="1.5"/>
          <!-- Axis arrows -->
          <polygon points="650,310 640,305 640,315" fill="var(--border-medium)"/>
          <polygon points="50,40 45,50 55,50" fill="var(--border-medium)"/>
        </g>

        <text x="360" y="335" text-anchor="middle" class="we-axis-label"
          fill="var(--text-dim)" font-family="var(--font-mono)" font-size="10">
          Dimension 1
        </text>
        <text x="30" y="175" text-anchor="middle" class="we-axis-label"
          fill="var(--text-dim)" font-family="var(--font-mono)" font-size="10"
          transform="rotate(-90, 30, 175)">
          Dimension 2
        </text>

        <!-- Similarity lines (drawn before dots so dots are on top) -->
        ${similarityLines}

        <!-- Word dots -->
        ${wordDots}

        <!-- Cluster labels -->
        ${clusterLabels}

        <!-- ===== Analogy: king - man + woman = queen ===== -->

        <!-- Analogy word dots -->
        <g class="we-analogy" style="animation-delay: 3.5s;">
          <!-- king -->
          <circle cx="${a.king.x}" cy="${a.king.y}" r="6" fill="#a855f7" stroke="#a855f740" stroke-width="3"/>
          <text x="${a.king.x}" y="${a.king.y - 12}" text-anchor="middle"
            fill="#a855f7" font-family="var(--font-mono)" font-size="11" font-weight="600">king</text>
        </g>

        <g class="we-analogy" style="animation-delay: 3.8s;">
          <!-- man -->
          <circle cx="${a.man.x}" cy="${a.man.y}" r="6" fill="#ef4444" stroke="#ef444440" stroke-width="3"/>
          <text x="${a.man.x}" y="${a.man.y + 18}" text-anchor="middle"
            fill="#ef4444" font-family="var(--font-mono)" font-size="11" font-weight="600">man</text>
        </g>

        <g class="we-analogy" style="animation-delay: 4.1s;">
          <!-- woman -->
          <circle cx="${a.woman.x}" cy="${a.woman.y}" r="6" fill="#22c55e" stroke="#22c55e40" stroke-width="3"/>
          <text x="${a.woman.x}" y="${a.woman.y + 18}" text-anchor="middle"
            fill="#22c55e" font-family="var(--font-mono)" font-size="11" font-weight="600">woman</text>
        </g>

        <g class="we-analogy" style="animation-delay: 4.8s;">
          <!-- queen (result) with glow -->
          <circle cx="${a.queen.x}" cy="${a.queen.y}" r="8" fill="#a855f7" stroke="#a855f7" stroke-width="2">
            <animate attributeName="r" values="6;9;6" dur="2s" repeatCount="indefinite" begin="5.5s"/>
          </circle>
          <circle cx="${a.queen.x}" cy="${a.queen.y}" r="14" fill="#a855f7" opacity="0" filter="url(#weGlow)">
            <animate attributeName="opacity" values="0;0.4;0" dur="2s" repeatCount="indefinite" begin="5.5s"/>
          </circle>
          <text x="${a.queen.x}" y="${a.queen.y - 14}" text-anchor="middle"
            fill="#a855f7" font-family="var(--font-mono)" font-size="12" font-weight="700">queen</text>
        </g>

        <!-- Analogy arrows -->
        <!-- king -> man (subtract) -->
        <line x1="${a.king.x}" y1="${a.king.y}" x2="${a.man.x}" y2="${a.man.y}"
          stroke="#ef4444" stroke-width="1.5" class="we-analogy-arrow" style="animation-delay: 4s;"
          marker-end="url(#weArrowRed)"/>
        <text x="${(a.king.x + a.man.x) / 2 + 12}" y="${(a.king.y + a.man.y) / 2}"
          text-anchor="start" class="we-analogy" style="animation-delay: 4.2s;"
          fill="#ef4444" font-family="var(--font-mono)" font-size="10" font-weight="600">- man</text>

        <!-- man -> woman (the offset) -->
        <line x1="${a.man.x}" y1="${a.man.y}" x2="${a.woman.x}" y2="${a.woman.y}"
          stroke="#22c55e" stroke-width="1.5" stroke-dasharray="5 3" class="we-analogy-arrow" style="animation-delay: 4.3s;"
          marker-end="url(#weArrowGreen)"/>
        <text x="${(a.man.x + a.woman.x) / 2}" y="${a.man.y + 30}"
          text-anchor="middle" class="we-analogy" style="animation-delay: 4.5s;"
          fill="#22c55e" font-family="var(--font-mono)" font-size="10" font-weight="600">+ woman</text>

        <!-- king -> queen (the result vector) -->
        <line x1="${a.king.x}" y1="${a.king.y}" x2="${a.queen.x + 8}" y2="${a.queen.y}"
          stroke="#a855f7" stroke-width="2" class="we-analogy-arrow" style="animation-delay: 4.6s;"
          marker-end="url(#weArrowPurple)"/>
        <text x="${(a.king.x + a.queen.x) / 2}" y="${a.king.y - 10}"
          text-anchor="middle" class="we-analogy" style="animation-delay: 4.8s;"
          fill="#a855f7" font-family="var(--font-mono)" font-size="10" font-weight="700">= queen</text>

        <!-- Analogy equation box -->
        <g class="we-analogy" style="animation-delay: 5.2s;">
          <rect x="200" y="345" width="300" height="26" rx="6"
            fill="var(--bg-elevated)" stroke="#a855f7" stroke-width="1" opacity="0.9"/>
          <text x="350" y="363" text-anchor="middle"
            fill="#a855f7" font-family="var(--font-mono)" font-size="11" font-weight="600">
            king \u2212 man + woman \u2248 queen
          </text>
        </g>

        <!-- Legend -->
        <g class="we-analogy" style="animation-delay: 3.2s;">
          <circle cx="75" cy="355" r="4" fill="#3b82f6"/>
          <text x="85" y="359" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="9">Animals</text>
          <circle cx="140" cy="355" r="4" fill="#f59e0b"/>
          <text x="150" y="359" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="9">Food</text>
          <circle cx="195" cy="355" r="4" fill="#00d4aa"/>
          <text x="205" y="359" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="9">Tech</text>
        </g>
      </svg>
      <p style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-dim); margin-top: var(--space-3);">
        Word Embeddings \u2014 Semantic Similarity and Vector Arithmetic
      </p>
    </div>
  `;
}
