export function render(container) {
  // Generate deterministic cluster points
  const clusters = [
    { cx: 200, cy: 160, color: '#a855f7', label: 'Science', count: 12 },
    { cx: 320, cy: 100, color: '#f59e0b', label: 'Finance', count: 10 },
    { cx: 260, cy: 260, color: '#ec4899', label: 'Health', count: 11 },
    { cx: 150, cy: 280, color: '#22c55e', label: 'Tech', count: 9 },
  ];

  // Seed-based pseudo-random for deterministic dots
  const dots = [];
  let seed = 42;
  function seededRandom() {
    seed = (seed * 16807 + 0) % 2147483647;
    return (seed - 1) / 2147483646;
  }

  clusters.forEach((cluster, ci) => {
    for (let i = 0; i < cluster.count; i++) {
      const angle = seededRandom() * Math.PI * 2;
      const r = 20 + seededRandom() * 45;
      const x = Math.round(cluster.cx + Math.cos(angle) * r);
      const y = Math.round(cluster.cy + Math.sin(angle) * r);
      const delay = (0.1 + seededRandom() * 0.8).toFixed(2);
      dots.push({ x, y, color: cluster.color, delay, cluster: ci });
    }
  });

  // Query point
  const qx = 280, qy = 200;

  // Find 3 nearest neighbors
  const withDist = dots.map((d, i) => ({
    ...d,
    idx: i,
    dist: Math.sqrt((d.x - qx) ** 2 + (d.y - qy) ** 2)
  }));
  withDist.sort((a, b) => a.dist - b.dist);
  const nearest = withDist.slice(0, 3);
  const nearestIndices = new Set(nearest.map(n => n.idx));

  const dotsSvg = dots.map((d, i) => {
    const isNearest = nearestIndices.has(i);
    const matchInfo = isNearest ? nearest.find(n => n.idx === i) : null;
    const baseR = isNearest ? 5 : 3;
    const opacity = isNearest ? 1 : 0.5;
    return `<circle cx="${d.x}" cy="${d.y}" r="${baseR}" fill="${d.color}" opacity="0"
      class="vdb-dot" style="animation-delay: ${d.delay}s;">
      ${isNearest ? `<animate attributeName="r" values="5;7;5" dur="2s" repeatCount="indefinite" begin="2.5s"/>` : ''}
    </circle>`;
  }).join('\n        ');

  // Nearest neighbor lines
  const nnLines = nearest.map((n, i) => {
    const score = (1 - n.dist / 200).toFixed(2);
    return `<line x1="${qx}" y1="${qy}" x2="${n.x}" y2="${n.y}" stroke="#00d4aa" stroke-width="1.5" opacity="0"
      class="vdb-nn-line" style="animation-delay: ${2.8 + i * 0.2}s;" stroke-dasharray="4 2"/>`;
  }).join('\n        ');

  // Results panel data
  const results = nearest.map((n, i) => {
    const score = Math.max(0.75, (1 - n.dist / 200)).toFixed(2);
    const clusterInfo = clusters[n.cluster];
    return { score, label: clusterInfo.label, color: clusterInfo.color, delay: (3.5 + i * 0.25).toFixed(2) };
  });

  const resultsSvg = results.map((r, i) => `
    <g class="vdb-result" style="animation-delay: ${r.delay}s;">
      <rect x="510" y="${60 + i * 55}" width="210" height="45" rx="8" fill="${r.color}10" stroke="${r.color}" stroke-width="1"/>
      <text x="525" y="${78 + i * 55}" font-family="var(--font-heading)" font-size="11" fill="${r.color}" font-weight="600">#${i + 1} ${r.label} doc</text>
      <rect x="525" y="${84 + i * 55}" width="${r.score * 140}" height="6" rx="3" fill="${r.color}" opacity="0.7"/>
      <rect x="525" y="${84 + i * 55}" width="140" height="6" rx="3" fill="${r.color}" opacity="0.15"/>
      <text x="700" y="${93 + i * 55}" text-anchor="end" font-family="var(--font-mono)" font-size="9" fill="var(--text-dim)">${r.score}</text>
    </g>
  `).join('');

  container.innerHTML = `
    <div style="text-align: center; width: 100%;">
      <svg viewBox="0 0 750 380" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
        <style>
          .vdb-dot { animation: vdbDotIn 0.4s ease forwards; }
          .vdb-fade { animation: vdbFadeIn 0.6s ease forwards; opacity: 0; }
          .vdb-query { animation: vdbQueryIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) forwards; opacity: 0; transform-origin: ${qx}px ${qy}px; transform: scale(0); }
          .vdb-ring { animation: vdbRingExpand 1s ease-out forwards; opacity: 0; transform-origin: ${qx}px ${qy}px; transform: scale(0); }
          .vdb-nn-line { animation: vdbFadeIn 0.4s ease forwards; }
          .vdb-result { animation: vdbSlideIn 0.5s ease forwards; opacity: 0; transform: translateX(20px); }
          .vdb-label { font-family: var(--font-mono); font-size: 10px; fill: var(--text-dim); }
          .vdb-title { font-family: var(--font-heading); font-size: 13px; font-weight: 600; }
          @keyframes vdbDotIn { to { opacity: 1; } }
          @keyframes vdbFadeIn { to { opacity: 1; } }
          @keyframes vdbQueryIn { to { opacity: 1; transform: scale(1); } }
          @keyframes vdbRingExpand { to { opacity: 0.3; transform: scale(1); } }
          @keyframes vdbSlideIn { to { opacity: 1; transform: translateX(0); } }
          @keyframes vdbPulse { 0%, 100% { r: 6; opacity: 0.9; } 50% { r: 9; opacity: 1; } }
        </style>

        <defs>
          <radialGradient id="vdb-space-grad" cx="50%" cy="50%">
            <stop offset="0%" stop-color="#1e2740" stop-opacity="0.8"/>
            <stop offset="100%" stop-color="#111827" stop-opacity="1"/>
          </radialGradient>
        </defs>

        <!-- Vector space background -->
        <g class="vdb-fade" style="animation-delay: 0s;">
          <ellipse cx="250" cy="200" rx="230" ry="180" fill="url(#vdb-space-grad)" stroke="var(--border-subtle)" stroke-width="1.5"/>
          <text x="250" y="370" text-anchor="middle" font-family="var(--font-heading)" font-size="11" fill="var(--text-dim)">Vector Space</text>
        </g>

        <!-- Axis hint lines -->
        <g class="vdb-fade" style="animation-delay: 0.1s;" opacity="0.12">
          <line x1="30" y1="200" x2="470" y2="200" stroke="var(--text-dim)" stroke-width="0.5"/>
          <line x1="250" y1="25" x2="250" y2="375" stroke="var(--text-dim)" stroke-width="0.5"/>
        </g>

        <!-- Cluster labels -->
        <g class="vdb-fade" style="animation-delay: 0.8s;">
          ${clusters.map(c => `<text x="${c.cx}" y="${c.cy - 55}" text-anchor="middle" font-family="var(--font-mono)" font-size="9" fill="${c.color}" opacity="0.7">${c.label}</text>`).join('\n          ')}
        </g>

        <!-- Scatter dots -->
        ${dotsSvg}

        <!-- Distance rings from query -->
        <circle cx="${qx}" cy="${qy}" r="50" fill="none" stroke="#3b82f6" stroke-width="0.8" stroke-dasharray="3 3"
          class="vdb-ring" style="animation-delay: 2.2s;"/>
        <circle cx="${qx}" cy="${qy}" r="100" fill="none" stroke="#3b82f6" stroke-width="0.6" stroke-dasharray="3 3"
          class="vdb-ring" style="animation-delay: 2.4s;"/>
        <circle cx="${qx}" cy="${qy}" r="150" fill="none" stroke="#3b82f6" stroke-width="0.4" stroke-dasharray="3 3"
          class="vdb-ring" style="animation-delay: 2.6s;"/>

        <!-- Nearest neighbor connecting lines -->
        ${nnLines}

        <!-- Query vector (pulsing diamond) -->
        <g class="vdb-query" style="animation-delay: 1.8s;">
          <polygon points="${qx},${qy - 10} ${qx + 8},${qy} ${qx},${qy + 10} ${qx - 8},${qy}" fill="#3b82f6" stroke="#fff" stroke-width="1">
            <animate attributeName="opacity" values="0.8;1;0.8" dur="1.5s" repeatCount="indefinite"/>
          </polygon>
          <!-- Pulse ring -->
          <circle cx="${qx}" cy="${qy}" r="6" fill="none" stroke="#3b82f6" stroke-width="1.5" opacity="0">
            <animate attributeName="r" values="6;18;6" dur="2s" repeatCount="indefinite"/>
            <animate attributeName="opacity" values="0.6;0;0.6" dur="2s" repeatCount="indefinite"/>
          </circle>
          <text x="${qx}" y="${qy - 16}" text-anchor="middle" font-family="var(--font-heading)" font-size="10" fill="#3b82f6" font-weight="600">Query</text>
        </g>

        <!-- Results panel -->
        <g class="vdb-fade" style="animation-delay: 3.2s;">
          <rect x="500" y="25" width="230" height="230" rx="12" fill="#11182795" stroke="var(--border-medium)" stroke-width="1.5"/>
          <text x="615" y="50" text-anchor="middle" class="vdb-title" fill="var(--text-primary)">Top 3 Results</text>
        </g>

        ${resultsSvg}

        <!-- K-NN label -->
        <g class="vdb-fade" style="animation-delay: 3.8s;">
          <rect x="500" y="275" width="230" height="70" rx="10" fill="#00d4aa10" stroke="#00d4aa" stroke-width="1" stroke-dasharray="4"/>
          <text x="615" y="298" text-anchor="middle" font-family="var(--font-heading)" font-size="11" fill="#00d4aa" font-weight="600">Approximate Nearest Neighbors</text>
          <text x="615" y="315" text-anchor="middle" font-family="var(--font-mono)" font-size="9" fill="var(--text-dim)">HNSW / IVF index for O(log n) search</text>
          <text x="615" y="332" text-anchor="middle" font-family="var(--font-mono)" font-size="9" fill="var(--text-dim)">Cosine similarity / L2 distance</text>
        </g>

      </svg>
      <p style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-dim); margin-top: var(--space-3);">
        Vector Database \u2014 Semantic Similarity Search in Embedding Space
      </p>
    </div>
  `;
}
