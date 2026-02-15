// Embedding Projector — Interactive 2D embedding space explorer
// Users click two words to measure cosine similarity between position vectors

export function mount(container) {
  // ── Word data with semantic clusters ──────────────────────────────
  const clusters = {
    animals: {
      label: 'Animals',
      color: 'var(--accent-blue)',
      colorHex: '#3b82f6',
      items: [
        { word: 'cat', x: 120, y: 100 },
        { word: 'dog', x: 140, y: 120 },
        { word: 'fish', x: 100, y: 140 },
        { word: 'bird', x: 160, y: 90 },
      ],
    },
    food: {
      label: 'Food',
      color: 'var(--accent-warm)',
      colorHex: '#f59e0b',
      items: [
        { word: 'pizza', x: 400, y: 250 },
        { word: 'pasta', x: 420, y: 270 },
        { word: 'burger', x: 380, y: 230 },
      ],
    },
    tech: {
      label: 'Tech',
      color: 'var(--accent-primary)',
      colorHex: '#00d4aa',
      items: [
        { word: 'computer', x: 300, y: 80 },
        { word: 'GPU', x: 320, y: 60 },
        { word: 'algorithm', x: 280, y: 100 },
        { word: 'neural', x: 340, y: 90 },
      ],
    },
    emotions: {
      label: 'Emotions',
      color: 'var(--accent-purple)',
      colorHex: '#a855f7',
      items: [
        { word: 'happy', x: 150, y: 300 },
        { word: 'sad', x: 180, y: 320 },
        { word: 'angry', x: 200, y: 290 },
      ],
    },
  };

  // Flatten all words into a single array with cluster metadata
  const allWords = [];
  for (const [key, cluster] of Object.entries(clusters)) {
    for (const item of cluster.items) {
      allWords.push({
        ...item,
        cluster: key,
        clusterLabel: cluster.label,
        color: cluster.color,
        colorHex: cluster.colorHex,
      });
    }
  }

  // ── State ─────────────────────────────────────────────────────────
  let selected = []; // indices into allWords (max 2)

  // ── Maths helpers ─────────────────────────────────────────────────
  function euclidean(a, b) {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
  }

  function cosineSimilarity(a, b) {
    const dot = a.x * b.x + a.y * b.y;
    const magA = Math.sqrt(a.x ** 2 + a.y ** 2);
    const magB = Math.sqrt(b.x ** 2 + b.y ** 2);
    if (magA === 0 || magB === 0) return 0;
    return dot / (magA * magB);
  }

  function similarityColor(cos) {
    if (cos > 0.9) return '#00d4aa';   // green — similar
    if (cos >= 0.7) return '#f59e0b';  // amber — moderate
    return '#ef4444';                   // red — dissimilar
  }

  function similarityLabel(cos) {
    if (cos > 0.9) return 'High similarity';
    if (cos >= 0.7) return 'Moderate similarity';
    return 'Low similarity';
  }

  // ── Cluster boundary helpers ──────────────────────────────────────
  function clusterCenter(items) {
    const cx = items.reduce((s, i) => s + i.x, 0) / items.length;
    const cy = items.reduce((s, i) => s + i.y, 0) / items.length;
    return { cx, cy };
  }

  function clusterRadius(items, center) {
    let maxDist = 0;
    for (const item of items) {
      const d = Math.sqrt((item.x - center.cx) ** 2 + (item.y - center.cy) ** 2);
      if (d > maxDist) maxDist = d;
    }
    return maxDist + 30; // padding
  }

  // ── Inject styles ─────────────────────────────────────────────────
  const styleId = 'ep-styles';
  if (!document.getElementById(styleId)) {
    const style = document.createElement('style');
    style.id = styleId;
    style.textContent = `
      .ep-widget {
        background: var(--bg-elevated, #1e2740);
        border: 1px solid var(--border-subtle, #2d3a52);
        border-radius: var(--radius-md, 12px);
        overflow: hidden;
        font-family: var(--font-body, system-ui, sans-serif);
      }
      .ep-header {
        padding: var(--space-4, 16px) var(--space-6, 24px);
        border-bottom: 1px solid var(--border-subtle, #2d3a52);
      }
      .ep-header h3 {
        margin: 0 0 var(--space-2, 8px) 0;
        font-family: var(--font-heading, system-ui, sans-serif);
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary, #f0f4f8);
      }
      .ep-header p {
        margin: 0;
        font-size: 0.85rem;
        color: var(--text-secondary, #94a3b8);
      }
      .ep-canvas-wrap {
        padding: var(--space-4, 16px);
        display: flex;
        justify-content: center;
        background: var(--bg-surface, #111827);
      }
      .ep-canvas-wrap svg {
        max-width: 100%;
        height: auto;
        display: block;
      }
      .ep-word-group {
        cursor: pointer;
        transition: transform 0.15s ease;
      }
      .ep-word-group:hover .ep-dot-halo {
        opacity: 0.35;
      }
      .ep-word-group.ep-selected .ep-dot-halo {
        opacity: 0.5;
      }
      .ep-word-group.ep-selected .ep-dot-core {
        r: 8;
      }
      .ep-dot-halo {
        opacity: 0.15;
        transition: opacity 0.2s ease;
      }
      .ep-dot-core {
        transition: r 0.15s ease;
      }
      .ep-cluster-boundary {
        fill: none;
        stroke-dasharray: 6 4;
        opacity: 0.2;
        transition: opacity 0.3s ease;
      }
      .ep-connection-line {
        stroke-width: 2;
        stroke-dasharray: 6 4;
        animation: ep-draw-line 0.4s ease forwards;
      }
      .ep-panel {
        padding: var(--space-4, 16px) var(--space-6, 24px);
        border-top: 1px solid var(--border-subtle, #2d3a52);
        min-height: 72px;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .ep-panel-empty {
        color: var(--text-dim, #64748b);
        font-size: 0.85rem;
        text-align: center;
        font-family: var(--font-mono, monospace);
      }
      .ep-panel-result {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-4, 16px);
        align-items: center;
        justify-content: center;
        width: 100%;
      }
      .ep-metric {
        text-align: center;
      }
      .ep-metric-label {
        font-family: var(--font-mono, monospace);
        font-size: 0.7rem;
        color: var(--text-dim, #64748b);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 2px;
      }
      .ep-metric-value {
        font-family: var(--font-mono, monospace);
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-primary, #f0f4f8);
      }
      .ep-metric-badge {
        display: inline-block;
        font-family: var(--font-mono, monospace);
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 9999px;
        margin-top: 4px;
      }
      .ep-pair-label {
        font-family: var(--font-mono, monospace);
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-primary, #f0f4f8);
      }
      .ep-pair-sep {
        color: var(--text-dim, #64748b);
        margin: 0 4px;
      }
      .ep-divider {
        width: 1px;
        height: 40px;
        background: var(--border-subtle, #2d3a52);
      }
      .ep-fade-in {
        animation: ep-fade-in 0.5s ease forwards;
        opacity: 0;
      }
      .ep-dot-enter {
        animation: ep-dot-enter 0.45s ease forwards;
        opacity: 0;
      }
      @keyframes ep-fade-in {
        to { opacity: 1; }
      }
      @keyframes ep-dot-enter {
        0%   { opacity: 0; transform: scale(0); }
        70%  { transform: scale(1.15); }
        100% { opacity: 1; transform: scale(1); }
      }
      @keyframes ep-draw-line {
        from { stroke-dashoffset: 800; }
        to   { stroke-dashoffset: 0; }
      }
    `;
    document.head.appendChild(style);
  }

  // ── Build initial HTML ────────────────────────────────────────────
  container.innerHTML = '';

  const widget = document.createElement('div');
  widget.className = 'ep-widget';

  // Header
  const header = document.createElement('div');
  header.className = 'ep-header';
  header.innerHTML = `
    <h3>Embedding Space Explorer</h3>
    <p>Click two words to measure their similarity</p>
  `;
  widget.appendChild(header);

  // SVG canvas wrapper
  const canvasWrap = document.createElement('div');
  canvasWrap.className = 'ep-canvas-wrap';
  widget.appendChild(canvasWrap);

  // Similarity panel
  const panel = document.createElement('div');
  panel.className = 'ep-panel';
  widget.appendChild(panel);

  container.appendChild(widget);

  // ── Create SVG ────────────────────────────────────────────────────
  const svgNS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(svgNS, 'svg');
  svg.setAttribute('viewBox', '0 0 600 400');
  svg.setAttribute('xmlns', svgNS);
  svg.setAttribute('width', '600');
  svg.setAttribute('height', '400');
  canvasWrap.appendChild(svg);

  // ── Grid lines ────────────────────────────────────────────────────
  const gridGroup = document.createElementNS(svgNS, 'g');
  gridGroup.classList.add('ep-fade-in');
  gridGroup.style.animationDelay = '0s';

  for (let x = 50; x <= 550; x += 50) {
    const line = document.createElementNS(svgNS, 'line');
    line.setAttribute('x1', x);
    line.setAttribute('y1', 10);
    line.setAttribute('x2', x);
    line.setAttribute('y2', 380);
    line.setAttribute('stroke', 'var(--border-subtle, #2d3a52)');
    line.setAttribute('stroke-width', '0.5');
    line.setAttribute('opacity', '0.25');
    gridGroup.appendChild(line);
  }
  for (let y = 10; y <= 380; y += 50) {
    const line = document.createElementNS(svgNS, 'line');
    line.setAttribute('x1', 50);
    line.setAttribute('y1', y);
    line.setAttribute('x2', 550);
    line.setAttribute('y2', y);
    line.setAttribute('stroke', 'var(--border-subtle, #2d3a52)');
    line.setAttribute('stroke-width', '0.5');
    line.setAttribute('opacity', '0.25');
    gridGroup.appendChild(line);
  }
  svg.appendChild(gridGroup);

  // ── Axes ──────────────────────────────────────────────────────────
  const axesGroup = document.createElementNS(svgNS, 'g');
  axesGroup.classList.add('ep-fade-in');
  axesGroup.style.animationDelay = '0.1s';

  // x axis
  const xAxis = document.createElementNS(svgNS, 'line');
  xAxis.setAttribute('x1', '30');
  xAxis.setAttribute('y1', '390');
  xAxis.setAttribute('x2', '580');
  xAxis.setAttribute('y2', '390');
  xAxis.setAttribute('stroke', 'var(--border-medium, #3d4a62)');
  xAxis.setAttribute('stroke-width', '1');
  axesGroup.appendChild(xAxis);

  // y axis
  const yAxis = document.createElementNS(svgNS, 'line');
  yAxis.setAttribute('x1', '30');
  yAxis.setAttribute('y1', '390');
  yAxis.setAttribute('x2', '30');
  yAxis.setAttribute('y2', '5');
  yAxis.setAttribute('stroke', 'var(--border-medium, #3d4a62)');
  yAxis.setAttribute('stroke-width', '1');
  axesGroup.appendChild(yAxis);

  // Axis label: Dimension 1
  const dim1Label = document.createElementNS(svgNS, 'text');
  dim1Label.setAttribute('x', '310');
  dim1Label.setAttribute('y', '398');
  dim1Label.setAttribute('text-anchor', 'middle');
  dim1Label.setAttribute('fill', 'var(--text-dim, #64748b)');
  dim1Label.setAttribute('font-family', 'var(--font-mono, monospace)');
  dim1Label.setAttribute('font-size', '10');
  dim1Label.setAttribute('opacity', '0.6');
  dim1Label.textContent = 'Dimension 1';
  axesGroup.appendChild(dim1Label);

  // Axis label: Dimension 2
  const dim2Label = document.createElementNS(svgNS, 'text');
  dim2Label.setAttribute('x', '18');
  dim2Label.setAttribute('y', '200');
  dim2Label.setAttribute('text-anchor', 'middle');
  dim2Label.setAttribute('fill', 'var(--text-dim, #64748b)');
  dim2Label.setAttribute('font-family', 'var(--font-mono, monospace)');
  dim2Label.setAttribute('font-size', '10');
  dim2Label.setAttribute('opacity', '0.6');
  dim2Label.setAttribute('transform', 'rotate(-90, 18, 200)');
  dim2Label.textContent = 'Dimension 2';
  axesGroup.appendChild(dim2Label);

  svg.appendChild(axesGroup);

  // ── Cluster boundaries (dashed circles) ───────────────────────────
  const boundaryGroup = document.createElementNS(svgNS, 'g');
  boundaryGroup.classList.add('ep-fade-in');
  boundaryGroup.style.animationDelay = '0.2s';

  for (const [, cluster] of Object.entries(clusters)) {
    const center = clusterCenter(cluster.items);
    const radius = clusterRadius(cluster.items, center);

    const circle = document.createElementNS(svgNS, 'circle');
    circle.setAttribute('cx', center.cx);
    circle.setAttribute('cy', center.cy);
    circle.setAttribute('r', radius);
    circle.setAttribute('stroke', cluster.colorHex);
    circle.setAttribute('stroke-width', '1');
    circle.classList.add('ep-cluster-boundary');
    boundaryGroup.appendChild(circle);

    // Cluster label
    const label = document.createElementNS(svgNS, 'text');
    label.setAttribute('x', center.cx);
    label.setAttribute('y', center.cy - radius - 6);
    label.setAttribute('text-anchor', 'middle');
    label.setAttribute('fill', cluster.colorHex);
    label.setAttribute('font-family', 'var(--font-heading, system-ui)');
    label.setAttribute('font-size', '10');
    label.setAttribute('font-weight', '600');
    label.setAttribute('opacity', '0.5');
    label.textContent = cluster.label;
    boundaryGroup.appendChild(label);
  }

  svg.appendChild(boundaryGroup);

  // ── Connection line layer (rendered between boundaries and dots) ──
  const connectionGroup = document.createElementNS(svgNS, 'g');
  svg.appendChild(connectionGroup);

  // ── Word dots ─────────────────────────────────────────────────────
  const dotGroups = [];

  allWords.forEach((w, idx) => {
    const g = document.createElementNS(svgNS, 'g');
    g.classList.add('ep-word-group', 'ep-dot-enter');
    g.style.animationDelay = `${0.3 + idx * 0.06}s`;
    g.dataset.index = idx;

    // Halo
    const halo = document.createElementNS(svgNS, 'circle');
    halo.setAttribute('cx', w.x);
    halo.setAttribute('cy', w.y);
    halo.setAttribute('r', '20');
    halo.setAttribute('fill', w.colorHex);
    halo.classList.add('ep-dot-halo');
    g.appendChild(halo);

    // Core dot
    const core = document.createElementNS(svgNS, 'circle');
    core.setAttribute('cx', w.x);
    core.setAttribute('cy', w.y);
    core.setAttribute('r', '6');
    core.setAttribute('fill', w.colorHex);
    core.setAttribute('stroke', w.colorHex + '60');
    core.setAttribute('stroke-width', '3');
    core.classList.add('ep-dot-core');
    g.appendChild(core);

    // Label
    const text = document.createElementNS(svgNS, 'text');
    text.setAttribute('x', w.x);
    text.setAttribute('y', w.y - 14);
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('fill', w.colorHex);
    text.setAttribute('font-family', 'var(--font-mono, monospace)');
    text.setAttribute('font-size', '11');
    text.setAttribute('font-weight', '600');
    text.textContent = w.word;
    g.appendChild(text);

    // Click handler
    g.addEventListener('click', () => handleWordClick(idx));

    svg.appendChild(g);
    dotGroups.push(g);
  });

  // ── Interaction logic ─────────────────────────────────────────────
  function handleWordClick(idx) {
    // If already selected, deselect it
    const existingIdx = selected.indexOf(idx);
    if (existingIdx !== -1) {
      selected.splice(existingIdx, 1);
      updateVisuals();
      return;
    }

    // If we already have 2, reset
    if (selected.length >= 2) {
      selected = [];
    }

    selected.push(idx);
    updateVisuals();
  }

  function updateVisuals() {
    // Update selected classes on all dot groups
    dotGroups.forEach((g, i) => {
      g.classList.toggle('ep-selected', selected.includes(i));
    });

    // Clear connection lines
    while (connectionGroup.firstChild) {
      connectionGroup.removeChild(connectionGroup.firstChild);
    }

    // If two words selected, draw connection and show metrics
    if (selected.length === 2) {
      const a = allWords[selected[0]];
      const b = allWords[selected[1]];
      const cos = cosineSimilarity(a, b);
      const dist = euclidean(a, b);
      const lineColor = similarityColor(cos);

      // Draw line
      const line = document.createElementNS(svgNS, 'line');
      line.setAttribute('x1', a.x);
      line.setAttribute('y1', a.y);
      line.setAttribute('x2', b.x);
      line.setAttribute('y2', b.y);
      line.setAttribute('stroke', lineColor);
      line.classList.add('ep-connection-line');
      line.setAttribute('stroke-dasharray', `${dist}`);
      line.setAttribute('stroke-dashoffset', `${dist}`);
      connectionGroup.appendChild(line);
      // Trigger animation
      requestAnimationFrame(() => {
        line.style.strokeDashoffset = '0';
        line.style.transition = 'stroke-dashoffset 0.4s ease';
      });

      // Midpoint label (distance)
      const mx = (a.x + b.x) / 2;
      const my = (a.y + b.y) / 2;
      const distLabel = document.createElementNS(svgNS, 'text');
      distLabel.setAttribute('x', mx);
      distLabel.setAttribute('y', my - 10);
      distLabel.setAttribute('text-anchor', 'middle');
      distLabel.setAttribute('fill', lineColor);
      distLabel.setAttribute('font-family', 'var(--font-mono, monospace)');
      distLabel.setAttribute('font-size', '10');
      distLabel.setAttribute('font-weight', '600');
      distLabel.setAttribute('opacity', '0.85');
      distLabel.textContent = `d = ${dist.toFixed(1)}`;
      connectionGroup.appendChild(distLabel);

      // Show panel
      renderPanel(a, b, cos, dist, lineColor);
    } else {
      renderPanelEmpty();
    }
  }

  // ── Similarity panel rendering ────────────────────────────────────
  function renderPanel(a, b, cos, dist, lineColor) {
    const cosFormatted = cos.toFixed(4);
    const distFormatted = dist.toFixed(1);
    const label = similarityLabel(cos);

    const badgeBg = lineColor + '20';

    panel.innerHTML = `
      <div class="ep-panel-result">
        <div class="ep-metric">
          <div class="ep-metric-label">Pair</div>
          <div class="ep-pair-label">
            <span style="color: ${a.colorHex}">${a.word}</span>
            <span class="ep-pair-sep">\u2194</span>
            <span style="color: ${b.colorHex}">${b.word}</span>
          </div>
        </div>
        <div class="ep-divider"></div>
        <div class="ep-metric">
          <div class="ep-metric-label">Cosine Similarity</div>
          <div class="ep-metric-value" style="color: ${lineColor}">${cosFormatted}</div>
        </div>
        <div class="ep-divider"></div>
        <div class="ep-metric">
          <div class="ep-metric-label">Euclidean Distance</div>
          <div class="ep-metric-value">${distFormatted} <span style="font-size: 0.7rem; color: var(--text-dim, #64748b);">px</span></div>
        </div>
        <div class="ep-divider"></div>
        <div class="ep-metric">
          <div class="ep-metric-label">Verdict</div>
          <div>
            <span class="ep-metric-badge" style="background: ${badgeBg}; color: ${lineColor};">${label}</span>
          </div>
        </div>
      </div>
    `;
  }

  function renderPanelEmpty() {
    const count = selected.length;
    if (count === 0) {
      panel.innerHTML = `<div class="ep-panel-empty">Select two words to compare their embedding vectors</div>`;
    } else if (count === 1) {
      const w = allWords[selected[0]];
      panel.innerHTML = `<div class="ep-panel-empty">
        Selected <span style="color: ${w.colorHex}; font-weight: 600;">${w.word}</span>
        &mdash; now click a second word
      </div>`;
    }
  }

  // Initial empty panel
  renderPanelEmpty();
}
