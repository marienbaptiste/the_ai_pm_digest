// Self-Attention Visualizer — interactive word-level attention weight explorer

export function mount(container) {
  const words = ['The', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'it', 'was', 'tired'];
  const n = words.length;

  // Pre-computed plausible attention weights (row = source/query word, col = target/key word)
  // Each row sums to ~1.0 (softmax-like distribution)
  // Indices: 0=The, 1=cat, 2=sat, 3=on, 4=the, 5=mat, 6=because, 7=it, 8=was, 9=tired
  const attentionMatrix = [
    // The -> attends to cat (determiner->noun), some to mat
    [0.05, 0.45, 0.08, 0.04, 0.06, 0.20, 0.03, 0.03, 0.03, 0.03],
    // cat -> attends to sat (subject->verb), some to The, it
    [0.15, 0.10, 0.35, 0.05, 0.03, 0.04, 0.03, 0.18, 0.03, 0.04],
    // sat -> attends to cat (verb->subject), on (verb->prep), mat
    [0.04, 0.30, 0.06, 0.25, 0.03, 0.18, 0.04, 0.03, 0.04, 0.03],
    // on -> attends to mat (prep->noun), sat (prep->verb)
    [0.03, 0.05, 0.20, 0.05, 0.08, 0.42, 0.04, 0.03, 0.05, 0.05],
    // the -> attends to mat (determiner->noun), some to cat
    [0.06, 0.18, 0.05, 0.04, 0.05, 0.48, 0.03, 0.03, 0.04, 0.04],
    // mat -> attends to the (noun->det), on (noun->prep), sat
    [0.04, 0.08, 0.15, 0.22, 0.28, 0.06, 0.04, 0.03, 0.05, 0.05],
    // because -> attends broadly, links clauses, some to it, was
    [0.04, 0.10, 0.12, 0.04, 0.04, 0.06, 0.06, 0.22, 0.18, 0.14],
    // it -> attends strongly to cat (coreference!), some to was, tired
    [0.04, 0.52, 0.05, 0.02, 0.02, 0.03, 0.04, 0.04, 0.14, 0.10],
    // was -> attends to it (aux->subject), tired (aux->predicate)
    [0.03, 0.08, 0.04, 0.02, 0.02, 0.03, 0.05, 0.25, 0.06, 0.42],
    // tired -> attends to was (pred->aux), it (pred->subject), cat
    [0.03, 0.15, 0.04, 0.02, 0.02, 0.03, 0.06, 0.20, 0.32, 0.13],
  ];

  // State
  let selectedIdx = null;

  // Unique ID prefix for this instance
  const uid = 'attnviz-' + Math.random().toString(36).slice(2, 8);

  // Inject styles
  const styleEl = document.createElement('style');
  styleEl.textContent = `
    .${uid}-widget {
      background: var(--bg-elevated, #1e2740);
      border: 1px solid var(--border-subtle, #2d3748);
      border-radius: var(--radius-md, 10px);
      padding: var(--space-6, 24px);
      font-family: var(--font-body, system-ui, sans-serif);
      max-width: 720px;
      margin: 0 auto;
    }
    .${uid}-header {
      margin-bottom: var(--space-5, 20px);
    }
    .${uid}-title {
      font-family: var(--font-heading, system-ui, sans-serif);
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--text-primary, #f1f5f9);
      margin: 0 0 4px 0;
    }
    .${uid}-subtitle {
      font-family: var(--font-mono, monospace);
      font-size: var(--text-xs, 0.75rem);
      color: var(--text-dim, #64748b);
      margin: 0;
    }
    .${uid}-legend {
      display: inline-flex;
      align-items: center;
      gap: var(--space-2, 8px);
      background: var(--bg-surface, #111827);
      border: 1px solid var(--border-subtle, #2d3748);
      border-radius: var(--radius-sm, 6px);
      padding: 6px 12px;
      margin-bottom: var(--space-4, 16px);
      font-family: var(--font-mono, monospace);
      font-size: var(--text-xs, 0.75rem);
      color: var(--text-secondary, #94a3b8);
    }
    .${uid}-legend-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent-primary, #00d4aa);
      flex-shrink: 0;
    }
    .${uid}-sentence-area {
      position: relative;
      margin-bottom: var(--space-6, 24px);
    }
    .${uid}-words-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      position: relative;
      z-index: 2;
      padding: var(--space-4, 16px);
      background: var(--bg-surface, #111827);
      border: 1px solid var(--border-subtle, #2d3748);
      border-radius: var(--radius-md, 10px);
    }
    .${uid}-word {
      font-family: var(--font-mono, monospace);
      font-size: var(--text-sm, 0.875rem);
      color: var(--text-primary, #f1f5f9);
      padding: 6px 12px;
      border-radius: var(--radius-sm, 6px);
      border: 2px solid transparent;
      cursor: pointer;
      transition: all 0.25s ease;
      position: relative;
      user-select: none;
      background: transparent;
    }
    .${uid}-word:hover {
      border-color: var(--border-medium, #4a5568);
    }
    .${uid}-word--selected {
      border-color: var(--accent-primary, #00d4aa) !important;
      box-shadow: 0 0 12px rgba(0, 212, 170, 0.3);
      color: var(--accent-primary, #00d4aa) !important;
    }
    .${uid}-word-weight {
      position: absolute;
      bottom: -16px;
      left: 50%;
      transform: translateX(-50%);
      font-family: var(--font-mono, monospace);
      font-size: 9px;
      color: var(--accent-primary, #00d4aa);
      opacity: 0;
      transition: opacity 0.25s ease;
      white-space: nowrap;
    }
    .${uid}-word-weight--visible {
      opacity: 1;
    }
    .${uid}-svg-overlay {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 1;
    }
    .${uid}-attn-line {
      transition: all 0.3s ease;
    }
    .${uid}-heatmap-section {
      margin-top: var(--space-4, 16px);
    }
    .${uid}-heatmap-label {
      font-family: var(--font-heading, system-ui, sans-serif);
      font-size: var(--text-sm, 0.875rem);
      font-weight: 600;
      color: var(--text-secondary, #94a3b8);
      margin-bottom: var(--space-3, 12px);
    }
    .${uid}-heatmap-container {
      overflow-x: auto;
      background: var(--bg-surface, #111827);
      border: 1px solid var(--border-subtle, #2d3748);
      border-radius: var(--radius-md, 10px);
      padding: var(--space-3, 12px);
    }
    .${uid}-heatmap-grid {
      display: grid;
      gap: 2px;
      width: fit-content;
      margin: 0 auto;
    }
    .${uid}-heatmap-cell {
      width: 28px;
      height: 28px;
      border-radius: 3px;
      cursor: pointer;
      transition: all 0.2s ease;
      border: 1px solid transparent;
      position: relative;
    }
    .${uid}-heatmap-cell:hover {
      border-color: var(--text-secondary, #94a3b8);
      transform: scale(1.15);
      z-index: 2;
    }
    .${uid}-heatmap-cell--row-selected {
      border-color: var(--accent-primary, #00d4aa) !important;
    }
    .${uid}-heatmap-row-label,
    .${uid}-heatmap-col-label {
      font-family: var(--font-mono, monospace);
      font-size: 9px;
      color: var(--text-dim, #64748b);
      display: flex;
      align-items: center;
      justify-content: flex-end;
      padding-right: 4px;
      user-select: none;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .${uid}-heatmap-col-label {
      justify-content: center;
      padding-right: 0;
      writing-mode: vertical-rl;
      text-orientation: mixed;
      transform: rotate(180deg);
      height: 40px;
    }
    .${uid}-heatmap-corner {
      /* empty top-left corner */
    }
    .${uid}-scale-bar {
      display: flex;
      align-items: center;
      gap: var(--space-2, 8px);
      margin-top: var(--space-3, 12px);
      justify-content: center;
    }
    .${uid}-scale-gradient {
      width: 120px;
      height: 10px;
      border-radius: 5px;
      background: linear-gradient(to right, rgba(0,212,170,0.05), rgba(0,212,170,1));
    }
    .${uid}-scale-label {
      font-family: var(--font-mono, monospace);
      font-size: 9px;
      color: var(--text-dim, #64748b);
    }
    .${uid}-tooltip {
      position: fixed;
      background: var(--bg-elevated, #1e2740);
      border: 1px solid var(--border-medium, #4a5568);
      border-radius: var(--radius-sm, 6px);
      padding: 4px 8px;
      font-family: var(--font-mono, monospace);
      font-size: 10px;
      color: var(--text-primary, #f1f5f9);
      pointer-events: none;
      z-index: 100;
      display: none;
      white-space: nowrap;
    }
  `;
  container.appendChild(styleEl);

  // Build DOM
  const widget = document.createElement('div');
  widget.className = `${uid}-widget`;

  // Header
  const header = document.createElement('div');
  header.className = `${uid}-header`;
  header.innerHTML = `
    <p class="${uid}-title">Self-Attention Visualizer</p>
    <p class="${uid}-subtitle">Click a word to see what it attends to</p>
  `;
  widget.appendChild(header);

  // Legend
  const legend = document.createElement('div');
  legend.className = `${uid}-legend`;
  legend.innerHTML = `<span class="${uid}-legend-dot"></span> Click a word to see its attention weights`;
  widget.appendChild(legend);

  // Sentence area (words + SVG overlay)
  const sentenceArea = document.createElement('div');
  sentenceArea.className = `${uid}-sentence-area`;

  const wordsRow = document.createElement('div');
  wordsRow.className = `${uid}-words-row`;

  const wordEls = [];
  const weightLabels = [];

  words.forEach((word, i) => {
    const span = document.createElement('span');
    span.className = `${uid}-word`;
    span.textContent = word;
    span.dataset.idx = i;

    const weightLabel = document.createElement('span');
    weightLabel.className = `${uid}-word-weight`;
    weightLabel.textContent = '';
    span.appendChild(weightLabel);

    span.addEventListener('click', () => selectWord(i));

    wordsRow.appendChild(span);
    wordEls.push(span);
    weightLabels.push(weightLabel);
  });

  // SVG overlay for connecting lines
  const svgOverlay = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svgOverlay.setAttribute('class', `${uid}-svg-overlay`);

  sentenceArea.appendChild(wordsRow);
  sentenceArea.appendChild(svgOverlay);
  widget.appendChild(sentenceArea);

  // Heatmap section
  const heatmapSection = document.createElement('div');
  heatmapSection.className = `${uid}-heatmap-section`;

  const heatmapLabel = document.createElement('div');
  heatmapLabel.className = `${uid}-heatmap-label`;
  heatmapLabel.textContent = 'Attention Heatmap (Query x Key)';
  heatmapSection.appendChild(heatmapLabel);

  const heatmapContainer = document.createElement('div');
  heatmapContainer.className = `${uid}-heatmap-container`;

  const heatmapGrid = document.createElement('div');
  heatmapGrid.className = `${uid}-heatmap-grid`;
  heatmapGrid.style.gridTemplateColumns = `48px repeat(${n}, 28px)`;
  heatmapGrid.style.gridTemplateRows = `40px repeat(${n}, 28px)`;

  // Corner cell
  const corner = document.createElement('div');
  corner.className = `${uid}-heatmap-corner`;
  heatmapGrid.appendChild(corner);

  // Column headers
  words.forEach((word) => {
    const colLabel = document.createElement('div');
    colLabel.className = `${uid}-heatmap-col-label`;
    colLabel.textContent = word;
    heatmapGrid.appendChild(colLabel);
  });

  // Grid cells (row by row)
  const heatCells = [];
  for (let r = 0; r < n; r++) {
    // Row label
    const rowLabel = document.createElement('div');
    rowLabel.className = `${uid}-heatmap-row-label`;
    rowLabel.textContent = words[r];
    heatmapGrid.appendChild(rowLabel);

    const rowCells = [];
    for (let c = 0; c < n; c++) {
      const cell = document.createElement('div');
      cell.className = `${uid}-heatmap-cell`;
      const weight = attentionMatrix[r][c];
      // Color: accent-primary with opacity proportional to weight
      cell.style.background = `rgba(0, 212, 170, ${(weight * 0.95 + 0.05).toFixed(3)})`;
      cell.dataset.row = r;
      cell.dataset.col = c;
      cell.title = `${words[r]} -> ${words[c]}: ${weight.toFixed(2)}`;

      cell.addEventListener('click', () => selectWord(r));

      cell.addEventListener('mouseenter', (e) => showTooltip(e, `${words[r]} \u2192 ${words[c]}: ${weight.toFixed(2)}`));
      cell.addEventListener('mouseleave', hideTooltip);

      heatmapGrid.appendChild(cell);
      rowCells.push(cell);
    }
    heatCells.push(rowCells);
  }

  heatmapContainer.appendChild(heatmapGrid);
  heatmapSection.appendChild(heatmapContainer);

  // Color scale bar
  const scaleBar = document.createElement('div');
  scaleBar.className = `${uid}-scale-bar`;
  scaleBar.innerHTML = `
    <span class="${uid}-scale-label">Low</span>
    <div class="${uid}-scale-gradient"></div>
    <span class="${uid}-scale-label">High</span>
  `;
  heatmapSection.appendChild(scaleBar);

  widget.appendChild(heatmapSection);

  // Tooltip element
  const tooltip = document.createElement('div');
  tooltip.className = `${uid}-tooltip`;
  widget.appendChild(tooltip);

  container.appendChild(widget);

  // --- Interaction Logic ---

  function showTooltip(e, text) {
    tooltip.textContent = text;
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX + 12) + 'px';
    tooltip.style.top = (e.clientY - 8) + 'px';
  }

  function hideTooltip() {
    tooltip.style.display = 'none';
  }

  function selectWord(idx) {
    if (selectedIdx === idx) {
      // Deselect
      selectedIdx = null;
      clearVisualization();
      return;
    }

    selectedIdx = idx;
    updateVisualization();
  }

  function clearVisualization() {
    // Remove word highlights
    wordEls.forEach((el, i) => {
      el.classList.remove(`${uid}-word--selected`);
      el.style.background = 'transparent';
      el.style.color = '';
      weightLabels[i].textContent = '';
      weightLabels[i].classList.remove(`${uid}-word-weight--visible`);
    });

    // Clear SVG lines
    while (svgOverlay.firstChild) {
      svgOverlay.removeChild(svgOverlay.firstChild);
    }

    // Clear heatmap highlights
    heatCells.forEach(row => {
      row.forEach(cell => {
        cell.classList.remove(`${uid}-heatmap-cell--row-selected`);
      });
    });
  }

  function updateVisualization() {
    clearVisualization();

    const idx = selectedIdx;
    if (idx === null) return;

    const weights = attentionMatrix[idx];

    // Highlight selected word
    wordEls[idx].classList.add(`${uid}-word--selected`);

    // Update other words with attention-based opacity
    wordEls.forEach((el, i) => {
      if (i === idx) return;
      const w = weights[i];
      // Background teal with opacity proportional to weight
      el.style.background = `rgba(0, 212, 170, ${(w * 0.85).toFixed(3)})`;
      // Text brightness scales with weight
      const brightness = Math.round(150 + w * 105);
      el.style.color = `rgb(${brightness}, ${brightness}, ${brightness})`;

      // Show weight label for significant weights
      weightLabels[i].textContent = w.toFixed(2);
      weightLabels[i].classList.add(`${uid}-word-weight--visible`);
    });

    // Highlight heatmap row
    heatCells[idx].forEach(cell => {
      cell.classList.add(`${uid}-heatmap-cell--row-selected`);
    });

    // Draw connecting lines from selected word to high-attention words
    drawLines(idx, weights);
  }

  function drawLines(sourceIdx, weights) {
    // Clear existing lines
    while (svgOverlay.firstChild) {
      svgOverlay.removeChild(svgOverlay.firstChild);
    }

    // Match SVG size to the words row
    const rowRect = wordsRow.getBoundingClientRect();
    svgOverlay.setAttribute('width', rowRect.width);
    svgOverlay.setAttribute('height', rowRect.height);
    svgOverlay.style.width = rowRect.width + 'px';
    svgOverlay.style.height = rowRect.height + 'px';

    const sourceEl = wordEls[sourceIdx];
    const sourceRect = sourceEl.getBoundingClientRect();
    const sx = sourceRect.left + sourceRect.width / 2 - rowRect.left;
    const sy = sourceRect.top + sourceRect.height / 2 - rowRect.top;

    // Threshold: only draw lines for weights above 0.08
    const threshold = 0.08;

    // Sort by weight descending so thicker lines are drawn first (behind thinner ones)
    const indexed = weights.map((w, i) => ({ w, i }))
      .filter(d => d.i !== sourceIdx && d.w >= threshold)
      .sort((a, b) => a.w - b.w);

    indexed.forEach(({ w, i }) => {
      const targetEl = wordEls[i];
      const targetRect = targetEl.getBoundingClientRect();
      const tx = targetRect.left + targetRect.width / 2 - rowRect.left;
      const ty = targetRect.top + targetRect.height / 2 - rowRect.top;

      // Curved line using a quadratic bezier
      const midX = (sx + tx) / 2;
      // Curve upward (negative offset) for words on same row
      const curvature = -Math.abs(sx - tx) * 0.35 - 20;
      const midY = Math.min(sy, ty) + curvature;

      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', `M ${sx} ${sy} Q ${midX} ${midY} ${tx} ${ty}`);
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', `rgba(0, 212, 170, ${(w * 1.2).toFixed(3)})`);
      path.setAttribute('stroke-width', (w * 6 + 0.5).toFixed(1));
      path.setAttribute('stroke-linecap', 'round');
      path.setAttribute('class', `${uid}-attn-line`);

      // Animate the line drawing
      const length = path.getTotalLength ? path.getTotalLength() : 200;
      path.style.strokeDasharray = length;
      path.style.strokeDashoffset = length;
      path.style.transition = 'stroke-dashoffset 0.4s ease';

      svgOverlay.appendChild(path);

      // Trigger animation
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          path.style.strokeDashoffset = '0';
        });
      });
    });

    // Draw a small circle at the source
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', sx);
    circle.setAttribute('cy', sy);
    circle.setAttribute('r', '4');
    circle.setAttribute('fill', 'rgba(0, 212, 170, 0.8)');
    svgOverlay.appendChild(circle);
  }

  // Redraw lines on resize
  let resizeTimer;
  const resizeObserver = new ResizeObserver(() => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (selectedIdx !== null) {
        drawLines(selectedIdx, attentionMatrix[selectedIdx]);
      }
    }, 100);
  });
  resizeObserver.observe(wordsRow);

  // Select "it" by default to show the coreference attention pattern
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      selectWord(7);
    });
  });
}
