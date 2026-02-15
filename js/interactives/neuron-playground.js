// Neuron Playground — interactive single-neuron visualisation
// Lets users adjust inputs, weights, bias, and activation function
// to see how a neuron computes its output in real time.

const NUM_INPUTS = 3;

// ── Activation functions ────────────────────────────────────────
const activations = {
  ReLU:    z => Math.max(0, z),
  Sigmoid: z => 1 / (1 + Math.exp(-z)),
  Tanh:    z => Math.tanh(z),
};

function activationRange(name) {
  if (name === 'ReLU')    return { min: -1, max: 4 };
  if (name === 'Sigmoid') return { min: -0.1, max: 1.1 };
  return { min: -1.1, max: 1.1 }; // Tanh
}

// ── Helpers ─────────────────────────────────────────────────────
function fmt(v) {
  const s = v.toFixed(2);
  return v >= 0 ? s : s;           // keep sign
}

function fmtSigned(v) {
  return v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2);
}

function lerp(a, b, t) { return a + (b - a) * t; }

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// ── Inject scoped styles ────────────────────────────────────────
function injectStyles() {
  if (document.getElementById('neuron-playground-styles')) return;
  const style = document.createElement('style');
  style.id = 'neuron-playground-styles';
  style.textContent = `
    .np-widget {
      background: var(--bg-elevated, #292524);
      border: 1px solid var(--border-subtle, #292524);
      border-radius: var(--radius-md, 10px);
      overflow: hidden;
      font-family: var(--font-body, Georgia, serif);
      color: var(--text-primary, #E7E0D8);
    }

    .np-header {
      padding: var(--space-5, 1.25rem) var(--space-6, 1.5rem) var(--space-3, 0.75rem);
    }
    .np-header h3 {
      margin: 0 0 var(--space-1, 0.25rem);
      font-family: var(--font-heading, 'Syne', sans-serif);
      font-size: var(--text-xl, 1.25rem);
      font-weight: 700;
      color: var(--text-primary, #E7E0D8);
    }
    .np-header p {
      margin: 0;
      font-size: var(--text-sm, 0.875rem);
      color: var(--text-secondary, #A8A29E);
      line-height: var(--leading-normal, 1.6);
    }

    .np-body {
      padding: 0 var(--space-6, 1.5rem) var(--space-6, 1.5rem);
      display: flex;
      flex-direction: column;
      gap: var(--space-5, 1.25rem);
    }

    /* ── SVG diagram ── */
    .np-diagram-wrap {
      width: 100%;
      position: relative;
    }
    .np-diagram-wrap svg {
      width: 100%;
      height: auto;
      display: block;
    }

    /* ── Slider rows ── */
    .np-controls {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: var(--space-4, 1rem) var(--space-6, 1.5rem);
    }
    @media (max-width: 600px) {
      .np-controls { grid-template-columns: 1fr; }
    }
    .np-ctrl-group {
      display: flex;
      flex-direction: column;
      gap: var(--space-2, 0.5rem);
    }
    .np-ctrl-group-title {
      font-family: var(--font-heading, 'Syne', sans-serif);
      font-size: var(--text-sm, 0.875rem);
      font-weight: 600;
      color: var(--text-secondary, #A8A29E);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .np-slider-row {
      display: flex;
      align-items: center;
      gap: var(--space-3, 0.75rem);
    }
    .np-slider-label {
      min-width: 28px;
      font-family: var(--font-mono, 'Fira Code', monospace);
      font-size: var(--text-sm, 0.875rem);
      color: var(--text-secondary, #A8A29E);
    }
    .np-slider-row input[type="range"] {
      flex: 1;
      -webkit-appearance: none;
      appearance: none;
      height: 6px;
      border-radius: 3px;
      background: var(--bg-surface, #1C1917);
      outline: none;
      cursor: pointer;
    }
    .np-slider-row input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      width: 16px; height: 16px;
      border-radius: 50%;
      background: var(--accent-primary, #F0B429);
      border: 2px solid var(--bg-elevated, #292524);
      cursor: pointer;
      transition: transform 120ms ease;
    }
    .np-slider-row input[type="range"]::-webkit-slider-thumb:hover {
      transform: scale(1.2);
    }
    .np-slider-row input[type="range"]::-moz-range-thumb {
      width: 16px; height: 16px;
      border-radius: 50%;
      background: var(--accent-primary, #F0B429);
      border: 2px solid var(--bg-elevated, #292524);
      cursor: pointer;
    }
    .np-slider-val {
      min-width: 44px;
      text-align: right;
      font-family: var(--font-mono, 'Fira Code', monospace);
      font-size: var(--text-sm, 0.875rem);
      color: var(--text-primary, #E7E0D8);
    }

    /* ── Activation dropdown ── */
    .np-activation-row {
      display: flex;
      align-items: center;
      gap: var(--space-3, 0.75rem);
    }
    .np-activation-row label {
      font-family: var(--font-heading, 'Syne', sans-serif);
      font-size: var(--text-sm, 0.875rem);
      font-weight: 600;
      color: var(--text-secondary, #A8A29E);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .np-activation-row select {
      padding: var(--space-2, 0.5rem) var(--space-3, 0.75rem);
      background: var(--bg-surface, #1C1917);
      color: var(--text-primary, #E7E0D8);
      border: 1px solid var(--border-medium, #44403C);
      border-radius: var(--radius-sm, 6px);
      font-family: var(--font-mono, 'Fira Code', monospace);
      font-size: var(--text-sm, 0.875rem);
      cursor: pointer;
      outline: none;
    }
    .np-activation-row select:focus {
      border-color: var(--accent-primary, #F0B429);
    }

    /* ── Formula display ── */
    .np-formula {
      background: var(--bg-surface, #1C1917);
      border: 1px solid var(--border-subtle, #292524);
      border-radius: var(--radius-sm, 6px);
      padding: var(--space-3, 0.75rem) var(--space-4, 1rem);
      font-family: var(--font-mono, 'Fira Code', monospace);
      font-size: var(--text-sm, 0.875rem);
      color: var(--text-secondary, #A8A29E);
      line-height: 1.7;
      overflow-x: auto;
      white-space: nowrap;
    }
    .np-formula .np-val   { color: var(--accent-primary, #F0B429); }
    .np-formula .np-op    { color: var(--text-dim, #6B6560); }
    .np-formula .np-eq    { color: var(--accent-blue, #7EB8DA); }
    .np-formula .np-label { color: var(--text-secondary, #A8A29E); }

    /* ── Activation graph ── */
    .np-graph-section {
      display: flex;
      flex-direction: column;
      gap: var(--space-2, 0.5rem);
    }
    .np-graph-title {
      font-family: var(--font-heading, 'Syne', sans-serif);
      font-size: var(--text-sm, 0.875rem);
      font-weight: 600;
      color: var(--text-secondary, #A8A29E);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .np-graph-wrap {
      background: var(--bg-surface, #1C1917);
      border: 1px solid var(--border-subtle, #292524);
      border-radius: var(--radius-sm, 6px);
      padding: var(--space-3, 0.75rem);
    }
    .np-graph-wrap svg {
      width: 100%;
      height: auto;
      display: block;
    }

    /* ── Bottom row: graph + formula ── */
    .np-bottom-row {
      display: grid;
      grid-template-columns: 280px 1fr;
      gap: var(--space-5, 1.25rem);
      align-items: start;
    }
    @media (max-width: 700px) {
      .np-bottom-row { grid-template-columns: 1fr; }
    }

    .np-formula-section {
      display: flex;
      flex-direction: column;
      gap: var(--space-2, 0.5rem);
    }
    .np-formula-title {
      font-family: var(--font-heading, 'Syne', sans-serif);
      font-size: var(--text-sm, 0.875rem);
      font-weight: 600;
      color: var(--text-secondary, #A8A29E);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
  `;
  document.head.appendChild(style);
}

// ── Build the diagram SVG ───────────────────────────────────────
function buildDiagramSVG() {
  const W = 520, H = 200;
  const ns = 'http://www.w3.org/2000/svg';

  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

  // positions
  const inputX = 60;
  const neuronX = 260;
  const outputX = 460;
  const inputYs = [45, 100, 155];
  const neuronY = 100;
  const inputR = 22;
  const neuronR = 30;
  const outputR = 22;

  // connection lines (input -> neuron)
  const lines = [];
  for (let i = 0; i < NUM_INPUTS; i++) {
    const line = document.createElementNS(ns, 'line');
    line.setAttribute('x1', inputX + inputR);
    line.setAttribute('y1', inputYs[i]);
    line.setAttribute('x2', neuronX - neuronR);
    line.setAttribute('y2', neuronY);
    line.setAttribute('stroke-linecap', 'round');
    svg.appendChild(line);
    lines.push(line);
  }

  // bias line (comes from above)
  const biasLine = document.createElementNS(ns, 'line');
  biasLine.setAttribute('x1', neuronX);
  biasLine.setAttribute('y1', 8);
  biasLine.setAttribute('x2', neuronX);
  biasLine.setAttribute('y2', neuronY - neuronR);
  biasLine.setAttribute('stroke', 'var(--accent-purple, #C4A7E7)');
  biasLine.setAttribute('stroke-width', '2');
  biasLine.setAttribute('stroke-dasharray', '4 3');
  svg.appendChild(biasLine);

  // bias label
  const biasLabel = document.createElementNS(ns, 'text');
  biasLabel.setAttribute('x', neuronX + 12);
  biasLabel.setAttribute('y', 14);
  biasLabel.setAttribute('font-family', "var(--font-mono, 'Fira Code', monospace)");
  biasLabel.setAttribute('font-size', '11');
  biasLabel.setAttribute('fill', 'var(--accent-purple, #C4A7E7)');
  biasLabel.textContent = 'b';
  svg.appendChild(biasLabel);

  // output line (neuron -> output)
  const outLine = document.createElementNS(ns, 'line');
  outLine.setAttribute('x1', neuronX + neuronR);
  outLine.setAttribute('y1', neuronY);
  outLine.setAttribute('x2', outputX - outputR);
  outLine.setAttribute('y2', neuronY);
  outLine.setAttribute('stroke', 'var(--accent-primary, #F0B429)');
  outLine.setAttribute('stroke-width', '2.5');
  svg.appendChild(outLine);

  // input circles + labels
  const inputValueTexts = [];
  const inputLabels = ['x\u2081', 'x\u2082', 'x\u2083'];
  for (let i = 0; i < NUM_INPUTS; i++) {
    const g = document.createElementNS(ns, 'g');

    const circle = document.createElementNS(ns, 'circle');
    circle.setAttribute('cx', inputX);
    circle.setAttribute('cy', inputYs[i]);
    circle.setAttribute('r', inputR);
    circle.setAttribute('fill', 'var(--bg-surface, #1C1917)');
    circle.setAttribute('stroke', 'var(--accent-blue, #7EB8DA)');
    circle.setAttribute('stroke-width', '2');
    g.appendChild(circle);

    const label = document.createElementNS(ns, 'text');
    label.setAttribute('x', inputX);
    label.setAttribute('y', inputYs[i] - 6);
    label.setAttribute('text-anchor', 'middle');
    label.setAttribute('font-family', "var(--font-mono, 'Fira Code', monospace)");
    label.setAttribute('font-size', '11');
    label.setAttribute('fill', 'var(--text-secondary, #A8A29E)');
    label.textContent = inputLabels[i];
    g.appendChild(label);

    const val = document.createElementNS(ns, 'text');
    val.setAttribute('x', inputX);
    val.setAttribute('y', inputYs[i] + 10);
    val.setAttribute('text-anchor', 'middle');
    val.setAttribute('font-family', "var(--font-mono, 'Fira Code', monospace)");
    val.setAttribute('font-size', '12');
    val.setAttribute('font-weight', '600');
    val.setAttribute('fill', 'var(--text-primary, #E7E0D8)');
    g.appendChild(val);
    inputValueTexts.push(val);

    svg.appendChild(g);
  }

  // weight labels on connections
  const weightTexts = [];
  for (let i = 0; i < NUM_INPUTS; i++) {
    const mx = (inputX + inputR + neuronX - neuronR) / 2;
    const my = (inputYs[i] + neuronY) / 2;
    const offsetY = i === 1 ? -10 : (i === 0 ? -8 : 8);
    const wt = document.createElementNS(ns, 'text');
    wt.setAttribute('x', mx);
    wt.setAttribute('y', my + offsetY);
    wt.setAttribute('text-anchor', 'middle');
    wt.setAttribute('font-family', "var(--font-mono, 'Fira Code', monospace)");
    wt.setAttribute('font-size', '10');
    wt.setAttribute('fill', 'var(--text-dim, #6B6560)');
    svg.appendChild(wt);
    weightTexts.push(wt);
  }

  // neuron circle
  const neuronCircle = document.createElementNS(ns, 'circle');
  neuronCircle.setAttribute('cx', neuronX);
  neuronCircle.setAttribute('cy', neuronY);
  neuronCircle.setAttribute('r', neuronR);
  neuronCircle.setAttribute('fill', 'var(--bg-surface, #1C1917)');
  neuronCircle.setAttribute('stroke', 'var(--accent-primary, #F0B429)');
  neuronCircle.setAttribute('stroke-width', '2.5');
  svg.appendChild(neuronCircle);

  // sigma symbol inside neuron
  const sigma = document.createElementNS(ns, 'text');
  sigma.setAttribute('x', neuronX);
  sigma.setAttribute('y', neuronY + 2);
  sigma.setAttribute('text-anchor', 'middle');
  sigma.setAttribute('dominant-baseline', 'middle');
  sigma.setAttribute('font-family', "var(--font-body, Georgia, serif)");
  sigma.setAttribute('font-size', '22');
  sigma.setAttribute('fill', 'var(--accent-primary, #F0B429)');
  sigma.textContent = '\u03A3';
  svg.appendChild(sigma);

  // activation label below neuron
  const actLabel = document.createElementNS(ns, 'text');
  actLabel.setAttribute('x', neuronX);
  actLabel.setAttribute('y', neuronY + neuronR + 16);
  actLabel.setAttribute('text-anchor', 'middle');
  actLabel.setAttribute('font-family', "var(--font-mono, 'Fira Code', monospace)");
  actLabel.setAttribute('font-size', '10');
  actLabel.setAttribute('fill', 'var(--text-dim, #6B6560)');
  actLabel.textContent = '\u03C3(z)';
  svg.appendChild(actLabel);

  // output circle + value
  const outputCircle = document.createElementNS(ns, 'circle');
  outputCircle.setAttribute('cx', outputX);
  outputCircle.setAttribute('cy', neuronY);
  outputCircle.setAttribute('r', outputR);
  outputCircle.setAttribute('fill', 'var(--bg-surface, #1C1917)');
  outputCircle.setAttribute('stroke', 'var(--accent-warm, #E8553A)');
  outputCircle.setAttribute('stroke-width', '2');
  svg.appendChild(outputCircle);

  const outLabel = document.createElementNS(ns, 'text');
  outLabel.setAttribute('x', outputX);
  outLabel.setAttribute('y', neuronY - 6);
  outLabel.setAttribute('text-anchor', 'middle');
  outLabel.setAttribute('font-family', "var(--font-mono, 'Fira Code', monospace)");
  outLabel.setAttribute('font-size', '11');
  outLabel.setAttribute('fill', 'var(--text-secondary, #A8A29E)');
  outLabel.textContent = 'a';
  svg.appendChild(outLabel);

  const outVal = document.createElementNS(ns, 'text');
  outVal.setAttribute('x', outputX);
  outVal.setAttribute('y', neuronY + 10);
  outVal.setAttribute('text-anchor', 'middle');
  outVal.setAttribute('font-family', "var(--font-mono, 'Fira Code', monospace)");
  outVal.setAttribute('font-size', '12');
  outVal.setAttribute('font-weight', '600');
  outVal.setAttribute('fill', 'var(--text-primary, #E7E0D8)');
  svg.appendChild(outVal);

  return { svg, lines, inputValueTexts, weightTexts, outVal, actLabel };
}

// ── Build the activation graph SVG ──────────────────────────────
function buildGraphSVG() {
  const W = 250, H = 160;
  const pad = { top: 12, right: 16, bottom: 24, left: 32 };
  const ns = 'http://www.w3.org/2000/svg';

  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  // axes
  const xAxis = document.createElementNS(ns, 'line');
  xAxis.setAttribute('x1', pad.left); xAxis.setAttribute('y1', H - pad.bottom);
  xAxis.setAttribute('x2', W - pad.right); xAxis.setAttribute('y2', H - pad.bottom);
  xAxis.setAttribute('stroke', 'var(--border-medium, #44403C)'); xAxis.setAttribute('stroke-width', '1');
  svg.appendChild(xAxis);

  const yAxis = document.createElementNS(ns, 'line');
  yAxis.setAttribute('x1', pad.left); yAxis.setAttribute('y1', pad.top);
  yAxis.setAttribute('x2', pad.left); yAxis.setAttribute('y2', H - pad.bottom);
  yAxis.setAttribute('stroke', 'var(--border-medium, #44403C)'); yAxis.setAttribute('stroke-width', '1');
  svg.appendChild(yAxis);

  // axis labels
  const xLabel = document.createElementNS(ns, 'text');
  xLabel.setAttribute('x', pad.left + plotW / 2); xLabel.setAttribute('y', H - 3);
  xLabel.setAttribute('text-anchor', 'middle');
  xLabel.setAttribute('font-family', "var(--font-mono, 'Fira Code', monospace)");
  xLabel.setAttribute('font-size', '10'); xLabel.setAttribute('fill', 'var(--text-dim, #6B6560)');
  xLabel.textContent = 'z';
  svg.appendChild(xLabel);

  const yLabel = document.createElementNS(ns, 'text');
  yLabel.setAttribute('x', 8); yLabel.setAttribute('y', pad.top + plotH / 2);
  yLabel.setAttribute('text-anchor', 'middle');
  yLabel.setAttribute('dominant-baseline', 'middle');
  yLabel.setAttribute('font-family', "var(--font-mono, 'Fira Code', monospace)");
  yLabel.setAttribute('font-size', '10'); yLabel.setAttribute('fill', 'var(--text-dim, #6B6560)');
  yLabel.setAttribute('transform', `rotate(-90, 8, ${pad.top + plotH / 2})`);
  yLabel.textContent = 'a(z)';
  svg.appendChild(yLabel);

  // tick marks & labels
  const xTickLabels = [];
  const yTickLabels = [];

  // we'll create placeholder ticks and update them later
  for (let i = 0; i <= 4; i++) {
    const t = document.createElementNS(ns, 'text');
    t.setAttribute('y', H - pad.bottom + 14);
    t.setAttribute('text-anchor', 'middle');
    t.setAttribute('font-family', "var(--font-mono, 'Fira Code', monospace)");
    t.setAttribute('font-size', '9');
    t.setAttribute('fill', 'var(--text-dim, #6B6560)');
    svg.appendChild(t);
    xTickLabels.push(t);
  }
  for (let i = 0; i <= 2; i++) {
    const t = document.createElementNS(ns, 'text');
    t.setAttribute('x', pad.left - 5);
    t.setAttribute('text-anchor', 'end');
    t.setAttribute('dominant-baseline', 'middle');
    t.setAttribute('font-family', "var(--font-mono, 'Fira Code', monospace)");
    t.setAttribute('font-size', '9');
    t.setAttribute('fill', 'var(--text-dim, #6B6560)');
    svg.appendChild(t);
    yTickLabels.push(t);
  }

  // curve path
  const curvePath = document.createElementNS(ns, 'path');
  curvePath.setAttribute('fill', 'none');
  curvePath.setAttribute('stroke', 'var(--accent-primary, #F0B429)');
  curvePath.setAttribute('stroke-width', '2');
  curvePath.setAttribute('stroke-linejoin', 'round');
  svg.appendChild(curvePath);

  // current-z vertical guide line
  const guideLine = document.createElementNS(ns, 'line');
  guideLine.setAttribute('stroke', 'var(--accent-primary, #F0B429)');
  guideLine.setAttribute('stroke-width', '1');
  guideLine.setAttribute('stroke-dasharray', '3 3');
  guideLine.setAttribute('stroke-opacity', '0.4');
  svg.appendChild(guideLine);

  // dot for current z
  const dot = document.createElementNS(ns, 'circle');
  dot.setAttribute('r', '5');
  dot.setAttribute('fill', 'var(--accent-warm, #E8553A)');
  dot.setAttribute('stroke', 'var(--bg-surface, #1C1917)');
  dot.setAttribute('stroke-width', '2');
  svg.appendChild(dot);

  function update(actName, currentZ) {
    const fn = activations[actName];
    const zRange = { min: -5, max: 5 };
    const aRange = activationRange(actName);

    // helper: z -> pixel x
    const zToX = z => pad.left + ((z - zRange.min) / (zRange.max - zRange.min)) * plotW;
    // helper: a -> pixel y
    const aToY = a => pad.top + plotH - ((a - aRange.min) / (aRange.max - aRange.min)) * plotH;

    // update x tick labels
    const xSteps = [zRange.min, zRange.min / 2, 0, zRange.max / 2, zRange.max];
    xTickLabels.forEach((t, i) => {
      const v = xSteps[i];
      t.setAttribute('x', zToX(v));
      t.textContent = v.toFixed(0);
    });

    // update y tick labels
    const ySteps = [aRange.min, (aRange.min + aRange.max) / 2, aRange.max];
    yTickLabels.forEach((t, i) => {
      const v = ySteps[i];
      t.setAttribute('y', aToY(v));
      t.textContent = v % 1 === 0 ? v.toFixed(0) : v.toFixed(1);
    });

    // draw curve
    const steps = 100;
    let d = '';
    for (let i = 0; i <= steps; i++) {
      const z = zRange.min + (i / steps) * (zRange.max - zRange.min);
      const a = fn(z);
      const px = zToX(z);
      const py = aToY(clamp(a, aRange.min, aRange.max));
      d += (i === 0 ? 'M' : 'L') + `${px.toFixed(1)},${py.toFixed(1)}`;
    }
    curvePath.setAttribute('d', d);

    // position dot at current z
    const clampedZ = clamp(currentZ, zRange.min, zRange.max);
    const aVal = fn(clampedZ);
    const dx = zToX(clampedZ);
    const dy = aToY(clamp(aVal, aRange.min, aRange.max));
    dot.setAttribute('cx', dx);
    dot.setAttribute('cy', dy);

    // guide line
    guideLine.setAttribute('x1', dx);
    guideLine.setAttribute('y1', pad.top);
    guideLine.setAttribute('x2', dx);
    guideLine.setAttribute('y2', H - pad.bottom);
  }

  return { svg, update };
}

// ── Build a single slider row ───────────────────────────────────
function buildSliderRow(label, initial, onChange) {
  const row = document.createElement('div');
  row.className = 'np-slider-row';

  const lbl = document.createElement('span');
  lbl.className = 'np-slider-label';
  lbl.textContent = label;

  const input = document.createElement('input');
  input.type = 'range';
  input.min = '-2';
  input.max = '2';
  input.step = '0.05';
  input.value = String(initial);

  const val = document.createElement('span');
  val.className = 'np-slider-val';
  val.textContent = fmt(initial);

  input.addEventListener('input', () => {
    const v = parseFloat(input.value);
    val.textContent = fmt(v);
    onChange(v);
  });

  row.appendChild(lbl);
  row.appendChild(input);
  row.appendChild(val);

  return { row, input, val };
}

// ── Formula builder ─────────────────────────────────────────────
function buildFormulaHTML(xs, ws, bias, z, actName, a) {
  // z = (w1)(x1) + (w2)(x2) + (w3)(x3) + bias = result
  const terms = [];
  for (let i = 0; i < NUM_INPUTS; i++) {
    const wStr = `<span class="np-val">${fmt(ws[i])}</span>`;
    const xStr = `<span class="np-val">${fmt(xs[i])}</span>`;
    terms.push(`<span class="np-op">(</span>${wStr}<span class="np-op">)(</span>${xStr}<span class="np-op">)</span>`);
  }
  const biasStr = `<span class="np-val">${fmtSigned(bias)}</span>`;

  const zLine = `<span class="np-label">z</span> <span class="np-op">=</span> ${terms.join(` <span class="np-op">+</span> `)} ${biasStr} <span class="np-op">=</span> <span class="np-eq">${fmt(z)}</span>`;

  const aLine = `<span class="np-label">a</span> <span class="np-op">=</span> <span class="np-label">${actName}</span><span class="np-op">(</span><span class="np-eq">${fmt(z)}</span><span class="np-op">)</span> <span class="np-op">=</span> <span class="np-eq">${fmt(a)}</span>`;

  return `${zLine}<br>${aLine}`;
}

// ── Colour helpers for connection lines ─────────────────────────
function weightColor(w) {
  // positive = accent-green / teal, negative = accent-red
  if (w >= 0) return 'var(--accent-green, #9CCFA4)';
  return 'var(--accent-red, #EB6F92)';
}

function weightWidth(w) {
  // 1 at w=0, up to 5 at |w|=2
  return 1 + Math.abs(w) * 2;
}

// ── Mount ───────────────────────────────────────────────────────
export function mount(container) {
  injectStyles();

  // state
  const state = {
    xs: [1.0, 0.5, -0.5],
    ws: [0.8, -0.6, 1.2],
    bias: 0.1,
    activation: 'ReLU',
  };

  // root
  const widget = document.createElement('div');
  widget.className = 'np-widget';

  // header
  const header = document.createElement('div');
  header.className = 'np-header';
  header.innerHTML = `
    <h3>Neuron Playground</h3>
    <p>Adjust inputs, weights, and activation to see how a neuron computes</p>
  `;
  widget.appendChild(header);

  // body
  const body = document.createElement('div');
  body.className = 'np-body';

  // ── Diagram ──
  const diagramWrap = document.createElement('div');
  diagramWrap.className = 'np-diagram-wrap';
  const diagram = buildDiagramSVG();
  diagramWrap.appendChild(diagram.svg);
  body.appendChild(diagramWrap);

  // ── Controls grid ──
  const controls = document.createElement('div');
  controls.className = 'np-controls';

  // inputs group
  const inputsGroup = document.createElement('div');
  inputsGroup.className = 'np-ctrl-group';
  const inputsTitle = document.createElement('div');
  inputsTitle.className = 'np-ctrl-group-title';
  inputsTitle.textContent = 'Inputs';
  inputsGroup.appendChild(inputsTitle);

  const inputSliders = [];
  const inputLabels = ['x\u2081', 'x\u2082', 'x\u2083'];
  for (let i = 0; i < NUM_INPUTS; i++) {
    const idx = i;
    const s = buildSliderRow(inputLabels[i], state.xs[i], v => {
      state.xs[idx] = v;
      refresh();
    });
    inputsGroup.appendChild(s.row);
    inputSliders.push(s);
  }
  controls.appendChild(inputsGroup);

  // weights group
  const weightsGroup = document.createElement('div');
  weightsGroup.className = 'np-ctrl-group';
  const weightsTitle = document.createElement('div');
  weightsTitle.className = 'np-ctrl-group-title';
  weightsTitle.textContent = 'Weights';
  weightsGroup.appendChild(weightsTitle);

  const weightSliders = [];
  const weightLabelsArr = ['w\u2081', 'w\u2082', 'w\u2083'];
  for (let i = 0; i < NUM_INPUTS; i++) {
    const idx = i;
    const s = buildSliderRow(weightLabelsArr[i], state.ws[i], v => {
      state.ws[idx] = v;
      refresh();
    });
    weightsGroup.appendChild(s.row);
    weightSliders.push(s);
  }
  controls.appendChild(weightsGroup);

  body.appendChild(controls);

  // ── Bias + Activation row ──
  const biasActRow = document.createElement('div');
  biasActRow.className = 'np-controls';
  biasActRow.style.gridTemplateColumns = '1fr 1fr';

  // bias group
  const biasGroup = document.createElement('div');
  biasGroup.className = 'np-ctrl-group';
  const biasTitle = document.createElement('div');
  biasTitle.className = 'np-ctrl-group-title';
  biasTitle.textContent = 'Bias';
  biasGroup.appendChild(biasTitle);

  const biasSlider = buildSliderRow('b', state.bias, v => {
    state.bias = v;
    refresh();
  });
  biasGroup.appendChild(biasSlider.row);
  biasActRow.appendChild(biasGroup);

  // activation dropdown
  const actGroup = document.createElement('div');
  actGroup.className = 'np-ctrl-group';
  const actRow = document.createElement('div');
  actRow.className = 'np-activation-row';
  const actLabel = document.createElement('label');
  actLabel.textContent = 'Activation';
  const actSelect = document.createElement('select');
  for (const name of Object.keys(activations)) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    if (name === state.activation) opt.selected = true;
    actSelect.appendChild(opt);
  }
  actSelect.addEventListener('change', () => {
    state.activation = actSelect.value;
    refresh();
  });
  actRow.appendChild(actLabel);
  actRow.appendChild(actSelect);
  actGroup.appendChild(actRow);
  biasActRow.appendChild(actGroup);

  body.appendChild(biasActRow);

  // ── Bottom row: graph + formula ──
  const bottomRow = document.createElement('div');
  bottomRow.className = 'np-bottom-row';

  // graph section
  const graphSection = document.createElement('div');
  graphSection.className = 'np-graph-section';
  const graphTitle = document.createElement('div');
  graphTitle.className = 'np-graph-title';
  graphTitle.textContent = 'Activation Function';
  graphSection.appendChild(graphTitle);

  const graphWrap = document.createElement('div');
  graphWrap.className = 'np-graph-wrap';
  const graph = buildGraphSVG();
  graphWrap.appendChild(graph.svg);
  graphSection.appendChild(graphWrap);
  bottomRow.appendChild(graphSection);

  // formula section
  const formulaSection = document.createElement('div');
  formulaSection.className = 'np-formula-section';
  const formulaTitle = document.createElement('div');
  formulaTitle.className = 'np-formula-title';
  formulaTitle.textContent = 'Computation';
  formulaSection.appendChild(formulaTitle);

  const formulaDiv = document.createElement('div');
  formulaDiv.className = 'np-formula';
  formulaSection.appendChild(formulaDiv);
  bottomRow.appendChild(formulaSection);

  body.appendChild(bottomRow);
  widget.appendChild(body);
  container.appendChild(widget);

  // ── Refresh everything ──
  function refresh() {
    const { xs, ws, bias, activation: actName } = state;
    const fn = activations[actName];

    // weighted sum
    let z = bias;
    for (let i = 0; i < NUM_INPUTS; i++) {
      z += ws[i] * xs[i];
    }
    const a = fn(z);

    // diagram: input values
    for (let i = 0; i < NUM_INPUTS; i++) {
      diagram.inputValueTexts[i].textContent = fmt(xs[i]);
    }

    // diagram: weight labels + connection styles
    for (let i = 0; i < NUM_INPUTS; i++) {
      diagram.weightTexts[i].textContent = `w${i + 1}=${fmt(ws[i])}`;
      diagram.lines[i].setAttribute('stroke', weightColor(ws[i]));
      diagram.lines[i].setAttribute('stroke-width', weightWidth(ws[i]));
    }

    // diagram: output value
    diagram.outVal.textContent = fmt(a);

    // diagram: activation label
    diagram.actLabel.textContent = `${actName}(z)`;

    // formula
    formulaDiv.innerHTML = buildFormulaHTML(xs, ws, bias, z, actName, a);

    // activation graph
    graph.update(actName, z);
  }

  // initial render
  refresh();
}
