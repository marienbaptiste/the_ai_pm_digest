// Gradient Descent Visualizer — interactive module
// Exports mount(container) for the interactive loader

// --- Loss function and its derivative ---
function lossFunc(x) {
  return 0.5 * x * x + 0.3 * Math.sin(3 * x);
}

function lossGrad(x) {
  return x + 0.9 * Math.cos(3 * x);
}

// --- Configuration ---
const X_MIN = -4.5;
const X_MAX = 4.5;
const SVG_WIDTH = 700;
const SVG_HEIGHT = 340;
const PADDING = { top: 30, right: 30, bottom: 40, left: 50 };
const PLOT_W = SVG_WIDTH - PADDING.left - PADDING.right;
const PLOT_H = SVG_HEIGHT - PADDING.top - PADDING.bottom;
const MAX_TRAIL = 30;
const INITIAL_X = 3.8;
const INITIAL_LR = 0.15;

// --- Coordinate mapping ---
function yRange() {
  // Precompute y range over the visible x domain
  let yMin = Infinity, yMax = -Infinity;
  for (let px = 0; px <= PLOT_W; px++) {
    const x = X_MIN + (px / PLOT_W) * (X_MAX - X_MIN);
    const y = lossFunc(x);
    if (y < yMin) yMin = y;
    if (y > yMax) yMax = y;
  }
  const pad = (yMax - yMin) * 0.12;
  return { yMin: yMin - pad, yMax: yMax + pad };
}

const Y_RANGE = yRange();

function xToSvg(x) {
  return PADDING.left + ((x - X_MIN) / (X_MAX - X_MIN)) * PLOT_W;
}

function yToSvg(y) {
  // Flip y so higher loss is higher on screen (lower SVG y)
  return PADDING.top + (1 - (y - Y_RANGE.yMin) / (Y_RANGE.yMax - Y_RANGE.yMin)) * PLOT_H;
}

// --- Build the SVG path for the loss curve ---
function buildCurvePath() {
  const steps = 300;
  let d = '';
  for (let i = 0; i <= steps; i++) {
    const x = X_MIN + (i / steps) * (X_MAX - X_MIN);
    const y = lossFunc(x);
    const sx = xToSvg(x);
    const sy = yToSvg(y);
    d += (i === 0 ? 'M' : 'L') + sx.toFixed(2) + ',' + sy.toFixed(2);
  }
  return d;
}

// --- Build the filled area beneath the curve ---
function buildFillPath() {
  const curvePath = buildCurvePath();
  const bottomRight = `L${xToSvg(X_MAX).toFixed(2)},${yToSvg(Y_RANGE.yMin).toFixed(2)}`;
  const bottomLeft = `L${xToSvg(X_MIN).toFixed(2)},${yToSvg(Y_RANGE.yMin).toFixed(2)}`;
  return curvePath + bottomRight + bottomLeft + 'Z';
}

// --- SVG namespace helper ---
function svgEl(tag, attrs = {}) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) {
    el.setAttribute(k, v);
  }
  return el;
}

function htmlEl(tag, attrs = {}, text = '') {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'style' && typeof v === 'object') {
      Object.assign(el.style, v);
    } else {
      el.setAttribute(k, v);
    }
  }
  if (text) el.textContent = text;
  return el;
}

// --- Inject scoped styles ---
function injectStyles(container) {
  const id = 'gd-viz-styles';
  if (container.querySelector('#' + id)) return;

  const style = document.createElement('style');
  style.id = id;
  style.textContent = `
    .gd-viz {
      background: var(--bg-elevated, #292524);
      border: 1px solid var(--border-medium, #44403C);
      border-radius: var(--radius-md, 10px);
      padding: var(--space-6, 1.5rem);
      font-family: var(--font-body, Georgia, serif);
      color: var(--text-primary, #E7E0D8);
      max-width: 760px;
      margin: 0 auto;
    }

    .gd-viz__header {
      margin-bottom: var(--space-5, 1.25rem);
    }

    .gd-viz__title {
      font-family: var(--font-heading, 'Syne', system-ui, sans-serif);
      font-size: var(--text-2xl, 1.5rem);
      font-weight: 700;
      color: var(--text-primary, #E7E0D8);
      margin: 0 0 var(--space-1, 0.25rem) 0;
      line-height: var(--leading-tight, 1.25);
    }

    .gd-viz__subtitle {
      font-size: var(--text-sm, 0.875rem);
      color: var(--text-secondary, #A8A29E);
      margin: 0;
      line-height: var(--leading-normal, 1.6);
    }

    .gd-viz__svg-wrap {
      background: var(--bg-surface, #1C1917);
      border: 1px solid var(--border-subtle, #292524);
      border-radius: var(--radius-sm, 6px);
      overflow: hidden;
      margin-bottom: var(--space-4, 1rem);
    }

    .gd-viz__svg-wrap svg {
      display: block;
      width: 100%;
      height: auto;
    }

    .gd-viz__controls {
      display: flex;
      flex-wrap: wrap;
      gap: var(--space-3, 0.75rem);
      align-items: center;
      margin-bottom: var(--space-4, 1rem);
    }

    .gd-viz__btn {
      font-family: var(--font-heading, 'Syne', system-ui, sans-serif);
      font-size: var(--text-sm, 0.875rem);
      font-weight: 600;
      padding: var(--space-2, 0.5rem) var(--space-5, 1.25rem);
      border-radius: var(--radius-sm, 6px);
      border: 1px solid transparent;
      cursor: pointer;
      transition: all var(--duration-fast, 150ms) var(--ease-out, ease);
      letter-spacing: 0.01em;
    }

    .gd-viz__btn--step {
      background: var(--accent-primary, #F0B429);
      color: var(--text-inverse, #0C0A09);
      border-color: var(--accent-primary, #F0B429);
    }
    .gd-viz__btn--step:hover {
      filter: brightness(1.1);
      box-shadow: var(--shadow-glow, 0 0 20px rgba(240,180,41,0.2));
    }

    .gd-viz__btn--reset {
      background: transparent;
      color: var(--text-secondary, #A8A29E);
      border-color: var(--border-medium, #44403C);
    }
    .gd-viz__btn--reset:hover {
      background: var(--bg-hover, #3C3836);
      color: var(--text-primary, #E7E0D8);
    }

    .gd-viz__btn--auto {
      background: transparent;
      color: var(--accent-blue, #7EB8DA);
      border-color: var(--accent-blue, #7EB8DA);
    }
    .gd-viz__btn--auto:hover {
      background: var(--accent-blue-dim, rgba(126,184,218,0.19));
    }
    .gd-viz__btn--auto.active {
      background: var(--accent-blue, #7EB8DA);
      color: var(--text-inverse, #0C0A09);
    }

    .gd-viz__slider-group {
      display: flex;
      align-items: center;
      gap: var(--space-2, 0.5rem);
      flex: 1;
      min-width: 200px;
    }

    .gd-viz__slider-label {
      font-family: var(--font-mono, 'Fira Code', monospace);
      font-size: var(--text-xs, 0.75rem);
      color: var(--text-secondary, #A8A29E);
      white-space: nowrap;
    }

    .gd-viz__slider {
      flex: 1;
      -webkit-appearance: none;
      appearance: none;
      height: 4px;
      border-radius: 2px;
      background: var(--border-medium, #44403C);
      outline: none;
      cursor: pointer;
    }
    .gd-viz__slider::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: var(--accent-primary, #F0B429);
      border: 2px solid var(--bg-elevated, #292524);
      cursor: pointer;
    }
    .gd-viz__slider::-moz-range-thumb {
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: var(--accent-primary, #F0B429);
      border: 2px solid var(--bg-elevated, #292524);
      cursor: pointer;
    }

    .gd-viz__slider-value {
      font-family: var(--font-mono, 'Fira Code', monospace);
      font-size: var(--text-sm, 0.875rem);
      color: var(--accent-primary, #F0B429);
      min-width: 3.2em;
      text-align: right;
      font-weight: 600;
    }

    .gd-viz__stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: var(--space-3, 0.75rem);
    }

    .gd-viz__stat {
      background: var(--bg-surface, #1C1917);
      border: 1px solid var(--border-subtle, #292524);
      border-radius: var(--radius-sm, 6px);
      padding: var(--space-3, 0.75rem) var(--space-4, 1rem);
      text-align: center;
    }

    .gd-viz__stat-label {
      font-family: var(--font-mono, 'Fira Code', monospace);
      font-size: var(--text-xs, 0.75rem);
      color: var(--text-dim, #6B6560);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: var(--space-1, 0.25rem);
    }

    .gd-viz__stat-value {
      font-family: var(--font-mono, 'Fira Code', monospace);
      font-size: var(--text-lg, 1.125rem);
      font-weight: 700;
    }

    .gd-viz__stat-value--x      { color: var(--accent-blue, #7EB8DA); }
    .gd-viz__stat-value--loss    { color: var(--accent-warm, #E8553A); }
    .gd-viz__stat-value--lr      { color: var(--accent-primary, #F0B429); }
    .gd-viz__stat-value--steps   { color: var(--accent-purple, #C4A7E7); }

    .gd-viz__gradient-toggle {
      display: flex;
      align-items: center;
      gap: var(--space-2, 0.5rem);
      font-family: var(--font-mono, 'Fira Code', monospace);
      font-size: var(--text-xs, 0.75rem);
      color: var(--text-dim, #6B6560);
      cursor: pointer;
      user-select: none;
    }

    .gd-viz__gradient-toggle input {
      accent-color: var(--accent-warm, #E8553A);
      cursor: pointer;
    }

    /* Dot pulse animation */
    @keyframes gd-pulse {
      0%, 100% { r: 7; opacity: 1; }
      50%      { r: 10; opacity: 0.8; }
    }
  `;
  container.prepend(style);
}

// --- Mount ---
export function mount(container) {
  injectStyles(container);

  // State
  let currentX = INITIAL_X;
  let learningRate = INITIAL_LR;
  let stepCount = 0;
  let trail = []; // array of {x, y}
  let showGradient = true;
  let autoRunning = false;
  let autoTimer = null;

  // --- Build DOM ---
  const root = htmlEl('div', { class: 'gd-viz' });

  // Header
  const header = htmlEl('div', { class: 'gd-viz__header' });
  header.appendChild(htmlEl('h3', { class: 'gd-viz__title' }, 'Gradient Descent Visualizer'));
  header.appendChild(htmlEl('p', { class: 'gd-viz__subtitle' }, 'Adjust the learning rate and step through optimization'));
  root.appendChild(header);

  // SVG container
  const svgWrap = htmlEl('div', { class: 'gd-viz__svg-wrap' });
  const svg = svgEl('svg', {
    viewBox: `0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`,
    preserveAspectRatio: 'xMidYMid meet',
  });

  // Defs — gradient fill for curve area
  const defs = svgEl('defs');
  const grad = svgEl('linearGradient', { id: 'gd-curve-fill', x1: '0', y1: '0', x2: '0', y2: '1' });
  const stop1 = svgEl('stop', { offset: '0%', 'stop-color': 'var(--accent-warm, #E8553A)', 'stop-opacity': '0.18' });
  const stop2 = svgEl('stop', { offset: '100%', 'stop-color': 'var(--accent-warm, #E8553A)', 'stop-opacity': '0.02' });
  grad.appendChild(stop1);
  grad.appendChild(stop2);
  defs.appendChild(grad);
  svg.appendChild(defs);

  // Plot group
  const plotG = svgEl('g');

  // Grid lines
  const gridG = svgEl('g');
  // Horizontal grid
  const yTicks = 6;
  for (let i = 0; i <= yTicks; i++) {
    const yVal = Y_RANGE.yMin + (i / yTicks) * (Y_RANGE.yMax - Y_RANGE.yMin);
    const sy = yToSvg(yVal);
    gridG.appendChild(svgEl('line', {
      x1: PADDING.left, y1: sy, x2: SVG_WIDTH - PADDING.right, y2: sy,
      stroke: 'var(--border-subtle, #292524)', 'stroke-width': '1', 'stroke-dasharray': '3,4',
    }));
    // Y-axis label
    const label = svgEl('text', {
      x: PADDING.left - 8, y: sy + 4,
      fill: 'var(--text-dim, #6B6560)',
      'font-size': '10',
      'font-family': 'var(--font-mono, monospace)',
      'text-anchor': 'end',
    });
    label.textContent = yVal.toFixed(1);
    gridG.appendChild(label);
  }
  // Vertical grid
  const xTicks = 9;
  for (let i = 0; i <= xTicks; i++) {
    const xVal = X_MIN + (i / xTicks) * (X_MAX - X_MIN);
    const sx = xToSvg(xVal);
    gridG.appendChild(svgEl('line', {
      x1: sx, y1: PADDING.top, x2: sx, y2: SVG_HEIGHT - PADDING.bottom,
      stroke: 'var(--border-subtle, #292524)', 'stroke-width': '1', 'stroke-dasharray': '3,4',
    }));
    // X-axis label
    const label = svgEl('text', {
      x: sx, y: SVG_HEIGHT - PADDING.bottom + 16,
      fill: 'var(--text-dim, #6B6560)',
      'font-size': '10',
      'font-family': 'var(--font-mono, monospace)',
      'text-anchor': 'middle',
    });
    label.textContent = xVal.toFixed(1);
    gridG.appendChild(label);
  }
  plotG.appendChild(gridG);

  // Axis labels
  const xAxisLabel = svgEl('text', {
    x: PADDING.left + PLOT_W / 2,
    y: SVG_HEIGHT - 4,
    fill: 'var(--text-secondary, #A8A29E)',
    'font-size': '12',
    'font-family': 'var(--font-heading, sans-serif)',
    'text-anchor': 'middle',
    'font-weight': '600',
  });
  xAxisLabel.textContent = 'Parameter (x)';
  plotG.appendChild(xAxisLabel);

  const yAxisLabel = svgEl('text', {
    x: 14,
    y: PADDING.top + PLOT_H / 2,
    fill: 'var(--text-secondary, #A8A29E)',
    'font-size': '12',
    'font-family': 'var(--font-heading, sans-serif)',
    'text-anchor': 'middle',
    'font-weight': '600',
    transform: `rotate(-90, 14, ${PADDING.top + PLOT_H / 2})`,
  });
  yAxisLabel.textContent = 'Loss f(x)';
  plotG.appendChild(yAxisLabel);

  // Filled area
  const fillPath = svgEl('path', {
    d: buildFillPath(),
    fill: 'url(#gd-curve-fill)',
  });
  plotG.appendChild(fillPath);

  // Loss curve
  const curvePath = svgEl('path', {
    d: buildCurvePath(),
    fill: 'none',
    stroke: 'var(--accent-warm, #E8553A)',
    'stroke-width': '2.5',
    'stroke-linecap': 'round',
    'stroke-linejoin': 'round',
  });
  plotG.appendChild(curvePath);

  // Trail group (ghost dots)
  const trailG = svgEl('g');
  plotG.appendChild(trailG);

  // Gradient arrow group
  const gradArrowG = svgEl('g');
  plotG.appendChild(gradArrowG);

  // Current position dot
  const dot = svgEl('circle', {
    cx: xToSvg(currentX),
    cy: yToSvg(lossFunc(currentX)),
    r: '7',
    fill: 'var(--accent-red, #EB6F92)',
    stroke: 'var(--bg-surface, #1C1917)',
    'stroke-width': '2.5',
    style: 'transition: cx 0.35s cubic-bezier(0.34,1.56,0.64,1), cy 0.35s cubic-bezier(0.34,1.56,0.64,1); filter: drop-shadow(0 0 6px rgba(235,111,146,0.5));',
  });
  plotG.appendChild(dot);

  // Small label near the dot
  const dotLabel = svgEl('text', {
    x: xToSvg(currentX),
    y: yToSvg(lossFunc(currentX)) - 14,
    fill: 'var(--accent-red, #EB6F92)',
    'font-size': '10',
    'font-family': 'var(--font-mono, monospace)',
    'text-anchor': 'middle',
    'font-weight': '700',
    style: 'transition: x 0.35s cubic-bezier(0.34,1.56,0.64,1), y 0.35s cubic-bezier(0.34,1.56,0.64,1);',
  });
  dotLabel.textContent = `(${currentX.toFixed(2)}, ${lossFunc(currentX).toFixed(2)})`;
  plotG.appendChild(dotLabel);

  svg.appendChild(plotG);
  svgWrap.appendChild(svg);
  root.appendChild(svgWrap);

  // Controls row
  const controls = htmlEl('div', { class: 'gd-viz__controls' });

  const stepBtn = htmlEl('button', { class: 'gd-viz__btn gd-viz__btn--step' }, 'Step');
  const resetBtn = htmlEl('button', { class: 'gd-viz__btn gd-viz__btn--reset' }, 'Reset');
  const autoBtn = htmlEl('button', { class: 'gd-viz__btn gd-viz__btn--auto' }, 'Auto Run');

  const sliderGroup = htmlEl('div', { class: 'gd-viz__slider-group' });
  const sliderLabel = htmlEl('span', { class: 'gd-viz__slider-label' }, 'LR:');
  const slider = htmlEl('input', {
    type: 'range',
    class: 'gd-viz__slider',
    min: '0.01',
    max: '1.0',
    step: '0.01',
    value: String(learningRate),
  });
  const sliderValue = htmlEl('span', { class: 'gd-viz__slider-value' }, learningRate.toFixed(2));

  sliderGroup.appendChild(sliderLabel);
  sliderGroup.appendChild(slider);
  sliderGroup.appendChild(sliderValue);

  const gradToggle = htmlEl('label', { class: 'gd-viz__gradient-toggle' });
  const gradCheckbox = htmlEl('input', { type: 'checkbox', checked: '' });
  gradCheckbox.checked = showGradient;
  gradToggle.appendChild(gradCheckbox);
  gradToggle.appendChild(document.createTextNode('Show gradient'));

  controls.appendChild(stepBtn);
  controls.appendChild(resetBtn);
  controls.appendChild(autoBtn);
  controls.appendChild(sliderGroup);
  controls.appendChild(gradToggle);
  root.appendChild(controls);

  // Stats row
  const stats = htmlEl('div', { class: 'gd-viz__stats' });

  function makeStat(label, valueClass) {
    const card = htmlEl('div', { class: 'gd-viz__stat' });
    card.appendChild(htmlEl('div', { class: 'gd-viz__stat-label' }, label));
    const val = htmlEl('div', { class: `gd-viz__stat-value ${valueClass}` }, '--');
    card.appendChild(val);
    stats.appendChild(card);
    return val;
  }

  const statX = makeStat('Position (x)', 'gd-viz__stat-value--x');
  const statLoss = makeStat('Loss f(x)', 'gd-viz__stat-value--loss');
  const statLR = makeStat('Learning Rate', 'gd-viz__stat-value--lr');
  const statSteps = makeStat('Steps', 'gd-viz__stat-value--steps');

  root.appendChild(stats);
  container.appendChild(root);

  // --- Rendering helpers ---

  function updateDot(animate = true) {
    const sx = xToSvg(currentX);
    const sy = yToSvg(lossFunc(currentX));

    if (!animate) {
      dot.style.transition = 'none';
      dotLabel.style.transition = 'none';
    }

    dot.setAttribute('cx', sx);
    dot.setAttribute('cy', sy);

    // Use a short timeout to re-apply SVG attribute-based positioning for the label
    // SVG <text> uses x/y attributes, not cx/cy, and CSS transitions on attributes
    // need the transition set via the style. We apply with setAttribute for correct SVG rendering.
    dotLabel.setAttribute('x', sx);
    dotLabel.setAttribute('y', sy - 14);
    dotLabel.textContent = `(${currentX.toFixed(2)}, ${lossFunc(currentX).toFixed(2)})`;

    if (!animate) {
      // Force reflow and restore transitions
      dot.getBoundingClientRect();
      dot.style.transition = '';
      dotLabel.style.transition = '';
    }
  }

  function renderTrail() {
    // Clear old trail
    while (trailG.firstChild) trailG.removeChild(trailG.firstChild);

    const len = trail.length;
    for (let i = 0; i < len; i++) {
      const { x, y } = trail[i];
      const opacity = 0.12 + 0.45 * (i / Math.max(len - 1, 1));
      const radius = 3 + 2 * (i / Math.max(len - 1, 1));
      const ghostDot = svgEl('circle', {
        cx: xToSvg(x),
        cy: yToSvg(y),
        r: String(radius.toFixed(1)),
        fill: 'var(--accent-red, #EB6F92)',
        opacity: String(opacity.toFixed(2)),
      });
      trailG.appendChild(ghostDot);

      // Draw a faint line connecting consecutive trail points
      if (i > 0) {
        const prev = trail[i - 1];
        const line = svgEl('line', {
          x1: xToSvg(prev.x),
          y1: yToSvg(prev.y),
          x2: xToSvg(x),
          y2: yToSvg(y),
          stroke: 'var(--accent-red, #EB6F92)',
          'stroke-width': '1',
          opacity: String((opacity * 0.5).toFixed(2)),
          'stroke-dasharray': '2,3',
        });
        trailG.appendChild(line);
      }
    }
  }

  function renderGradientArrow() {
    while (gradArrowG.firstChild) gradArrowG.removeChild(gradArrowG.firstChild);
    if (!showGradient) return;

    const g = lossGrad(currentX);
    const y = lossFunc(currentX);
    const sx = xToSvg(currentX);
    const sy = yToSvg(y);

    // Tangent line: draw a short segment centered on current point
    const dx = 0.8; // half-width in x units
    const x1 = currentX - dx;
    const x2 = currentX + dx;
    const y1 = y - g * dx;
    const y2 = y + g * dx;

    const tangent = svgEl('line', {
      x1: xToSvg(x1),
      y1: yToSvg(y1),
      x2: xToSvg(x2),
      y2: yToSvg(y2),
      stroke: 'var(--accent-purple, #C4A7E7)',
      'stroke-width': '1.8',
      opacity: '0.7',
      'stroke-dasharray': '5,3',
    });
    gradArrowG.appendChild(tangent);

    // Arrow showing descent direction
    const arrowLen = Math.min(Math.abs(g) * learningRate, 2.5) * Math.sign(-g);
    const arrowEndX = currentX + arrowLen;
    const arrowEndY = lossFunc(currentX); // keep arrow horizontal for clarity

    const arrow = svgEl('line', {
      x1: sx,
      y1: sy,
      x2: xToSvg(arrowEndX),
      y2: sy,
      stroke: 'var(--accent-primary, #F0B429)',
      'stroke-width': '2.2',
      'marker-end': 'url(#gd-arrowhead)',
      opacity: '0.9',
    });
    gradArrowG.appendChild(arrow);

    // Gradient value label
    const gradLabel = svgEl('text', {
      x: sx + 4,
      y: sy + 20,
      fill: 'var(--accent-purple, #C4A7E7)',
      'font-size': '10',
      'font-family': 'var(--font-mono, monospace)',
      opacity: '0.85',
    });
    gradLabel.textContent = `grad = ${g.toFixed(3)}`;
    gradArrowG.appendChild(gradLabel);
  }

  // Arrowhead marker
  const arrowMarker = svgEl('marker', {
    id: 'gd-arrowhead',
    viewBox: '0 0 10 10',
    refX: '9',
    refY: '5',
    markerWidth: '6',
    markerHeight: '6',
    orient: 'auto-start-reverse',
  });
  const arrowPath = svgEl('path', {
    d: 'M 0 0 L 10 5 L 0 10 Z',
    fill: 'var(--accent-primary, #F0B429)',
  });
  arrowMarker.appendChild(arrowPath);
  defs.appendChild(arrowMarker);

  function updateStats() {
    const y = lossFunc(currentX);
    statX.textContent = currentX.toFixed(4);
    statLoss.textContent = y.toFixed(4);
    statLR.textContent = learningRate.toFixed(2);
    statSteps.textContent = String(stepCount);
  }

  function fullRender(animate = true) {
    updateDot(animate);
    renderTrail();
    renderGradientArrow();
    updateStats();
  }

  // --- Actions ---

  function doStep() {
    const g = lossGrad(currentX);
    const prevX = currentX;
    const prevY = lossFunc(currentX);

    // Record trail
    trail.push({ x: prevX, y: prevY });
    if (trail.length > MAX_TRAIL) trail.shift();

    // Gradient descent update
    currentX = currentX - learningRate * g;

    // Clamp to visible range
    currentX = Math.max(X_MIN + 0.05, Math.min(X_MAX - 0.05, currentX));

    stepCount++;
    fullRender(true);
  }

  function doReset() {
    stopAuto();
    currentX = INITIAL_X;
    stepCount = 0;
    trail = [];
    fullRender(false);
  }

  function toggleAuto() {
    if (autoRunning) {
      stopAuto();
    } else {
      startAuto();
    }
  }

  function startAuto() {
    autoRunning = true;
    autoBtn.classList.add('active');
    autoBtn.textContent = 'Stop';
    autoTimer = setInterval(() => {
      doStep();
    }, 450);
  }

  function stopAuto() {
    autoRunning = false;
    autoBtn.classList.remove('active');
    autoBtn.textContent = 'Auto Run';
    if (autoTimer) {
      clearInterval(autoTimer);
      autoTimer = null;
    }
  }

  // --- Event listeners ---

  stepBtn.addEventListener('click', doStep);
  resetBtn.addEventListener('click', doReset);
  autoBtn.addEventListener('click', toggleAuto);

  slider.addEventListener('input', () => {
    learningRate = parseFloat(slider.value);
    sliderValue.textContent = learningRate.toFixed(2);
    updateStats();
  });

  gradCheckbox.addEventListener('change', () => {
    showGradient = gradCheckbox.checked;
    renderGradientArrow();
  });

  // --- Initial render ---
  fullRender(false);
}
