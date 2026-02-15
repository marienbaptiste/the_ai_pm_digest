// Confusion Matrix Explorer — interactive widget
// Users adjust TP, FP, TN, FN counts and see precision, recall, F1, accuracy in real-time.

const SCENARIOS = {
  'Balanced':       { tp: 40, fp: 10, fn: 10, tn: 40 },
  'High Precision': { tp: 30, fp: 2,  fn: 20, tn: 48 },
  'High Recall':    { tp: 45, fp: 25, fn: 2,  tn: 28 },
  'Imbalanced':     { tp: 5,  fp: 3,  fn: 2,  tn: 90 },
};

const CELLS = [
  { key: 'tp', label: 'True Positive',  abbr: 'TP', row: 0, col: 0, bg: '#16a34a', bgAlpha: '33' },
  { key: 'fp', label: 'False Positive', abbr: 'FP', row: 0, col: 1, bg: '#ef4444', bgAlpha: '30' },
  { key: 'fn', label: 'False Negative', abbr: 'FN', row: 1, col: 0, bg: '#ef4444', bgAlpha: '1a' },
  { key: 'tn', label: 'True Negative',  abbr: 'TN', row: 1, col: 1, bg: '#9ccfa4', bgAlpha: '30' },
];

function computeMetrics({ tp, fp, fn, tn }) {
  const total = tp + tn + fp + fn;
  const accuracy  = total > 0 ? (tp + tn) / total : 0;
  const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
  const recall    = (tp + fn) > 0 ? tp / (tp + fn) : 0;
  const f1        = (precision + recall) > 0
    ? 2 * (precision * recall) / (precision + recall)
    : 0;
  return { accuracy, precision, recall, f1 };
}

function metricBarColor(value) {
  if (value > 0.8) return 'var(--accent-green, #9ccfa4)';
  if (value >= 0.5) return 'var(--accent-warm, #f59e0b)';
  return 'var(--accent-red, #ef4444)';
}

const STYLE = `
  .cm-widget {
    background: var(--bg-elevated, #1e2740);
    border: 1px solid var(--border-subtle, #2a3450);
    border-radius: var(--radius-md, 12px);
    padding: var(--space-6, 24px);
    font-family: var(--font-body, system-ui, -apple-system, sans-serif);
    color: var(--text-primary, #f0f4f8);
    max-width: 640px;
    margin: 0 auto;
  }

  .cm-widget * { box-sizing: border-box; }

  .cm-header {
    text-align: center;
    margin-bottom: var(--space-5, 20px);
  }

  .cm-title {
    font-family: var(--font-heading, var(--font-body, system-ui));
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-primary, #f0f4f8);
    margin: 0 0 4px 0;
  }

  .cm-subtitle {
    font-family: var(--font-body, system-ui);
    font-size: 0.85rem;
    color: var(--text-secondary, #94a3b8);
    margin: 0;
  }

  /* Scenario selector */
  .cm-scenario-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-3, 12px);
    margin-bottom: var(--space-5, 20px);
  }

  .cm-scenario-label {
    font-size: 0.8rem;
    color: var(--text-secondary, #94a3b8);
    font-family: var(--font-mono, 'Fira Code', monospace);
  }

  .cm-scenario-select {
    background: var(--bg-surface, #111827);
    color: var(--text-primary, #f0f4f8);
    border: 1px solid var(--border-medium, #3a4560);
    border-radius: 6px;
    padding: 6px 12px;
    font-family: var(--font-mono, 'Fira Code', monospace);
    font-size: 0.8rem;
    cursor: pointer;
    outline: none;
    transition: border-color 0.2s;
  }

  .cm-scenario-select:hover,
  .cm-scenario-select:focus {
    border-color: var(--accent-primary, #00d4aa);
  }

  /* Grid layout */
  .cm-grid-wrapper {
    display: grid;
    grid-template-columns: auto 1fr 1fr;
    grid-template-rows: auto 1fr 1fr;
    gap: 0;
    margin-bottom: var(--space-6, 24px);
  }

  .cm-corner { /* top-left empty corner */ }

  .cm-col-header {
    text-align: center;
    font-family: var(--font-mono, 'Fira Code', monospace);
    font-size: 0.7rem;
    color: var(--text-dim, #64748b);
    padding: var(--space-2, 8px) var(--space-2, 8px);
    line-height: 1.3;
    border-bottom: 1px solid var(--border-subtle, #2a3450);
  }

  .cm-row-header {
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-mono, 'Fira Code', monospace);
    font-size: 0.7rem;
    color: var(--text-dim, #64748b);
    padding: var(--space-2, 8px);
    writing-mode: vertical-rl;
    text-orientation: mixed;
    transform: rotate(180deg);
    border-right: 1px solid var(--border-subtle, #2a3450);
    white-space: nowrap;
    min-width: 32px;
  }

  /* Span the row header across both rows of its side */
  .cm-row-header-actual-pos {
    grid-row: 2 / 3;
    grid-column: 1 / 2;
  }
  .cm-row-header-actual-neg {
    grid-row: 3 / 4;
    grid-column: 1 / 2;
  }

  .cm-cell {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: var(--space-4, 16px) var(--space-3, 12px);
    border: 1px solid var(--border-subtle, #2a3450);
    min-height: 110px;
    transition: background 0.3s;
  }

  .cm-cell-label {
    font-family: var(--font-mono, 'Fira Code', monospace);
    font-size: 0.7rem;
    color: var(--text-secondary, #94a3b8);
    margin-bottom: 6px;
    text-align: center;
    line-height: 1.3;
  }

  .cm-cell-abbr {
    font-weight: 700;
    font-size: 0.85rem;
    color: var(--text-primary, #f0f4f8);
  }

  .cm-cell-count {
    font-family: var(--font-mono, 'Fira Code', monospace);
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary, #f0f4f8);
    margin: 4px 0 8px 0;
    min-width: 48px;
    text-align: center;
  }

  .cm-cell-buttons {
    display: flex;
    gap: 6px;
  }

  .cm-btn {
    width: 30px;
    height: 30px;
    border-radius: 6px;
    border: 1px solid var(--border-medium, #3a4560);
    background: var(--bg-surface, #111827);
    color: var(--text-primary, #f0f4f8);
    font-size: 1.1rem;
    font-weight: 700;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background 0.15s, border-color 0.15s;
    user-select: none;
    line-height: 1;
    padding: 0;
    font-family: var(--font-mono, 'Fira Code', monospace);
  }

  .cm-btn:hover {
    background: var(--bg-elevated, #1e2740);
    border-color: var(--accent-primary, #00d4aa);
  }

  .cm-btn:active {
    transform: scale(0.92);
  }

  /* Metrics section */
  .cm-metrics {
    display: flex;
    flex-direction: column;
    gap: var(--space-3, 12px);
  }

  .cm-metrics-title {
    font-family: var(--font-mono, 'Fira Code', monospace);
    font-size: 0.75rem;
    color: var(--text-dim, #64748b);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: var(--space-2, 8px);
  }

  .cm-metric-row {
    display: grid;
    grid-template-columns: 90px 1fr 56px;
    align-items: center;
    gap: var(--space-3, 12px);
  }

  .cm-metric-name {
    font-family: var(--font-mono, 'Fira Code', monospace);
    font-size: 0.78rem;
    color: var(--text-secondary, #94a3b8);
  }

  .cm-metric-bar-track {
    height: 10px;
    background: var(--bg-surface, #111827);
    border-radius: 5px;
    overflow: hidden;
    border: 1px solid var(--border-subtle, #2a3450);
  }

  .cm-metric-bar-fill {
    height: 100%;
    border-radius: 5px;
    transition: width 0.4s ease, background 0.4s ease;
    min-width: 0;
  }

  .cm-metric-value {
    font-family: var(--font-mono, 'Fira Code', monospace);
    font-size: 0.82rem;
    font-weight: 600;
    text-align: right;
    transition: color 0.3s;
  }

  .cm-formula {
    font-family: var(--font-mono, 'Fira Code', monospace);
    font-size: 0.65rem;
    color: var(--text-dim, #64748b);
    text-align: center;
    margin-top: var(--space-4, 16px);
    line-height: 1.7;
  }
`;

export function mount(container) {
  // State
  let state = { tp: 40, fp: 10, fn: 5, tn: 45 };

  // Inject styles
  const styleEl = document.createElement('style');
  styleEl.textContent = STYLE;

  // Build DOM
  const widget = document.createElement('div');
  widget.className = 'cm-widget';

  // --- Header ---
  const header = document.createElement('div');
  header.className = 'cm-header';
  header.innerHTML = `
    <h3 class="cm-title">Confusion Matrix Explorer</h3>
    <p class="cm-subtitle">Adjust counts to see how metrics change</p>
  `;

  // --- Scenario selector ---
  const scenarioRow = document.createElement('div');
  scenarioRow.className = 'cm-scenario-row';
  const scenarioLabel = document.createElement('span');
  scenarioLabel.className = 'cm-scenario-label';
  scenarioLabel.textContent = 'Scenario:';

  const scenarioSelect = document.createElement('select');
  scenarioSelect.className = 'cm-scenario-select';

  // Add a "Custom" option at the top
  const customOpt = document.createElement('option');
  customOpt.value = '__custom__';
  customOpt.textContent = 'Custom';
  scenarioSelect.appendChild(customOpt);

  for (const name of Object.keys(SCENARIOS)) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    scenarioSelect.appendChild(opt);
  }
  scenarioSelect.value = '__custom__'; // start as custom since initial values differ from Balanced

  scenarioRow.appendChild(scenarioLabel);
  scenarioRow.appendChild(scenarioSelect);

  // --- Grid ---
  const gridWrapper = document.createElement('div');
  gridWrapper.className = 'cm-grid-wrapper';

  // Corner (empty)
  const corner = document.createElement('div');
  corner.className = 'cm-corner';
  gridWrapper.appendChild(corner);

  // Column headers (row 0, cols 1 & 2)
  const colH1 = document.createElement('div');
  colH1.className = 'cm-col-header';
  colH1.textContent = 'Predicted Positive';
  const colH2 = document.createElement('div');
  colH2.className = 'cm-col-header';
  colH2.textContent = 'Predicted Negative';
  gridWrapper.appendChild(colH1);
  gridWrapper.appendChild(colH2);

  // Cell references for updates
  const cellEls = {};

  // Rows
  const rowLabels = ['Actually Positive', 'Actually Negative'];
  const rowCells = [
    [CELLS[0], CELLS[1]], // TP, FP
    [CELLS[2], CELLS[3]], // FN, TN
  ];

  rowLabels.forEach((label, ri) => {
    // Row header
    const rowH = document.createElement('div');
    rowH.className = 'cm-row-header';
    rowH.textContent = label;
    gridWrapper.appendChild(rowH);

    // Two cells
    rowCells[ri].forEach((cellDef) => {
      const cell = document.createElement('div');
      cell.className = 'cm-cell';
      cell.style.background = `${cellDef.bg}${cellDef.bgAlpha}`;

      const labelEl = document.createElement('div');
      labelEl.className = 'cm-cell-label';
      labelEl.innerHTML = `${cellDef.label}<br><span class="cm-cell-abbr">${cellDef.abbr}</span>`;

      const countEl = document.createElement('div');
      countEl.className = 'cm-cell-count';
      countEl.textContent = state[cellDef.key];

      const btns = document.createElement('div');
      btns.className = 'cm-cell-buttons';

      const minusBtn = document.createElement('button');
      minusBtn.className = 'cm-btn';
      minusBtn.textContent = '\u2212'; // minus sign
      minusBtn.setAttribute('aria-label', `Decrease ${cellDef.abbr}`);

      const plusBtn = document.createElement('button');
      plusBtn.className = 'cm-btn';
      plusBtn.textContent = '+';
      plusBtn.setAttribute('aria-label', `Increase ${cellDef.abbr}`);

      minusBtn.addEventListener('click', () => {
        if (state[cellDef.key] > 0) {
          state[cellDef.key]--;
          scenarioSelect.value = '__custom__';
          render();
        }
      });

      plusBtn.addEventListener('click', () => {
        state[cellDef.key]++;
        scenarioSelect.value = '__custom__';
        render();
      });

      btns.appendChild(minusBtn);
      btns.appendChild(plusBtn);

      cell.appendChild(labelEl);
      cell.appendChild(countEl);
      cell.appendChild(btns);
      gridWrapper.appendChild(cell);

      cellEls[cellDef.key] = countEl;
    });
  });

  // --- Metrics ---
  const metricsSection = document.createElement('div');
  metricsSection.className = 'cm-metrics';

  const metricsTitle = document.createElement('div');
  metricsTitle.className = 'cm-metrics-title';
  metricsTitle.textContent = 'Calculated Metrics';
  metricsSection.appendChild(metricsTitle);

  const metricDefs = [
    { key: 'accuracy',  name: 'Accuracy',  formula: '(TP+TN) / Total' },
    { key: 'precision', name: 'Precision', formula: 'TP / (TP+FP)' },
    { key: 'recall',    name: 'Recall',    formula: 'TP / (TP+FN)' },
    { key: 'f1',        name: 'F1 Score',  formula: '2(P\u00b7R) / (P+R)' },
  ];

  const metricEls = {};
  metricDefs.forEach((md) => {
    const row = document.createElement('div');
    row.className = 'cm-metric-row';

    const nameEl = document.createElement('span');
    nameEl.className = 'cm-metric-name';
    nameEl.textContent = md.name;

    const barTrack = document.createElement('div');
    barTrack.className = 'cm-metric-bar-track';

    const barFill = document.createElement('div');
    barFill.className = 'cm-metric-bar-fill';
    barTrack.appendChild(barFill);

    const valueEl = document.createElement('span');
    valueEl.className = 'cm-metric-value';

    row.appendChild(nameEl);
    row.appendChild(barTrack);
    row.appendChild(valueEl);
    metricsSection.appendChild(row);

    metricEls[md.key] = { barFill, valueEl };
  });

  // Formula reference
  const formulaEl = document.createElement('div');
  formulaEl.className = 'cm-formula';
  formulaEl.innerHTML =
    'Accuracy = (TP+TN)/Total &nbsp;&bull;&nbsp; Precision = TP/(TP+FP) &nbsp;&bull;&nbsp; Recall = TP/(TP+FN)' +
    '<br>F1 = 2 &middot; (Precision &middot; Recall) / (Precision + Recall)';

  // --- Scenario change handler ---
  scenarioSelect.addEventListener('change', () => {
    const name = scenarioSelect.value;
    if (name === '__custom__') return;
    const preset = SCENARIOS[name];
    if (preset) {
      state = { ...preset };
      render();
    }
  });

  // --- Render function ---
  function render() {
    // Update cell counts
    for (const cellDef of CELLS) {
      cellEls[cellDef.key].textContent = state[cellDef.key];
    }

    // Compute and update metrics
    const m = computeMetrics(state);
    for (const md of metricDefs) {
      const val = m[md.key];
      const { barFill, valueEl } = metricEls[md.key];
      const pct = (val * 100).toFixed(1);
      const color = metricBarColor(val);

      barFill.style.width = `${pct}%`;
      barFill.style.background = color;
      valueEl.textContent = `${pct}%`;
      valueEl.style.color = color;
    }
  }

  // --- Assemble ---
  container.innerHTML = '';
  container.appendChild(styleEl);
  widget.appendChild(header);
  widget.appendChild(scenarioRow);
  widget.appendChild(gridWrapper);
  widget.appendChild(metricsSection);
  widget.appendChild(formulaEl);
  container.appendChild(widget);

  // Initial render
  render();
}
