// Prompt Comparison Tool — interactive side-by-side comparison of prompting strategies
// Shows zero-shot, few-shot, and chain-of-thought prompting applied to the same question

const DEFAULT_QUESTION = 'What is the capital of the country that hosted the 2024 Olympics?';

const STRATEGIES = [
  {
    id: 'zero-shot',
    label: 'Zero-Shot',
    description: 'Direct question with no examples or guidance',
    buildPrompt(question) {
      return [
        { role: 'system', text: 'You are a helpful assistant.' },
        { role: 'user', text: question },
      ];
    },
    response: 'Paris.',
    ratings: { accuracy: 5, reasoning: 1, tokenCost: 1 },
  },
  {
    id: 'few-shot',
    label: 'Few-Shot',
    description: 'Question preceded by worked examples',
    buildPrompt(question) {
      return [
        { role: 'system', text: 'You are a helpful assistant. Answer questions about geography by identifying the country first, then its capital.' },
        { role: 'example', text: 'Q: What is the capital of the country that invented sushi?\nA: Sushi was invented in Japan. The capital of Japan is Tokyo.' },
        { role: 'example', text: 'Q: What is the capital of the country where the Colosseum is located?\nA: The Colosseum is located in Italy. The capital of Italy is Rome.' },
        { role: 'user', text: question },
      ];
    },
    response: 'The 2024 Olympics were hosted by France. The capital of France is Paris.',
    ratings: { accuracy: 5, reasoning: 3, tokenCost: 3 },
  },
  {
    id: 'chain-of-thought',
    label: 'Chain-of-Thought',
    description: 'Explicit step-by-step reasoning instruction',
    buildPrompt(question) {
      return [
        { role: 'system', text: 'You are a helpful assistant. When answering, think through the problem step by step before giving your final answer.' },
        { role: 'user', text: question + "\n\nLet's think step by step." },
      ];
    },
    response: 'Step 1: The 2024 Olympics were hosted in Paris, France.\nStep 2: The country is France.\nStep 3: The capital of France is Paris.\n\nTherefore, the answer is Paris.',
    ratings: { accuracy: 5, reasoning: 5, tokenCost: 4 },
  },
];

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function renderPromptBlock(parts) {
  return parts.map(part => {
    const escaped = escapeHtml(part.text);
    let roleLabel, roleClass;
    switch (part.role) {
      case 'system':
        roleLabel = 'SYSTEM';
        roleClass = 'pc-role-system';
        break;
      case 'user':
        roleLabel = 'USER';
        roleClass = 'pc-role-user';
        break;
      case 'example':
        roleLabel = 'EXAMPLE';
        roleClass = 'pc-role-example';
        break;
      default:
        roleLabel = part.role.toUpperCase();
        roleClass = '';
    }
    return `<div class="pc-prompt-part ${roleClass}"><span class="pc-role-badge ${roleClass}">${roleLabel}</span><pre class="pc-prompt-text">${escaped}</pre></div>`;
  }).join('');
}

function renderRatingBar(label, value, maxVal) {
  const pips = [];
  for (let i = 1; i <= maxVal; i++) {
    pips.push(`<span class="pc-pip ${i <= value ? 'pc-pip-filled' : ''}"></span>`);
  }
  return `<div class="pc-rating-row"><span class="pc-rating-label">${label}</span><div class="pc-pips">${pips.join('')}</div><span class="pc-rating-value">${value}/${maxVal}</span></div>`;
}

export function mount(container) {
  container.innerHTML = `
    <style>
      .pc-widget {
        background: var(--bg-elevated, #1e2740);
        border: 1px solid var(--border-subtle, #2d3748);
        border-radius: var(--radius-md, 10px);
        padding: var(--space-6, 24px);
        font-family: var(--font-body, system-ui, sans-serif);
      }

      .pc-header {
        display: flex;
        align-items: center;
        gap: var(--space-3, 12px);
        margin-bottom: var(--space-2, 8px);
      }

      .pc-header-icon {
        width: 32px;
        height: 32px;
        border-radius: var(--radius-sm, 6px);
        background: rgba(168, 85, 247, 0.12);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
      }

      .pc-header-icon svg {
        width: 18px;
        height: 18px;
      }

      .pc-title {
        font-family: var(--font-heading, system-ui, sans-serif);
        font-size: var(--text-base, 16px);
        font-weight: 700;
        color: var(--text-primary, #f1f5f9);
        margin: 0;
      }

      .pc-subtitle {
        font-size: var(--text-sm, 14px);
        color: var(--text-dim, #64748b);
        margin: 0 0 var(--space-5, 20px) 0;
        padding-left: 44px;
      }

      /* Question section */
      .pc-question-section {
        background: var(--bg-surface, #111827);
        border: 1px solid var(--border-subtle, #2d3748);
        border-radius: var(--radius-sm, 6px);
        padding: var(--space-4, 16px);
        margin-bottom: var(--space-5, 20px);
      }

      .pc-question-label {
        font-size: var(--text-xs, 12px);
        font-family: var(--font-mono, monospace);
        color: var(--text-secondary, #94a3b8);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0 0 var(--space-2, 8px) 0;
      }

      .pc-question-display {
        font-size: var(--text-base, 16px);
        color: var(--text-primary, #f1f5f9);
        font-weight: 600;
        margin: 0 0 var(--space-3, 12px) 0;
        line-height: 1.5;
      }

      .pc-question-input {
        width: 100%;
        background: var(--bg-elevated, #1e2740);
        border: 1px solid var(--border-subtle, #2d3748);
        border-radius: var(--radius-sm, 6px);
        color: var(--text-primary, #f1f5f9);
        font-family: var(--font-body, system-ui, sans-serif);
        font-size: var(--text-sm, 14px);
        padding: var(--space-2, 8px) var(--space-3, 12px);
        outline: none;
        transition: border-color 0.2s ease;
        box-sizing: border-box;
      }

      .pc-question-input:focus {
        border-color: var(--accent-primary, #00d4aa);
      }

      .pc-question-input::placeholder {
        color: var(--text-dim, #64748b);
      }

      .pc-input-hint {
        font-size: var(--text-xs, 12px);
        color: var(--text-dim, #64748b);
        margin-top: var(--space-2, 8px);
      }

      /* Tabs */
      .pc-tabs {
        display: flex;
        gap: var(--space-2, 8px);
        margin-bottom: var(--space-4, 16px);
        flex-wrap: wrap;
      }

      .pc-tab {
        flex: 1;
        min-width: 140px;
        background: var(--bg-surface, #111827);
        border: 2px solid var(--border-subtle, #2d3748);
        border-radius: var(--radius-sm, 6px);
        padding: var(--space-3, 12px) var(--space-4, 16px);
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: center;
      }

      .pc-tab:hover {
        border-color: var(--border-medium, #374151);
      }

      .pc-tab-label {
        font-family: var(--font-heading, system-ui, sans-serif);
        font-size: var(--text-sm, 14px);
        font-weight: 700;
        color: var(--text-secondary, #94a3b8);
        margin: 0;
        transition: color 0.2s ease;
      }

      .pc-tab-desc {
        font-size: var(--text-xs, 12px);
        color: var(--text-dim, #64748b);
        margin: var(--space-1, 4px) 0 0 0;
      }

      .pc-tab--active {
        border-color: var(--accent-primary, #00d4aa);
        background: rgba(0, 212, 170, 0.06);
      }

      .pc-tab--active .pc-tab-label {
        color: var(--accent-primary, #00d4aa);
      }

      .pc-tab:not(.pc-tab--active) {
        opacity: 0.65;
      }

      .pc-tab:not(.pc-tab--active):hover {
        opacity: 0.85;
      }

      /* Content panel */
      .pc-panel {
        display: none;
      }

      .pc-panel--active {
        display: block;
      }

      .pc-section-label {
        font-size: var(--text-xs, 12px);
        font-family: var(--font-mono, monospace);
        color: var(--text-secondary, #94a3b8);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0 0 var(--space-2, 8px) 0;
      }

      /* Prompt block */
      .pc-prompt-block {
        background: var(--bg-surface, #111827);
        border: 1px solid var(--border-subtle, #2d3748);
        border-radius: var(--radius-sm, 6px);
        padding: var(--space-4, 16px);
        margin-bottom: var(--space-4, 16px);
      }

      .pc-prompt-part {
        margin-bottom: var(--space-3, 12px);
      }

      .pc-prompt-part:last-child {
        margin-bottom: 0;
      }

      .pc-role-badge {
        display: inline-block;
        font-size: 10px;
        font-family: var(--font-mono, monospace);
        font-weight: 700;
        letter-spacing: 0.08em;
        padding: 2px 8px;
        border-radius: 3px;
        margin-bottom: var(--space-2, 8px);
        text-transform: uppercase;
      }

      .pc-role-system .pc-role-badge,
      .pc-role-badge.pc-role-system {
        background: rgba(168, 85, 247, 0.15);
        color: var(--accent-purple, #a855f7);
        border: 1px solid rgba(168, 85, 247, 0.3);
      }

      .pc-role-user .pc-role-badge,
      .pc-role-badge.pc-role-user {
        background: rgba(59, 130, 246, 0.15);
        color: var(--accent-blue, #3b82f6);
        border: 1px solid rgba(59, 130, 246, 0.3);
      }

      .pc-role-example .pc-role-badge,
      .pc-role-badge.pc-role-example {
        background: rgba(245, 158, 11, 0.15);
        color: var(--accent-warm, #f59e0b);
        border: 1px solid rgba(245, 158, 11, 0.3);
      }

      .pc-prompt-text {
        font-family: var(--font-mono, monospace);
        font-size: var(--text-sm, 14px);
        color: var(--text-primary, #f1f5f9);
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        line-height: 1.6;
        padding: var(--space-2, 8px) var(--space-3, 12px);
        border-radius: 4px;
      }

      .pc-role-system .pc-prompt-text {
        background: rgba(168, 85, 247, 0.06);
        border-left: 3px solid rgba(168, 85, 247, 0.4);
      }

      .pc-role-user .pc-prompt-text {
        background: rgba(59, 130, 246, 0.06);
        border-left: 3px solid rgba(59, 130, 246, 0.4);
      }

      .pc-role-example .pc-prompt-text {
        background: rgba(245, 158, 11, 0.06);
        border-left: 3px solid rgba(245, 158, 11, 0.4);
      }

      /* Response block */
      .pc-response-block {
        background: var(--bg-surface, #111827);
        border: 1px solid var(--border-subtle, #2d3748);
        border-left: 3px solid var(--accent-primary, #00d4aa);
        border-radius: var(--radius-sm, 6px);
        padding: var(--space-4, 16px);
        margin-bottom: var(--space-4, 16px);
      }

      .pc-response-badge {
        display: inline-block;
        font-size: 10px;
        font-family: var(--font-mono, monospace);
        font-weight: 700;
        letter-spacing: 0.08em;
        padding: 2px 8px;
        border-radius: 3px;
        margin-bottom: var(--space-2, 8px);
        text-transform: uppercase;
        background: rgba(0, 212, 170, 0.15);
        color: var(--accent-primary, #00d4aa);
        border: 1px solid rgba(0, 212, 170, 0.3);
      }

      .pc-response-text {
        font-family: var(--font-body, system-ui, sans-serif);
        font-size: var(--text-sm, 14px);
        color: var(--text-primary, #f1f5f9);
        line-height: 1.7;
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
      }

      .pc-response-note {
        font-size: var(--text-xs, 12px);
        color: var(--text-dim, #64748b);
        font-style: italic;
        margin-top: var(--space-3, 12px);
      }

      /* Quality assessment */
      .pc-assessment {
        background: var(--bg-surface, #111827);
        border: 1px solid var(--border-subtle, #2d3748);
        border-radius: var(--radius-sm, 6px);
        padding: var(--space-4, 16px);
      }

      .pc-rating-row {
        display: flex;
        align-items: center;
        gap: var(--space-3, 12px);
        margin-bottom: var(--space-2, 8px);
      }

      .pc-rating-row:last-child {
        margin-bottom: 0;
      }

      .pc-rating-label {
        font-size: var(--text-xs, 12px);
        font-family: var(--font-mono, monospace);
        color: var(--text-secondary, #94a3b8);
        min-width: 120px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
      }

      .pc-pips {
        display: flex;
        gap: 4px;
        flex: 1;
      }

      .pc-pip {
        width: 24px;
        height: 8px;
        border-radius: 2px;
        background: var(--border-subtle, #2d3748);
        transition: background 0.2s ease;
      }

      .pc-pip-filled {
        background: var(--accent-primary, #00d4aa);
      }

      /* Different colors for different rows */
      .pc-rating-row:nth-child(1) .pc-pip-filled {
        background: var(--accent-primary, #00d4aa);
      }

      .pc-rating-row:nth-child(2) .pc-pip-filled {
        background: var(--accent-blue, #3b82f6);
      }

      .pc-rating-row:nth-child(3) .pc-pip-filled {
        background: var(--accent-warm, #f59e0b);
      }

      .pc-rating-value {
        font-size: var(--text-xs, 12px);
        font-family: var(--font-mono, monospace);
        color: var(--text-dim, #64748b);
        min-width: 28px;
        text-align: right;
      }
    </style>

    <div class="pc-widget">
      <div class="pc-header">
        <div class="pc-header-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="var(--accent-purple, #a855f7)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="16 18 22 12 16 6"></polyline>
            <polyline points="8 6 2 12 8 18"></polyline>
            <line x1="12" y1="2" x2="12" y2="22" opacity="0.4"></line>
          </svg>
        </div>
        <h3 class="pc-title">Prompt Engineering Lab</h3>
      </div>
      <p class="pc-subtitle">Compare zero-shot, few-shot, and chain-of-thought prompting</p>

      <div class="pc-question-section">
        <p class="pc-question-label">Question</p>
        <p class="pc-question-display" id="pc-question-text">${escapeHtml(DEFAULT_QUESTION)}</p>
        <input
          type="text"
          class="pc-question-input"
          id="pc-custom-input"
          placeholder="Try your own question..."
        />
        <p class="pc-input-hint">Type a custom question to see how each prompt template formats it (responses remain pre-written for the default question).</p>
      </div>

      <div class="pc-tabs" id="pc-tabs"></div>
      <div id="pc-panels"></div>
    </div>
  `;

  const tabsContainer = container.querySelector('#pc-tabs');
  const panelsContainer = container.querySelector('#pc-panels');
  const customInput = container.querySelector('#pc-custom-input');
  const questionDisplay = container.querySelector('#pc-question-text');

  let activeTab = 'zero-shot';
  let currentQuestion = DEFAULT_QUESTION;

  function render() {
    // Render tabs
    tabsContainer.innerHTML = STRATEGIES.map(s => {
      const isActive = s.id === activeTab;
      return `<button class="pc-tab ${isActive ? 'pc-tab--active' : ''}" data-tab="${s.id}">
        <p class="pc-tab-label">${escapeHtml(s.label)}</p>
        <p class="pc-tab-desc">${escapeHtml(s.description)}</p>
      </button>`;
    }).join('');

    // Render panels
    const isCustom = currentQuestion !== DEFAULT_QUESTION;

    panelsContainer.innerHTML = STRATEGIES.map(s => {
      const isActive = s.id === activeTab;
      const promptParts = s.buildPrompt(currentQuestion);
      const promptHtml = renderPromptBlock(promptParts);

      const responseContent = isCustom
        ? `<p class="pc-response-note">Enter the default question to see the pre-written model response, or observe how your question is formatted within each prompt template above.</p>`
        : `<pre class="pc-response-text">${escapeHtml(s.response)}</pre>`;

      const ratingsHtml = [
        renderRatingBar('Accuracy', s.ratings.accuracy, 5),
        renderRatingBar('Reasoning Depth', s.ratings.reasoning, 5),
        renderRatingBar('Token Cost', s.ratings.tokenCost, 5),
      ].join('');

      return `<div class="pc-panel ${isActive ? 'pc-panel--active' : ''}" data-panel="${s.id}">
        <p class="pc-section-label">Prompt</p>
        <div class="pc-prompt-block">${promptHtml}</div>

        <p class="pc-section-label">Model Response</p>
        <div class="pc-response-block">
          <span class="pc-response-badge">Assistant</span>
          ${responseContent}
        </div>

        <p class="pc-section-label">Quality Assessment</p>
        <div class="pc-assessment">${ratingsHtml}</div>
      </div>`;
    }).join('');

    // Attach tab click handlers
    tabsContainer.querySelectorAll('.pc-tab').forEach(btn => {
      btn.addEventListener('click', () => {
        activeTab = btn.dataset.tab;
        render();
      });
    });
  }

  // Custom question input handler
  customInput.addEventListener('input', () => {
    const val = customInput.value.trim();
    currentQuestion = val || DEFAULT_QUESTION;
    questionDisplay.textContent = currentQuestion;
    render();
  });

  // Initial render
  render();
}
