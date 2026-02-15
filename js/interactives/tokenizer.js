// Tokenizer Explorer — interactive BPE-like tokenization demo
// Users type text and see it broken into colored token spans in real-time

const DEFAULT_TEXT = 'The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms.';

// Muted color palette for token backgrounds
const TOKEN_COLORS = [
  'rgba(0, 212, 170, 0.18)',   // accent-primary
  'rgba(59, 130, 246, 0.18)',  // accent-blue
  'rgba(245, 158, 11, 0.18)', // accent-warm
  'rgba(168, 85, 247, 0.18)', // accent-purple
  'rgba(239, 68, 68, 0.18)',  // red
  'rgba(34, 197, 94, 0.18)',  // green
  'rgba(14, 165, 233, 0.18)', // sky
  'rgba(251, 191, 36, 0.18)', // amber
];

const TOKEN_BORDER_COLORS = [
  'rgba(0, 212, 170, 0.45)',
  'rgba(59, 130, 246, 0.45)',
  'rgba(245, 158, 11, 0.45)',
  'rgba(168, 85, 247, 0.45)',
  'rgba(239, 68, 68, 0.45)',
  'rgba(34, 197, 94, 0.45)',
  'rgba(14, 165, 233, 0.45)',
  'rgba(251, 191, 36, 0.45)',
];

// Common English prefixes and suffixes for BPE-like splitting
const SUFFIXES = ['tion', 'sion', 'ment', 'ness', 'able', 'ible', 'ized', 'ling', 'ous', 'ful', 'less', 'ing', 'ely', 'ally', 'ive', 'ed', 'ly', 'er', 'es', 'al', 'en'];
const PREFIXES = ['trans', 'intro', 'self', 'over', 'under', 'multi', 'pre', 'mis', 'dis', 'un', 're'];

/**
 * Simplified BPE-like tokenizer.
 * Splits text on whitespace and punctuation boundaries, then further
 * decomposes words by known prefixes/suffixes and subword chunks.
 */
function tokenize(text) {
  if (!text) return [];

  const tokens = [];

  // Step 1: Split into coarse segments — words, whitespace runs, and punctuation
  const segments = text.match(/\s+|[a-zA-Z0-9']+|[^\s\w]/g) || [];

  for (const segment of segments) {
    // Whitespace preserved as its own token
    if (/^\s+$/.test(segment)) {
      tokens.push(segment);
      continue;
    }

    // Punctuation stays atomic
    if (/^[^\w']$/.test(segment)) {
      tokens.push(segment);
      continue;
    }

    // Word — attempt BPE-like decomposition
    const subTokens = decomposeWord(segment);
    tokens.push(...subTokens);
  }

  return tokens;
}

/**
 * Decompose a single word into subword pieces using prefix/suffix
 * stripping and fixed-length chunking for the remainder.
 */
function decomposeWord(word) {
  if (word.length <= 3) return [word];

  const pieces = [];
  let remaining = word;

  // Try to peel off a known prefix
  let prefixFound = '';
  for (const pfx of PREFIXES) {
    if (remaining.toLowerCase().startsWith(pfx) && remaining.length > pfx.length + 1) {
      prefixFound = remaining.slice(0, pfx.length);
      remaining = remaining.slice(pfx.length);
      break;
    }
  }

  // Try to peel off a known suffix
  let suffixFound = '';
  for (const sfx of SUFFIXES) {
    if (remaining.toLowerCase().endsWith(sfx) && remaining.length > sfx.length + 1) {
      suffixFound = remaining.slice(remaining.length - sfx.length);
      remaining = remaining.slice(0, remaining.length - sfx.length);
      break;
    }
  }

  // Assemble: prefix + stem chunks + suffix
  if (prefixFound) pieces.push(prefixFound);

  // If the stem is still long, chunk it into 3-4 char subwords
  if (remaining.length > 5) {
    let i = 0;
    while (i < remaining.length) {
      const chunkSize = Math.min(remaining.length - i <= 5 ? remaining.length - i : 4, remaining.length - i);
      pieces.push(remaining.slice(i, i + chunkSize));
      i += chunkSize;
    }
  } else {
    pieces.push(remaining);
  }

  if (suffixFound) pieces.push(suffixFound);

  return pieces;
}

export function mount(container) {
  container.innerHTML = `
    <style>
      .tkz-widget {
        background: var(--bg-elevated, #1e2740);
        border: 1px solid var(--border-subtle, #2d3748);
        border-radius: var(--radius-md, 10px);
        padding: var(--space-6, 24px);
        font-family: var(--font-body, system-ui, sans-serif);
      }

      .tkz-header {
        display: flex;
        align-items: center;
        gap: var(--space-3, 12px);
        margin-bottom: var(--space-2, 8px);
      }

      .tkz-header-icon {
        width: 32px;
        height: 32px;
        border-radius: var(--radius-sm, 6px);
        background: rgba(0, 212, 170, 0.12);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
      }

      .tkz-header-icon svg {
        width: 18px;
        height: 18px;
      }

      .tkz-title {
        font-family: var(--font-heading, system-ui, sans-serif);
        font-size: var(--text-base, 16px);
        font-weight: 700;
        color: var(--text-primary, #f1f5f9);
        margin: 0;
      }

      .tkz-subtitle {
        font-size: var(--text-sm, 14px);
        color: var(--text-dim, #64748b);
        margin: 0 0 var(--space-5, 20px) 0;
        padding-left: 44px;
      }

      .tkz-input-label {
        display: block;
        font-size: var(--text-xs, 12px);
        font-family: var(--font-mono, monospace);
        color: var(--text-secondary, #94a3b8);
        margin-bottom: var(--space-2, 8px);
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }

      .tkz-textarea {
        width: 100%;
        min-height: 80px;
        background: var(--bg-surface, #111827);
        border: 1px solid var(--border-subtle, #2d3748);
        border-radius: var(--radius-sm, 6px);
        color: var(--text-primary, #f1f5f9);
        font-family: var(--font-body, system-ui, sans-serif);
        font-size: var(--text-sm, 14px);
        padding: var(--space-3, 12px);
        resize: vertical;
        outline: none;
        transition: border-color 0.2s ease;
        box-sizing: border-box;
        line-height: 1.6;
      }

      .tkz-textarea:focus {
        border-color: var(--accent-primary, #00d4aa);
      }

      .tkz-textarea::placeholder {
        color: var(--text-dim, #64748b);
      }

      .tkz-section-label {
        font-size: var(--text-xs, 12px);
        font-family: var(--font-mono, monospace);
        color: var(--text-secondary, #94a3b8);
        margin: var(--space-5, 20px) 0 var(--space-3, 12px) 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }

      .tkz-tokens-area {
        background: var(--bg-surface, #111827);
        border: 1px solid var(--border-subtle, #2d3748);
        border-radius: var(--radius-sm, 6px);
        padding: var(--space-4, 16px);
        min-height: 60px;
        line-height: 2;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 3px;
      }

      .tkz-token {
        display: inline-block;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: var(--font-mono, monospace);
        font-size: var(--text-sm, 14px);
        color: var(--text-primary, #f1f5f9);
        cursor: default;
        position: relative;
        border: 1px solid transparent;
        transition: transform 0.1s ease, border-color 0.15s ease;
        white-space: pre;
      }

      .tkz-token:hover {
        transform: translateY(-1px);
        z-index: 1;
      }

      .tkz-token[data-tooltip]:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: calc(100% + 6px);
        left: 50%;
        transform: translateX(-50%);
        background: var(--bg-elevated, #1e2740);
        color: var(--accent-primary, #00d4aa);
        font-size: 11px;
        font-family: var(--font-mono, monospace);
        padding: 3px 8px;
        border-radius: 4px;
        white-space: nowrap;
        pointer-events: none;
        border: 1px solid var(--border-medium, #374151);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      }

      .tkz-token-ws {
        opacity: 0.35;
        font-size: 10px;
        color: var(--text-dim, #64748b);
        padding: 2px 2px;
      }

      .tkz-stats {
        display: flex;
        gap: var(--space-4, 16px);
        margin-top: var(--space-4, 16px);
        flex-wrap: wrap;
      }

      .tkz-stat {
        background: var(--bg-surface, #111827);
        border: 1px solid var(--border-subtle, #2d3748);
        border-radius: var(--radius-sm, 6px);
        padding: var(--space-3, 12px) var(--space-4, 16px);
        flex: 1;
        min-width: 120px;
        text-align: center;
      }

      .tkz-stat-value {
        font-family: var(--font-mono, monospace);
        font-size: 22px;
        font-weight: 700;
        color: var(--text-primary, #f1f5f9);
        line-height: 1.2;
      }

      .tkz-stat-label {
        font-size: var(--text-xs, 12px);
        color: var(--text-dim, #64748b);
        margin-top: 4px;
      }

      .tkz-stat:nth-child(1) .tkz-stat-value { color: var(--accent-primary, #00d4aa); }
      .tkz-stat:nth-child(2) .tkz-stat-value { color: var(--accent-blue, #3b82f6); }
      .tkz-stat:nth-child(3) .tkz-stat-value { color: var(--accent-warm, #f59e0b); }

      .tkz-empty {
        color: var(--text-dim, #64748b);
        font-size: var(--text-sm, 14px);
        font-style: italic;
      }
    </style>

    <div class="tkz-widget">
      <div class="tkz-header">
        <div class="tkz-header-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="var(--accent-primary, #00d4aa)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M4 7V4h16v3"/>
            <path d="M9 20h6"/>
            <path d="M12 4v16"/>
          </svg>
        </div>
        <h3 class="tkz-title">Tokenizer Explorer</h3>
      </div>
      <p class="tkz-subtitle">Type text to see how LLMs break it into tokens</p>

      <label class="tkz-input-label">Input Text</label>
      <textarea class="tkz-textarea" id="tkz-input" placeholder="Type or paste text here...">${DEFAULT_TEXT}</textarea>

      <div class="tkz-section-label">Tokenized Output</div>
      <div class="tkz-tokens-area" id="tkz-output"></div>

      <div class="tkz-stats">
        <div class="tkz-stat">
          <div class="tkz-stat-value" id="tkz-token-count">0</div>
          <div class="tkz-stat-label">Tokens</div>
        </div>
        <div class="tkz-stat">
          <div class="tkz-stat-value" id="tkz-char-count">0</div>
          <div class="tkz-stat-label">Characters</div>
        </div>
        <div class="tkz-stat">
          <div class="tkz-stat-value" id="tkz-ratio">0</div>
          <div class="tkz-stat-label">Chars / Token</div>
        </div>
      </div>
    </div>
  `;

  const inputEl = container.querySelector('#tkz-input');
  const outputEl = container.querySelector('#tkz-output');
  const tokenCountEl = container.querySelector('#tkz-token-count');
  const charCountEl = container.querySelector('#tkz-char-count');
  const ratioEl = container.querySelector('#tkz-ratio');

  function update() {
    const text = inputEl.value;

    if (!text.trim()) {
      outputEl.innerHTML = '<span class="tkz-empty">Start typing to see tokens...</span>';
      tokenCountEl.textContent = '0';
      charCountEl.textContent = '0';
      ratioEl.textContent = '—';
      return;
    }

    const tokens = tokenize(text);

    // Count non-whitespace tokens for display stats
    const meaningfulTokens = tokens.filter(t => !/^\s+$/.test(t));
    const tokenCount = meaningfulTokens.length;
    const charCount = text.length;
    const ratio = tokenCount > 0 ? (charCount / tokenCount).toFixed(1) : '—';

    // Build token spans
    let colorIdx = 0;
    const spans = tokens.map((token, i) => {
      const isWhitespace = /^\s+$/.test(token);

      if (isWhitespace) {
        // Render whitespace as a subtle marker
        const display = token.replace(/ /g, '\u00B7').replace(/\n/g, '\u21B5').replace(/\t/g, '\u2192');
        return `<span class="tkz-token tkz-token-ws" data-tooltip="token ${i}">${display}</span>`;
      }

      const bg = TOKEN_COLORS[colorIdx % TOKEN_COLORS.length];
      const border = TOKEN_BORDER_COLORS[colorIdx % TOKEN_BORDER_COLORS.length];
      colorIdx++;

      // Escape HTML entities in the token text
      const escaped = token.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

      return `<span class="tkz-token" style="background:${bg}; border-color:${border};" data-tooltip="token ${i}">${escaped}</span>`;
    }).join('');

    outputEl.innerHTML = spans;
    tokenCountEl.textContent = tokenCount;
    charCountEl.textContent = charCount;
    ratioEl.textContent = ratio;
  }

  // Attach listener and run initial tokenization
  inputEl.addEventListener('input', update);
  update();
}
