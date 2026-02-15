export function render(container) {
  // Deterministic attention matrix: "The cat sat on mat"
  const tokens = ['The', 'cat', 'sat', 'on', 'mat'];
  const attnMatrix = [
    [0.80, 0.05, 0.05, 0.05, 0.05],
    [0.10, 0.70, 0.05, 0.05, 0.10],
    [0.05, 0.60, 0.20, 0.05, 0.10],
    [0.05, 0.05, 0.30, 0.50, 0.10],
    [0.10, 0.05, 0.05, 0.10, 0.70],
  ];

  // --- Heatmap cells ---
  const cellSize = 26;
  const cellGap = 3;
  const heatmapOriginX = 530;
  const heatmapOriginY = 195;

  let heatmapCells = '';
  let heatmapLabelsTop = '';
  let heatmapLabelsLeft = '';

  for (let r = 0; r < 5; r++) {
    // Row labels (left side — keys)
    heatmapLabelsLeft += `<text
      x="${heatmapOriginX - 8}"
      y="${heatmapOriginY + r * (cellSize + cellGap) + cellSize / 2 + 4}"
      text-anchor="end"
      class="tf-token-label tf-fadein"
      style="animation-delay: ${(1.6 + r * 0.06).toFixed(2)}s"
    >${tokens[r]}</text>`;

    // Column labels (top — queries)
    heatmapLabelsTop += `<text
      x="${heatmapOriginX + r * (cellSize + cellGap) + cellSize / 2}"
      y="${heatmapOriginY - 10}"
      text-anchor="middle"
      class="tf-token-label tf-fadein"
      style="animation-delay: ${(1.6 + r * 0.06).toFixed(2)}s"
    >${tokens[r]}</text>`;

    for (let c = 0; c < 5; c++) {
      const val = attnMatrix[r][c];
      const x = heatmapOriginX + c * (cellSize + cellGap);
      const y = heatmapOriginY + r * (cellSize + cellGap);
      const delay = (1.7 + r * 0.08 + c * 0.04).toFixed(2);
      // Interpolate color: low = dark surface, high = bright amber
      const opacity = (0.15 + val * 0.85).toFixed(2);
      heatmapCells += `
        <rect x="${x}" y="${y}" width="${cellSize}" height="${cellSize}" rx="4"
          fill="var(--accent-primary)" opacity="0"
          class="tf-fadein" style="animation-delay: ${delay}s">
          <animate attributeName="opacity"
            values="0;${opacity};${(opacity * 0.7).toFixed(2)};${opacity}"
            keyTimes="0;0.3;0.65;1"
            dur="4s" begin="${delay}s" fill="freeze" repeatCount="indefinite"/>
        </rect>
        ${val >= 0.3 ? `<text x="${x + cellSize / 2}" y="${y + cellSize / 2 + 4}"
          text-anchor="middle" fill="var(--bg-deep)"
          font-family="var(--font-mono)" font-size="8" font-weight="700"
          class="tf-fadein" style="animation-delay: ${(parseFloat(delay) + 0.15).toFixed(2)}s"
        >${val.toFixed(1)}</text>` : ''}`;
    }
  }

  // --- Architecture component definitions ---
  // Main column center X = 220, width = 220
  const colX = 110;
  const colW = 220;
  const colCx = colX + colW / 2; // 220

  // Component heights and Y positions (bottom to top)
  const compH = 38;
  const gap = 10;

  // Y positions (bottom-to-top layout)
  const yInput = 510;
  const yEmbed = 440;
  const yAttn = 340;
  const yAddNorm1 = 275;
  const yFFN = 210;
  const yAddNorm2 = 150;
  const yOutput = 80;

  // --- Arrow paths ---
  function verticalArrow(fromY, toY, cx = colCx) {
    const midY_top = toY + compH;
    return `<line x1="${cx}" y1="${fromY}" x2="${cx}" y2="${midY_top + 5}"
      stroke="var(--border-strong)" stroke-width="1.5" class="tf-arrow" marker-end="url(#arrowHead)"/>`;
  }

  // Data flow particle along a vertical path
  function flowParticle(fromY, toY, cx, delay, color = 'var(--accent-primary)') {
    const topY = toY + compH + 5;
    return `<circle r="2.5" fill="${color}" opacity="0.9">
      <animateMotion dur="2s" begin="${delay}s" repeatCount="indefinite"
        path="M ${cx},${fromY} L ${cx},${topY}" fill="freeze"/>
      <animate attributeName="opacity" values="0;0.9;0.9;0" dur="2s" begin="${delay}s" repeatCount="indefinite"/>
    </circle>`;
  }

  // --- Residual connection skip arrows ---
  // Skip arrow from before Attention to Add&Norm1 (goes around the right side)
  const skipPath1 = `M ${colX + colW + 8},${yEmbed}
    L ${colX + colW + 30},${yEmbed}
    L ${colX + colW + 30},${yAddNorm1 + compH / 2}
    L ${colX + colW + 8},${yAddNorm1 + compH / 2}`;

  // Skip arrow from before FFN to Add&Norm2 (goes around the right side)
  const skipPath2 = `M ${colX + colW + 8},${yAddNorm1}
    L ${colX + colW + 45},${yAddNorm1}
    L ${colX + colW + 45},${yAddNorm2 + compH / 2}
    L ${colX + colW + 8},${yAddNorm2 + compH / 2}`;

  container.innerHTML = `
    <div style="text-align: center; width: 100%;">
      <svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg"
        style="max-width: 100%; height: auto; overflow: visible;"
        role="img" aria-label="Transformer Block Architecture diagram showing self-attention flow">
        <defs>
          <!-- Arrow marker -->
          <marker id="arrowHead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="var(--border-strong)"/>
          </marker>
          <marker id="arrowHeadAmber" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="var(--accent-primary)"/>
          </marker>
          <marker id="arrowHeadWarm" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
            <polygon points="0 0, 7 2.5, 0 5" fill="var(--accent-warm)"/>
          </marker>

          <!-- Glow filter for key elements -->
          <filter id="glowAmber" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur in="SourceGraphic" stdDeviation="3" result="blur"/>
            <feComposite in="SourceGraphic" in2="blur" operator="over"/>
          </filter>
          <filter id="glowSoft" x="-10%" y="-10%" width="120%" height="120%">
            <feGaussianBlur in="SourceAlpha" stdDeviation="4" result="blur"/>
            <feFlood flood-color="var(--accent-primary)" flood-opacity="0.15" result="color"/>
            <feComposite in="color" in2="blur" operator="in" result="glow"/>
            <feComposite in="SourceGraphic" in2="glow" operator="over"/>
          </filter>

          <!-- Gradient for main attention block -->
          <linearGradient id="attnGrad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="#F0B42918"/>
            <stop offset="100%" stop-color="#E8553A10"/>
          </linearGradient>
          <linearGradient id="ffnGrad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="#7EB8DA15"/>
            <stop offset="100%" stop-color="#C4A7E710"/>
          </linearGradient>
        </defs>

        <style>
          .tf-fadein {
            opacity: 0;
            animation: tfFadeSlideUp 0.65s var(--ease-out, cubic-bezier(0.16, 1, 0.3, 1)) forwards;
          }
          .tf-arrow {
            stroke-dasharray: 6 4;
            animation: tfDashFlow 1.2s linear infinite;
          }
          .tf-skip-arrow {
            stroke-dasharray: 5 3;
            animation: tfDashFlow 1.8s linear infinite;
          }
          .tf-label {
            font-family: var(--font-mono);
            font-size: 9px;
            fill: var(--text-dim);
            letter-spacing: 0.03em;
          }
          .tf-title {
            font-family: var(--font-heading);
            font-size: 12.5px;
            fill: var(--text-primary);
            font-weight: 600;
          }
          .tf-subtitle {
            font-family: var(--font-mono);
            font-size: 8.5px;
            fill: var(--text-dim);
          }
          .tf-section-title {
            font-family: var(--font-heading);
            font-size: 11px;
            fill: var(--accent-primary);
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
          }
          .tf-token-label {
            font-family: var(--font-mono);
            font-size: 8.5px;
            fill: var(--text-secondary);
          }
          .tf-qkv-badge {
            font-family: var(--font-mono);
            font-size: 9.5px;
            font-weight: 700;
          }
          .tf-bracket {
            fill: none;
            stroke: var(--text-dim);
            stroke-width: 1.2;
            stroke-dasharray: 3 2;
          }
          @keyframes tfFadeSlideUp {
            from { opacity: 0; transform: translateY(14px); }
            to   { opacity: 1; transform: translateY(0); }
          }
          @keyframes tfDashFlow {
            to { stroke-dashoffset: -20; }
          }
        </style>

        <!-- ════════════════════════════════════════ -->
        <!--         LEFT SIDE: ARCHITECTURE          -->
        <!-- ════════════════════════════════════════ -->

        <!-- ─── Input Tokens ─── -->
        <g class="tf-fadein" style="animation-delay: 0.1s">
          <rect x="${colX}" y="${yInput}" width="${colW}" height="${compH}" rx="6"
            fill="var(--bg-elevated)" stroke="var(--border-medium)" stroke-width="1.2"/>
          <text x="${colCx}" y="${yInput + 16}" text-anchor="middle" class="tf-title">Input Tokens</text>
          <text x="${colCx}" y="${yInput + 30}" text-anchor="middle" class="tf-subtitle">
            "The  cat  sat  on  mat"
          </text>
        </g>

        <!-- Arrow: Input -> Embedding -->
        ${verticalArrow(yInput, yEmbed)}

        <!-- ─── Token Embeddings + Positional Encoding ─── -->
        <g class="tf-fadein" style="animation-delay: 0.25s">
          <rect x="${colX}" y="${yEmbed}" width="${colW}" height="${compH}" rx="6"
            fill="var(--bg-elevated)" stroke="var(--accent-blue)" stroke-width="1.3" stroke-opacity="0.5"/>
          <!-- Two sub-blocks inside -->
          <rect x="${colX + 4}" y="${yEmbed + 4}" width="${colW / 2 - 8}" height="${compH - 8}" rx="4"
            fill="var(--accent-blue-dim)" stroke="none"/>
          <text x="${colX + colW / 4}" y="${yEmbed + compH / 2 + 3}" text-anchor="middle"
            class="tf-subtitle" fill="var(--accent-blue)" font-weight="600">Embed</text>

          <rect x="${colX + colW / 2 + 4}" y="${yEmbed + 4}" width="${colW / 2 - 8}" height="${compH - 8}" rx="4"
            fill="var(--accent-purple-dim)" stroke="none"/>
          <text x="${colX + 3 * colW / 4}" y="${yEmbed + compH / 2 + 3}" text-anchor="middle"
            class="tf-subtitle" fill="var(--accent-purple)" font-weight="600">+ Pos Enc</text>
        </g>

        <!-- Arrow: Embedding -> Attention -->
        ${verticalArrow(yEmbed, yAttn)}
        ${flowParticle(yEmbed, yAttn, colCx, 0.8)}

        <!-- ─── N× Bracket ─── -->
        <g class="tf-fadein" style="animation-delay: 0.9s">
          <!-- Bracket line on left side -->
          <path d="M ${colX - 28},${yAttn + compH + 4}
                   L ${colX - 38},${yAttn + compH + 4}
                   L ${colX - 38},${yAddNorm2 - 4}
                   L ${colX - 28},${yAddNorm2 - 4}"
            class="tf-bracket"/>
          <!-- N× label -->
          <text x="${colX - 43}" y="${(yAttn + compH + yAddNorm2) / 2 + 4}"
            text-anchor="middle" font-family="var(--font-heading)" font-size="14"
            fill="var(--accent-warm)" font-weight="700"
            transform="rotate(-90, ${colX - 43}, ${(yAttn + compH + yAddNorm2) / 2 + 4})">N\u00D7</text>
        </g>

        <!-- ─── Multi-Head Self-Attention ─── -->
        <g class="tf-fadein" style="animation-delay: 0.4s">
          <!-- Outer box -->
          <rect x="${colX}" y="${yAttn}" width="${colW}" height="${compH + 20}" rx="8"
            fill="url(#attnGrad)" stroke="var(--accent-primary)" stroke-width="1.5" stroke-opacity="0.6"
            filter="url(#glowSoft)"/>
          <text x="${colCx}" y="${yAttn + 18}" text-anchor="middle" class="tf-title"
            fill="var(--accent-primary)">Multi-Head Self-Attention</text>

          <!-- Q K V badges -->
          <rect x="${colCx - 65}" y="${yAttn + 28}" width="32" height="20" rx="4"
            fill="var(--accent-blue-dim)" stroke="var(--accent-blue)" stroke-width="0.8"/>
          <text x="${colCx - 49}" y="${yAttn + 42}" text-anchor="middle"
            class="tf-qkv-badge" fill="var(--accent-blue)">Q</text>

          <rect x="${colCx - 16}" y="${yAttn + 28}" width="32" height="20" rx="4"
            fill="var(--accent-purple-dim)" stroke="var(--accent-purple)" stroke-width="0.8"/>
          <text x="${colCx}" y="${yAttn + 42}" text-anchor="middle"
            class="tf-qkv-badge" fill="var(--accent-purple)">K</text>

          <rect x="${colCx + 33}" y="${yAttn + 28}" width="32" height="20" rx="4"
            fill="var(--accent-warm-dim)" stroke="var(--accent-warm)" stroke-width="0.8"/>
          <text x="${colCx + 49}" y="${yAttn + 42}" text-anchor="middle"
            class="tf-qkv-badge" fill="var(--accent-warm)">V</text>
        </g>

        <!-- Residual skip connection 1 (around Attention) -->
        <path d="${skipPath1}" fill="none"
          stroke="var(--accent-warm)" stroke-width="1.2" class="tf-skip-arrow tf-fadein"
          marker-end="url(#arrowHeadWarm)" style="animation-delay: 0.7s" stroke-opacity="0.6"/>
        <text x="${colX + colW + 34}" y="${(yEmbed + yAddNorm1 + compH / 2) / 2 + 6}"
          text-anchor="middle" class="tf-fadein"
          style="animation-delay: 0.75s; font-family: var(--font-mono); font-size: 7.5px; fill: var(--accent-warm); opacity: 0.7;"
          transform="rotate(-90, ${colX + colW + 34}, ${(yEmbed + yAddNorm1 + compH / 2) / 2 + 6})">residual</text>

        <!-- Arrow: Attention -> Add&Norm1 -->
        ${verticalArrow(yAttn, yAddNorm1)}
        ${flowParticle(yAttn, yAddNorm1, colCx, 1.4)}

        <!-- ─── Add & Norm 1 ─── -->
        <g class="tf-fadein" style="animation-delay: 0.55s">
          <rect x="${colX}" y="${yAddNorm1}" width="${colW}" height="${compH}" rx="6"
            fill="var(--bg-elevated)" stroke="var(--accent-primary)" stroke-width="1" stroke-opacity="0.35"/>
          <circle cx="${colX + 22}" cy="${yAddNorm1 + compH / 2}" r="9"
            fill="var(--accent-primary)" fill-opacity="0.15" stroke="var(--accent-primary)" stroke-width="0.8"/>
          <text x="${colX + 22}" y="${yAddNorm1 + compH / 2 + 4}" text-anchor="middle"
            fill="var(--accent-primary)" font-size="13" font-weight="700">+</text>
          <text x="${colCx + 14}" y="${yAddNorm1 + compH / 2 + 4}" text-anchor="middle"
            class="tf-title" font-size="11">Add & Layer Norm</text>
        </g>

        <!-- Arrow: Add&Norm1 -> FFN -->
        ${verticalArrow(yAddNorm1, yFFN)}
        ${flowParticle(yAddNorm1, yFFN, colCx, 2.0)}

        <!-- ─── Feed-Forward Network ─── -->
        <g class="tf-fadein" style="animation-delay: 0.65s">
          <rect x="${colX}" y="${yFFN}" width="${colW}" height="${compH + 8}" rx="8"
            fill="url(#ffnGrad)" stroke="var(--accent-blue)" stroke-width="1.3" stroke-opacity="0.5"/>
          <text x="${colCx}" y="${yFFN + 18}" text-anchor="middle" class="tf-title"
            fill="var(--accent-blue)">Feed-Forward Network</text>
          <text x="${colCx}" y="${yFFN + 34}" text-anchor="middle" class="tf-subtitle">
            Linear \u2192 GELU \u2192 Linear
          </text>
        </g>

        <!-- Residual skip connection 2 (around FFN) -->
        <path d="${skipPath2}" fill="none"
          stroke="var(--accent-warm)" stroke-width="1.2" class="tf-skip-arrow tf-fadein"
          marker-end="url(#arrowHeadWarm)" style="animation-delay: 0.85s" stroke-opacity="0.6"/>
        <text x="${colX + colW + 49}" y="${(yAddNorm1 + yAddNorm2 + compH / 2) / 2 + 6}"
          text-anchor="middle" class="tf-fadein"
          style="animation-delay: 0.88s; font-family: var(--font-mono); font-size: 7.5px; fill: var(--accent-warm); opacity: 0.7;"
          transform="rotate(-90, ${colX + colW + 49}, ${(yAddNorm1 + yAddNorm2 + compH / 2) / 2 + 6})">residual</text>

        <!-- Arrow: FFN -> Add&Norm2 -->
        ${verticalArrow(yFFN, yAddNorm2)}
        ${flowParticle(yFFN, yAddNorm2, colCx, 2.6)}

        <!-- ─── Add & Norm 2 ─── -->
        <g class="tf-fadein" style="animation-delay: 0.75s">
          <rect x="${colX}" y="${yAddNorm2}" width="${colW}" height="${compH}" rx="6"
            fill="var(--bg-elevated)" stroke="var(--accent-primary)" stroke-width="1" stroke-opacity="0.35"/>
          <circle cx="${colX + 22}" cy="${yAddNorm2 + compH / 2}" r="9"
            fill="var(--accent-primary)" fill-opacity="0.15" stroke="var(--accent-primary)" stroke-width="0.8"/>
          <text x="${colX + 22}" y="${yAddNorm2 + compH / 2 + 4}" text-anchor="middle"
            fill="var(--accent-primary)" font-size="13" font-weight="700">+</text>
          <text x="${colCx + 14}" y="${yAddNorm2 + compH / 2 + 4}" text-anchor="middle"
            class="tf-title" font-size="11">Add & Layer Norm</text>
        </g>

        <!-- Arrow: Add&Norm2 -> Output -->
        ${verticalArrow(yAddNorm2, yOutput)}
        ${flowParticle(yAddNorm2, yOutput, colCx, 3.2, 'var(--accent-green)')}

        <!-- ─── Output ─── -->
        <g class="tf-fadein" style="animation-delay: 0.85s">
          <rect x="${colX}" y="${yOutput}" width="${colW}" height="${compH + 4}" rx="8"
            fill="var(--bg-elevated)" stroke="var(--accent-green)" stroke-width="1.5" stroke-opacity="0.6"/>
          <text x="${colCx}" y="${yOutput + 17}" text-anchor="middle" class="tf-title"
            fill="var(--accent-green)">Output Representations</text>
          <text x="${colCx}" y="${yOutput + 33}" text-anchor="middle" class="tf-subtitle">
            to next layer or final head
          </text>
        </g>

        <!-- ════════════════════════════════════════ -->
        <!--     RIGHT SIDE: ATTENTION HEATMAP        -->
        <!-- ════════════════════════════════════════ -->

        <!-- Section background -->
        <g class="tf-fadein" style="animation-delay: 1.2s">
          <rect x="478" y="100" width="290" height="310" rx="12"
            fill="var(--bg-surface)" stroke="var(--border-subtle)" stroke-width="1" opacity="0.7"/>
        </g>

        <!-- Heatmap title -->
        <text x="623" y="132" text-anchor="middle"
          class="tf-section-title tf-fadein" style="animation-delay: 1.3s">
          Attention Weights
        </text>
        <text x="623" y="148" text-anchor="middle"
          class="tf-subtitle tf-fadein" style="animation-delay: 1.35s">
          Q\u00B7K\u1D40 / \u221Ad\u2096 \u2192 softmax
        </text>

        <!-- Axis labels -->
        <text x="${heatmapOriginX + 2.5 * (cellSize + cellGap) - cellGap / 2}" y="${heatmapOriginY - 26}"
          text-anchor="middle" class="tf-fadein"
          style="animation-delay: 1.45s; font-family: var(--font-heading); font-size: 8.5px; fill: var(--accent-blue); font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;">
          Query tokens
        </text>
        <text x="${heatmapOriginX - 20}" y="${heatmapOriginY + 2.5 * (cellSize + cellGap) - cellGap / 2 + 4}"
          text-anchor="middle" class="tf-fadein"
          style="animation-delay: 1.45s; font-family: var(--font-heading); font-size: 8.5px; fill: var(--accent-purple); font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;"
          transform="rotate(-90, ${heatmapOriginX - 20}, ${heatmapOriginY + 2.5 * (cellSize + cellGap) - cellGap / 2 + 4})">
          Key tokens
        </text>

        <!-- Token labels -->
        ${heatmapLabelsTop}
        ${heatmapLabelsLeft}

        <!-- Heatmap cells -->
        ${heatmapCells}

        <!-- Heatmap scale legend -->
        <g class="tf-fadein" style="animation-delay: 2.5s">
          <text x="510" y="360" class="tf-subtitle" font-size="7.5">Low</text>
          <rect x="530" y="352" width="12" height="8" rx="2" fill="var(--accent-primary)" opacity="0.15"/>
          <rect x="546" y="352" width="12" height="8" rx="2" fill="var(--accent-primary)" opacity="0.35"/>
          <rect x="562" y="352" width="12" height="8" rx="2" fill="var(--accent-primary)" opacity="0.55"/>
          <rect x="578" y="352" width="12" height="8" rx="2" fill="var(--accent-primary)" opacity="0.75"/>
          <rect x="594" y="352" width="12" height="8" rx="2" fill="var(--accent-primary)" opacity="1.0"/>
          <text x="612" y="360" class="tf-subtitle" font-size="7.5">High</text>
        </g>

        <!-- Insight annotation -->
        <g class="tf-fadein" style="animation-delay: 2.7s">
          <rect x="497" y="375" width="254" height="22" rx="4"
            fill="var(--accent-warm-dim)" stroke="var(--accent-warm)" stroke-width="0.6" stroke-opacity="0.4"/>
          <text x="624" y="390" text-anchor="middle"
            style="font-family: var(--font-body, 'Newsreader'); font-size: 8.5px; fill: var(--accent-warm); font-style: italic;">
            "sat" strongly attends to "cat" (0.6) \u2014 verb\u2192subject link
          </text>
        </g>

        <!-- ════════════════════════════════════════ -->
        <!--       CONNECTING LINE: ARCH -> HEATMAP   -->
        <!-- ════════════════════════════════════════ -->
        <g class="tf-fadein" style="animation-delay: 1.1s">
          <path d="M ${colX + colW},${yAttn + 28}
                   C ${colX + colW + 60},${yAttn + 28}
                     ${heatmapOriginX - 40},${heatmapOriginY + 60}
                     ${heatmapOriginX - 6},${heatmapOriginY + 60}"
            fill="none" stroke="var(--accent-primary)" stroke-width="1"
            stroke-dasharray="4 3" stroke-opacity="0.35"
            marker-end="url(#arrowHeadAmber)"/>
          <text x="${(colX + colW + heatmapOriginX) / 2}" y="${yAttn + 14}"
            text-anchor="middle" style="font-family: var(--font-mono); font-size: 7px; fill: var(--text-dim);">
            attention detail \u2192
          </text>
        </g>

      </svg>
      <p style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-dim); margin-top: var(--space-3); letter-spacing: 0.04em;">
        Transformer Block Architecture \u2014 Self-Attention Flow
      </p>
    </div>
  `;
}
