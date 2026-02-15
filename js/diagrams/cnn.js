export function render(container) {
  const cellSize = 28;
  const gap = 2;

  // Input grid colors (6x6) - represent a simple image pattern
  const inputColors = [
    ['#3D6F8A','#3D6F8A','#5A99B8','#5A99B8','#3D6F8A','#3D6F8A'],
    ['#3D6F8A','#7EB8DA','#7EB8DA','#7EB8DA','#7EB8DA','#3D6F8A'],
    ['#5A99B8','#7EB8DA','#9ECDE6','#9ECDE6','#7EB8DA','#5A99B8'],
    ['#5A99B8','#7EB8DA','#9ECDE6','#BFE0F0','#7EB8DA','#5A99B8'],
    ['#3D6F8A','#7EB8DA','#7EB8DA','#7EB8DA','#7EB8DA','#3D6F8A'],
    ['#3D6F8A','#3D6F8A','#5A99B8','#5A99B8','#3D6F8A','#3D6F8A']
  ];

  // Feature map values (4x4) - lighter = higher activation
  const featureColors = [
    ['#1C1917','#292524','#292524','#1C1917'],
    ['#292524','#C4A7E7','#D4BFF0','#292524'],
    ['#292524','#D4BFF0','#E8DDF5','#292524'],
    ['#1C1917','#292524','#292524','#1C1917']
  ];

  // Pooled map (2x2)
  const pooledColors = [
    ['#C4A7E7', '#D4BFF0'],
    ['#D4BFF0', '#E8DDF5']
  ];

  // Build input grid
  const inputX = 30;
  const inputY = 60;
  let inputGrid = '';
  inputColors.forEach((row, r) => {
    row.forEach((color, c) => {
      inputGrid += `<rect x="${inputX + c * (cellSize + gap)}" y="${inputY + r * (cellSize + gap)}"
        width="${cellSize}" height="${cellSize}" rx="3" fill="${color}" stroke="var(--border-subtle)" stroke-width="0.5"
        class="cnn-cell" style="animation-delay: ${(0.3 + r * 0.05 + c * 0.05).toFixed(2)}s;"/>`;
    });
  });

  // Kernel positions: the kernel slides across 4x4 positions on the 6x6 grid
  // We animate through 16 positions. Each position lasts for a fraction of the cycle.
  const kernelX = inputX;
  const kernelY = inputY;
  const kSize = 3 * (cellSize + gap) - gap;

  // Build kernel animation keyframes for cx,cy positions
  // 16 positions: rows 0-3, cols 0-3
  let xValues = '';
  let yValues = '';
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      const kx = kernelX + c * (cellSize + gap);
      const ky = kernelY + r * (cellSize + gap);
      xValues += kx + ';';
      yValues += ky + ';';
    }
  }
  // Close the loop
  xValues += kernelX;
  yValues += kernelY;

  // Build feature map
  const fmX = 280;
  const fmY = 80;
  let featureGrid = '';
  featureColors.forEach((row, r) => {
    row.forEach((color, c) => {
      const idx = r * 4 + c;
      const delay = (1.5 + idx * 0.25).toFixed(2);
      featureGrid += `<rect x="${fmX + c * (cellSize + gap)}" y="${fmY + r * (cellSize + gap)}"
        width="${cellSize}" height="${cellSize}" rx="3" fill="${color}" stroke="#C4A7E7" stroke-width="0.5"
        class="cnn-feature" style="animation-delay: ${delay}s;"/>`;
    });
  });

  // Build pooled map
  const poolX = 450;
  const poolY = 100;
  const poolCell = 34;
  let pooledGrid = '';
  pooledColors.forEach((row, r) => {
    row.forEach((color, c) => {
      const delay = (6 + r * 0.3 + c * 0.3).toFixed(2);
      pooledGrid += `<rect x="${poolX + c * (poolCell + gap)}" y="${poolY + r * (poolCell + gap)}"
        width="${poolCell}" height="${poolCell}" rx="4" fill="${color}" stroke="#9CCFA4" stroke-width="1"
        class="cnn-feature" style="animation-delay: ${delay}s;"/>`;
    });
  });

  // Connecting arrows
  const arrowY = 140;

  container.innerHTML = `
    <div style="text-align: center; width: 100%;">
      <svg viewBox="0 0 560 310" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
        <style>
          .cnn-cell {
            animation: cnnCellIn 0.3s ease forwards;
            opacity: 0;
          }
          .cnn-feature {
            animation: cnnFeatureIn 0.5s ease forwards;
            opacity: 0;
          }
          .cnn-label {
            animation: cnnFadeIn 0.5s ease forwards;
            opacity: 0;
          }
          .cnn-kernel {
            animation: cnnKernelSlide 8s steps(1) infinite 1s;
          }
          .cnn-arrow {
            stroke-dasharray: 6;
            animation: cnnArrowFlow 1s linear infinite;
          }
          .cnn-pool-arrow {
            stroke-dasharray: 150;
            stroke-dashoffset: 150;
            animation: cnnDrawArrow 0.8s ease forwards 5.5s;
          }
          @keyframes cnnCellIn {
            to { opacity: 1; }
          }
          @keyframes cnnFeatureIn {
            0% { opacity: 0; transform: scale(0.5); }
            100% { opacity: 1; transform: scale(1); }
          }
          @keyframes cnnFadeIn {
            to { opacity: 1; }
          }
          @keyframes cnnArrowFlow {
            to { stroke-dashoffset: -12; }
          }
          @keyframes cnnDrawArrow {
            to { stroke-dashoffset: 0; }
          }
          @keyframes cnnKernelPulse {
            0%, 100% { stroke-opacity: 0.8; }
            50% { stroke-opacity: 1; }
          }
        </style>

        <defs>
          <marker id="cnnArrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#F0B429"/>
          </marker>
          <marker id="cnnArrowTeal" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#9CCFA4"/>
          </marker>
        </defs>

        <!-- Title -->
        <text x="280" y="22" text-anchor="middle" class="cnn-label" style="animation-delay: 0s;"
          fill="var(--text-primary)" font-family="var(--font-heading)" font-size="15" font-weight="700">
          CNN Convolution + Pooling
        </text>

        <!-- ===== Input Grid 6x6 ===== -->
        ${inputGrid}

        <!-- Input label -->
        <text x="${inputX + 3 * (cellSize + gap) - gap / 2}" y="${inputY + 6 * (cellSize + gap) + 15}"
          text-anchor="middle" class="cnn-label" style="animation-delay: 0.8s;"
          fill="var(--text-dim)" font-family="var(--font-mono)" font-size="11">
          Input (6\u00D76)
        </text>

        <!-- Sliding kernel overlay -->
        <rect width="${kSize}" height="${kSize}" rx="4" fill="#F0B42915"
          stroke="#F0B429" stroke-width="2.5" stroke-dasharray="4 2" class="cnn-kernel">
          <animate attributeName="x"
            values="${[...Array(4)].map((_, r) => [...Array(4)].map((_, c) => kernelX + c * (cellSize + gap))).flat().join(';')};${kernelX}"
            dur="8s" repeatCount="indefinite" begin="1s" calcMode="discrete"/>
          <animate attributeName="y"
            values="${[...Array(4)].map((_, r) => [...Array(4)].map(() => kernelY + r * (cellSize + gap))).flat().join(';')};${kernelY}"
            dur="8s" repeatCount="indefinite" begin="1s" calcMode="discrete"/>
        </rect>

        <!-- Kernel label -->
        <text x="${inputX + 3 * (cellSize + gap) - gap / 2}" y="${inputY - 10}"
          text-anchor="middle" class="cnn-label" style="animation-delay: 0.9s;"
          fill="#F0B429" font-family="var(--font-mono)" font-size="11" font-weight="600">
          Kernel (3\u00D73)
        </text>

        <!-- Arrow: Input -> Feature Map -->
        <line x1="${inputX + 6 * (cellSize + gap) + 10}" y1="${arrowY}"
          x2="${fmX - 10}" y2="${arrowY}"
          stroke="#F0B429" stroke-width="2" class="cnn-arrow"
          marker-end="url(#cnnArrow)"/>
        <text x="${(inputX + 6 * (cellSize + gap) + fmX) / 2}" y="${arrowY - 10}"
          text-anchor="middle" class="cnn-label" style="animation-delay: 1.3s;"
          fill="#F0B429" font-family="var(--font-mono)" font-size="9">Convolve</text>

        <!-- ===== Feature Map 4x4 ===== -->
        ${featureGrid}

        <!-- Feature map label -->
        <text x="${fmX + 2 * (cellSize + gap) - gap / 2}" y="${fmY + 4 * (cellSize + gap) + 15}"
          text-anchor="middle" class="cnn-label" style="animation-delay: 3s;"
          fill="#C4A7E7" font-family="var(--font-mono)" font-size="11">
          Feature Map (4\u00D74)
        </text>

        <!-- Arrow: Feature Map -> Pooled -->
        <line x1="${fmX + 4 * (cellSize + gap) + 10}" y1="${arrowY}"
          x2="${poolX - 10}" y2="${arrowY}"
          stroke="#9CCFA4" stroke-width="2" class="cnn-pool-arrow"
          marker-end="url(#cnnArrowTeal)"/>
        <text x="${(fmX + 4 * (cellSize + gap) + poolX) / 2}" y="${arrowY - 10}"
          text-anchor="middle" class="cnn-label" style="animation-delay: 5.5s;"
          fill="#9CCFA4" font-family="var(--font-mono)" font-size="9">Max Pool</text>

        <!-- ===== Pooled Map 2x2 ===== -->
        ${pooledGrid}

        <!-- Pooled label -->
        <text x="${poolX + poolCell + gap / 2}" y="${poolY + 2 * (poolCell + gap) + 15}"
          text-anchor="middle" class="cnn-label" style="animation-delay: 6.5s;"
          fill="#9CCFA4" font-family="var(--font-mono)" font-size="11">
          Pooled (2\u00D72)
        </text>

        <!-- Operation explanation at bottom -->
        <g class="cnn-label" style="animation-delay: 3.5s;">
          <rect x="30" y="258" width="500" height="40" rx="8" fill="var(--bg-elevated)" stroke="var(--border-subtle)" stroke-width="1"/>
          <text x="280" y="275" text-anchor="middle" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="10">
            Convolution: element-wise multiply kernel \u00D7 receptive field, sum \u2192 single output value
          </text>
          <text x="280" y="290" text-anchor="middle" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="10">
            Max Pooling: take max value from each 2\u00D72 block \u2192 reduce spatial dimensions by half
          </text>
        </g>

        <!-- Highlight indicator on feature map showing current fill -->
        <rect width="${cellSize}" height="${cellSize}" rx="3" fill="none"
          stroke="#F0B429" stroke-width="2" opacity="0.8">
          <animate attributeName="x"
            values="${[...Array(4)].map((_, r) => [...Array(4)].map((_, c) => fmX + c * (cellSize + gap))).flat().join(';')};${fmX}"
            dur="8s" repeatCount="indefinite" begin="1s" calcMode="discrete"/>
          <animate attributeName="y"
            values="${[...Array(4)].map((_, r) => [...Array(4)].map(() => fmY + r * (cellSize + gap))).flat().join(';')};${fmY}"
            dur="8s" repeatCount="indefinite" begin="1s" calcMode="discrete"/>
          <animate attributeName="opacity" values="0.9;0.4;0.9" dur="0.5s" repeatCount="indefinite" begin="1s"/>
        </rect>
      </svg>
      <p style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-dim); margin-top: var(--space-3);">
        Convolutional Neural Network \u2014 Convolution and Max Pooling Operations
      </p>
    </div>
  `;
}
