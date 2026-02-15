export function render(container) {
  const steps = 6;
  const stepWidth = 90;
  const totalWidth = steps * stepWidth + 80;
  const height = 280;

  function noiseRect(x, y, w, h, noiseLevel, stepIndex) {
    let rects = '';
    const gridSize = 4;
    const cols = Math.floor(w / gridSize);
    const rows = Math.floor(h / gridSize);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const isNoise = Math.random() < noiseLevel;
        const px = x + c * gridSize;
        const py = y + r * gridSize;

        if (isNoise) {
          const brightness = Math.random();
          const color = `rgba(${Math.floor(brightness * 200)}, ${Math.floor(brightness * 200)}, ${Math.floor(brightness * 220)}, ${0.3 + brightness * 0.5})`;
          rects += `<rect x="${px}" y="${py}" width="${gridSize - 0.5}" height="${gridSize - 0.5}" fill="${color}" rx="0.5"/>`;
        } else {
          // Image-like colors (blues, greens for "clean" image)
          const hue = 160 + Math.random() * 40;
          const sat = 40 + Math.random() * 30;
          const light = 30 + Math.random() * 30;
          rects += `<rect x="${px}" y="${py}" width="${gridSize - 0.5}" height="${gridSize - 0.5}" fill="hsl(${hue}, ${sat}%, ${light}%)" rx="0.5"/>`;
        }
      }
    }
    return rects;
  }

  let stepsContent = '';
  const noiseLabels = ['Clean Image', 'Step t=1', 'Step t=2', 'Step t=3', 'Step t=4', 'Pure Noise'];
  const noiseLevels = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95];

  for (let i = 0; i < steps; i++) {
    const x = 40 + i * stepWidth;
    const y = 60;
    const delay = (i * 0.2).toFixed(2);

    stepsContent += `
      <g class="diff-step" style="animation: fadeIn 0.5s ease ${delay}s forwards; opacity: 0;">
        <rect x="${x}" y="${y}" width="70" height="70" rx="8" fill="var(--bg-elevated)" stroke="var(--border-medium)" stroke-width="1"/>
        ${noiseRect(x + 3, y + 3, 64, 64, noiseLevels[i], i)}
        <text x="${x + 35}" y="${y + 90}" text-anchor="middle" fill="var(--text-dim)" font-size="9" font-family="var(--font-mono)">${noiseLabels[i]}</text>
      </g>
    `;

    // Arrow between steps
    if (i < steps - 1) {
      stepsContent += `
        <g style="animation: fadeIn 0.3s ease ${(i * 0.2 + 0.1).toFixed(2)}s forwards; opacity: 0;">
          <line x1="${x + 72}" y1="${y + 35}" x2="${x + stepWidth - 2}" y2="${y + 35}"
            stroke="var(--accent-warm)" stroke-width="1.5" marker-end="url(#arrowhead)"/>
        </g>
      `;
    }
  }

  // Reverse process arrows
  let reverseArrows = '';
  for (let i = steps - 1; i > 0; i--) {
    const x1 = 40 + i * stepWidth + 35;
    const x2 = 40 + (i - 1) * stepWidth + 35;
    const delay = (1.5 + (steps - i) * 0.2).toFixed(2);

    reverseArrows += `
      <g style="animation: fadeIn 0.5s ease ${delay}s forwards; opacity: 0;">
        <path d="M${x1},170 Q${(x1 + x2) / 2},200 ${x2},170"
          fill="none" stroke="var(--accent-primary)" stroke-width="1.5" stroke-dasharray="4" marker-end="url(#arrowhead-green)"/>
      </g>
    `;
  }

  container.innerHTML = `
    <div style="text-align: center; width: 100%;">
      <svg viewBox="0 0 ${totalWidth} ${height}" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
        <style>
          @keyframes fadeIn { to { opacity: 1; } }
        </style>
        <defs>
          <marker id="arrowhead" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
            <polygon points="0 0, 6 2, 0 4" fill="var(--accent-warm)"/>
          </marker>
          <marker id="arrowhead-green" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
            <polygon points="0 0, 6 2, 0 4" fill="var(--accent-primary)"/>
          </marker>
        </defs>

        <!-- Title -->
        <text x="${totalWidth / 2}" y="25" text-anchor="middle" fill="var(--accent-warm)" font-family="var(--font-heading)" font-size="13" font-weight="600">
          Forward Process (Adding Noise)
        </text>

        <!-- Forward process steps -->
        ${stepsContent}

        <!-- Reverse label -->
        <text x="${totalWidth / 2}" y="215" text-anchor="middle" fill="var(--accent-primary)" font-family="var(--font-heading)" font-size="13" font-weight="600"
          style="animation: fadeIn 0.5s ease 1.5s forwards; opacity: 0;">
          Reverse Process (Denoising / Generation)
        </text>

        <!-- Reverse arrows -->
        ${reverseArrows}

        <!-- Legend -->
        <text x="${totalWidth / 2}" y="${height - 20}" text-anchor="middle" fill="var(--text-dim)" font-family="var(--font-mono)" font-size="10"
          style="animation: fadeIn 0.5s ease 2.5s forwards; opacity: 0;">
          The model learns to reverse the noise process, generating images from random noise
        </text>
      </svg>
      <p style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-dim); margin-top: var(--space-3);">
        Diffusion Model \u2014 Forward & Reverse Process
      </p>
    </div>
  `;
}
