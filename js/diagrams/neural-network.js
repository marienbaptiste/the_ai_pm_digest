export function render(container) {
  const width = 700;
  const height = 350;
  const layers = [
    { name: 'Input', nodes: 4, color: '#7EB8DA' },
    { name: 'Hidden 1', nodes: 6, color: '#C4A7E7' },
    { name: 'Hidden 2', nodes: 5, color: '#C4A7E7' },
    { name: 'Output', nodes: 3, color: '#9CCFA4' }
  ];

  const layerSpacing = width / (layers.length + 1);
  const nodeRadius = 12;

  // Build node positions
  const nodePositions = [];
  layers.forEach((layer, li) => {
    const x = layerSpacing * (li + 1);
    const layerNodes = [];
    const totalHeight = (layer.nodes - 1) * 50;
    const startY = (height - totalHeight) / 2;

    for (let ni = 0; ni < layer.nodes; ni++) {
      layerNodes.push({ x, y: startY + ni * 50, color: layer.color });
    }
    nodePositions.push(layerNodes);
  });

  // Build connections
  let connections = '';
  for (let li = 0; li < nodePositions.length - 1; li++) {
    const from = nodePositions[li];
    const to = nodePositions[li + 1];
    from.forEach((fn, fi) => {
      to.forEach((tn, ti) => {
        const delay = (li * 0.3 + fi * 0.05 + ti * 0.03).toFixed(2);
        connections += `<line x1="${fn.x}" y1="${fn.y}" x2="${tn.x}" y2="${tn.y}"
          stroke="var(--border-medium)" stroke-width="1" opacity="0.3"
          class="nn-connection" style="animation-delay: ${delay}s;" />`;
      });
    });
  }

  // Build nodes
  let nodes = '';
  nodePositions.forEach((layer, li) => {
    layer.forEach((node, ni) => {
      const delay = (li * 0.4 + ni * 0.1).toFixed(2);
      nodes += `
        <circle cx="${node.x}" cy="${node.y}" r="${nodeRadius}"
          fill="${node.color}20" stroke="${node.color}" stroke-width="2"
          class="nn-node" style="animation-delay: ${delay}s;" />
        <circle cx="${node.x}" cy="${node.y}" r="4"
          fill="${node.color}" class="nn-node-core"
          style="animation-delay: ${delay}s;" />
      `;
    });
  });

  // Layer labels
  let labels = '';
  layers.forEach((layer, li) => {
    const x = layerSpacing * (li + 1);
    labels += `<text x="${x}" y="${height - 10}" text-anchor="middle"
      fill="var(--text-dim)" font-family="var(--font-mono)" font-size="11"
      class="nn-label" style="animation-delay: ${(li * 0.3).toFixed(2)}s;">${layer.name}</text>`;
  });

  // Animated data flow particles
  let particles = '';
  for (let li = 0; li < nodePositions.length - 1; li++) {
    const from = nodePositions[li][Math.floor(nodePositions[li].length / 2)];
    const to = nodePositions[li + 1][Math.floor(nodePositions[li + 1].length / 2)];
    const delay = (li * 0.8 + 1).toFixed(2);
    particles += `
      <circle r="3" fill="var(--accent-primary)" opacity="0" class="nn-particle">
        <animate attributeName="cx" from="${from.x}" to="${to.x}" dur="1.2s" begin="${delay}s" repeatCount="indefinite" />
        <animate attributeName="cy" from="${from.y}" to="${to.y}" dur="1.2s" begin="${delay}s" repeatCount="indefinite" />
        <animate attributeName="opacity" values="0;1;1;0" dur="1.2s" begin="${delay}s" repeatCount="indefinite" />
      </circle>
    `;
  }

  container.innerHTML = `
    <div style="text-align: center;">
      <svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg"
        style="max-width: 100%; height: auto;">
        <style>
          .nn-connection { animation: fadeIn 0.6s ease forwards; opacity: 0; }
          .nn-node { animation: scalePop 0.5s ease forwards; opacity: 0; transform-origin: center; }
          .nn-node-core { animation: nodeBreathe 2s ease-in-out infinite; }
          .nn-label { animation: fadeIn 0.5s ease forwards; opacity: 0; }
          @keyframes fadeIn { to { opacity: 0.3; } }
          @keyframes scalePop { to { opacity: 1; } }
          @keyframes nodeBreathe {
            0%, 100% { opacity: 0.8; }
            50% { opacity: 1; }
          }
        </style>
        ${connections}
        ${nodes}
        ${labels}
        ${particles}
      </svg>
      <p style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-dim); margin-top: var(--space-3);">
        Neural Network \u2014 Forward Pass Visualization
      </p>
    </div>
  `;
}
