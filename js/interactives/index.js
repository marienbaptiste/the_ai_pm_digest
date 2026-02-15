// Interactive module loader — scans for <div class="interactive" data-interactive="name">
// and mounts the corresponding interactive component

const interactiveLoaders = {
  'tokenizer': () => import('./tokenizer.js'),
  'attention-viz': () => import('./attention-viz.js'),
  'gradient-descent': () => import('./gradient-descent.js'),
  'neuron-playground': () => import('./neuron-playground.js'),
  'confusion-matrix': () => import('./confusion-matrix.js'),
  'embedding-projector': () => import('./embedding-projector.js'),
  'prompt-compare': () => import('./prompt-compare.js'),
};

export async function attachInteractives(container) {
  const placeholders = container.querySelectorAll('.interactive[data-interactive]');

  for (const el of placeholders) {
    const name = el.dataset.interactive;
    const loader = interactiveLoaders[name];

    if (!loader) {
      console.warn(`Interactive module not found: ${name}`);
      continue;
    }

    try {
      const mod = await loader();
      mod.mount(el);
    } catch (e) {
      console.error(`Failed to load interactive: ${name}`, e);
      el.innerHTML = `<p style="color: var(--text-dim); text-align: center; padding: var(--space-4);">Interactive loading...</p>`;
    }
  }
}
