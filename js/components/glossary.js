let glossaryData = null;
let activeTooltip = null;

async function loadGlossary() {
  if (glossaryData) return glossaryData;
  const module = await import('../../data/glossary-terms.js');
  glossaryData = module.glossaryTerms;
  return glossaryData;
}

export async function attachGlossary(container) {
  const terms = await loadGlossary();
  const termElements = container.querySelectorAll('.term[data-term]');

  termElements.forEach(el => {
    const termKey = el.dataset.term.toLowerCase();
    const termData = terms[termKey];

    if (!termData) return;

    el.addEventListener('mouseenter', (e) => showTooltip(e, el, termData));
    el.addEventListener('mouseleave', () => hideTooltip());
    el.addEventListener('click', (e) => {
      e.preventDefault();
      if (activeTooltip) {
        hideTooltip();
      } else {
        showTooltip(e, el, termData);
      }
    });
  });
}

function showTooltip(event, anchor, termData) {
  hideTooltip();

  const portal = document.getElementById('glossary-portal');
  const tooltip = document.createElement('div');
  tooltip.className = 'glossary-tooltip';

  tooltip.innerHTML = `
    <div class="glossary-tooltip__term">${termData.term}</div>
    <div class="glossary-tooltip__definition">${termData.definition}</div>
    ${termData.analogy ? `
      <div class="glossary-tooltip__analogy">${termData.analogy}</div>
    ` : ''}
    ${termData.whyItMatters ? `
      <div class="glossary-tooltip__why">
        <strong style="color: var(--accent-warm); font-size: var(--text-xs); text-transform: uppercase; letter-spacing: 0.05em;">Why it matters for PMs:</strong>
        <p style="margin-top: 4px; font-size: var(--text-sm); color: var(--text-secondary);">${termData.whyItMatters}</p>
      </div>
    ` : ''}
    ${termData.related && termData.related.length > 0 ? `
      <div class="glossary-tooltip__related" style="margin-top: 8px;">
        ${termData.related.map(r => `<span class="tag tag--blue">${r}</span>`).join('')}
      </div>
    ` : ''}
  `;

  portal.appendChild(tooltip);
  activeTooltip = tooltip;

  // Position
  const rect = anchor.getBoundingClientRect();
  const tooltipRect = tooltip.getBoundingClientRect();
  const padding = 12;

  let top = rect.bottom + padding;
  let left = rect.left + rect.width / 2 - tooltipRect.width / 2;

  // Keep in viewport
  if (left < padding) left = padding;
  if (left + tooltipRect.width > window.innerWidth - padding) {
    left = window.innerWidth - tooltipRect.width - padding;
  }
  if (top + tooltipRect.height > window.innerHeight - padding) {
    top = rect.top - tooltipRect.height - padding;
  }

  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
}

function hideTooltip() {
  if (activeTooltip) {
    activeTooltip.remove();
    activeTooltip = null;
  }
}

// Close tooltip on scroll or click outside
document.addEventListener('scroll', hideTooltip, true);
document.addEventListener('click', (e) => {
  if (activeTooltip && !e.target.closest('.term') && !e.target.closest('.glossary-tooltip')) {
    hideTooltip();
  }
});
