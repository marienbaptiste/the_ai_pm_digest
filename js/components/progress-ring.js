export function createProgressRing(percent, size = 44, strokeWidth = 4) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (percent / 100) * circumference;

  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('width', size);
  svg.setAttribute('height', size);
  svg.setAttribute('viewBox', `0 0 ${size} ${size}`);
  svg.classList.add('progress-ring');

  svg.innerHTML = `
    <circle
      cx="${size / 2}" cy="${size / 2}" r="${radius}"
      fill="none"
      stroke="var(--bg-elevated)"
      stroke-width="${strokeWidth}"
    />
    <circle
      cx="${size / 2}" cy="${size / 2}" r="${radius}"
      fill="none"
      stroke="url(#progress-gradient-${size})"
      stroke-width="${strokeWidth}"
      stroke-linecap="round"
      stroke-dasharray="${circumference}"
      stroke-dashoffset="${offset}"
      transform="rotate(-90 ${size / 2} ${size / 2})"
      style="transition: stroke-dashoffset 0.8s cubic-bezier(0.16, 1, 0.3, 1); --circumference: ${circumference};"
    />
    <text
      x="${size / 2}" y="${size / 2}"
      text-anchor="middle"
      dominant-baseline="central"
      fill="var(--text-primary)"
      font-family="var(--font-heading)"
      font-size="${size * 0.24}px"
      font-weight="600"
    >${percent}%</text>
    <defs>
      <linearGradient id="progress-gradient-${size}" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="var(--accent-primary)"/>
        <stop offset="100%" stop-color="var(--accent-blue)"/>
      </linearGradient>
    </defs>
  `;

  return svg;
}
