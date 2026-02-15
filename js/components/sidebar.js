import { modules } from '../../data/modules.js';
import { storage } from '../storage.js';
import { createProgressRing } from './progress-ring.js';

export function renderSidebar() {
  const nav = document.getElementById('sidebar-nav');
  const currentHash = window.location.hash || '#/';

  nav.innerHTML = modules.map(mod => {
    const modProgress = storage.getModuleProgress(mod);
    const isOpen = currentHash.includes(mod.id);

    return `
      <div class="nav-module ${isOpen ? 'is-open' : ''}" data-module="${mod.id}">
        <button class="nav-module__header" aria-expanded="${isOpen}">
          <span class="nav-module__icon" style="background: ${mod.color}20; color: ${mod.color};">${mod.icon}</span>
          <span class="nav-module__label">${mod.number}. ${mod.title}</span>
          ${modProgress.completed > 0 ? `<span class="tag tag--primary" style="margin-left: auto; margin-right: 4px;">${modProgress.completed}/${modProgress.total}</span>` : ''}
          <svg class="nav-module__chevron" viewBox="0 0 16 16" fill="none">
            <path d="M6 4l4 4-4 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
          </svg>
        </button>
        <div class="nav-module__lessons">
          ${mod.lessons.map(lesson => {
            const key = `${mod.id}-${lesson.id}`;
            const isCompleted = storage.isLessonCompleted(key);
            const isActive = currentHash === `#/module/${mod.id}/lesson/${lesson.id}`;
            return `
              <a href="#/module/${mod.id}/lesson/${lesson.id}"
                 class="nav-lesson ${isActive ? 'is-active' : ''} ${isCompleted ? 'is-completed' : ''}"
                 data-lesson-key="${key}">
                <span class="nav-lesson__check"></span>
                <span>${lesson.title}</span>
              </a>
            `;
          }).join('')}
        </div>
      </div>
    `;
  }).join('');

  // Module toggle handlers
  nav.querySelectorAll('.nav-module__header').forEach(header => {
    header.addEventListener('click', () => {
      const module = header.closest('.nav-module');
      module.classList.toggle('is-open');
      header.setAttribute('aria-expanded', module.classList.contains('is-open'));
    });
  });

  // Update sidebar progress
  updateSidebarProgress();
}

export function updateSidebarProgress() {
  const overall = storage.getOverallProgress(modules);
  const container = document.getElementById('sidebar-progress-ring');
  const text = document.getElementById('sidebar-progress-text');

  container.innerHTML = '';
  container.appendChild(createProgressRing(overall.percent, 36, 3));
  text.textContent = `${overall.percent}% Complete`;
}

// Mobile sidebar toggle
export function initSidebarToggle() {
  const toggle = document.getElementById('sidebar-toggle');
  const sidebar = document.getElementById('sidebar');

  toggle.addEventListener('click', () => {
    sidebar.classList.toggle('is-open');
    toggle.classList.toggle('is-active');
  });

  // Close on navigation
  sidebar.addEventListener('click', (e) => {
    if (e.target.closest('.nav-lesson')) {
      sidebar.classList.remove('is-open');
      toggle.classList.remove('is-active');
    }
  });
}
