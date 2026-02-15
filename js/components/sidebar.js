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
  console.log(`[Sidebar] updateProgress: ${overall.completed}/${overall.total} = ${overall.percent}%`);
  const container = document.getElementById('sidebar-progress-ring');
  const text = document.getElementById('sidebar-progress-text');

  if (container) {
    container.innerHTML = '';
    container.appendChild(createProgressRing(overall.percent, 36, 3));
  } else {
    console.warn('[Sidebar] progress-ring container not found');
  }
  if (text) {
    text.textContent = `${overall.completed}/${overall.total} — ${overall.percent}% Complete`;
  } else {
    console.warn('[Sidebar] progress-text element not found');
  }
}

// Listen for progress changes from any source
window.addEventListener('progress-changed', (e) => {
  updateSidebarProgress();
  // Update lesson completion state in sidebar nav without full re-render
  const lessonKey = e.detail?.lessonKey;
  if (lessonKey) {
    const link = document.querySelector(`.nav-lesson[data-lesson-key="${lessonKey}"]`);
    if (link) {
      if (e.detail.reset) {
        link.classList.remove('is-completed');
      } else {
        link.classList.add('is-completed');
      }
    }
    // Update module progress count
    const modId = lessonKey.split('-')[0];
    const mod = modules.find(m => m.id === modId);
    if (mod) {
      const modProgress = storage.getModuleProgress(mod);
      const modHeader = document.querySelector(`.nav-module[data-module="${modId}"] .nav-module__header`);
      if (modHeader) {
        const existing = modHeader.querySelector('.tag');
        if (modProgress.completed > 0) {
          const tagHtml = `<span class="tag tag--primary" style="margin-left: auto; margin-right: 4px;">${modProgress.completed}/${modProgress.total}</span>`;
          if (existing) existing.outerHTML = tagHtml;
          else modHeader.querySelector('.nav-module__chevron').insertAdjacentHTML('beforebegin', tagHtml);
        } else if (existing) {
          existing.remove();
        }
      }
    }
  }
});

// Global reset button in sidebar footer
export function initGlobalReset() {
  const btn = document.getElementById('sidebar-reset-all');
  if (!btn) return;
  btn.addEventListener('click', () => {
    if (!confirm('Reset ALL progress? This will clear all quiz answers and lesson completions.')) return;
    storage.resetAll();
    renderSidebar();
    // If on a lesson page, re-navigate to reset quiz UI
    if (window.location.hash.includes('/lesson/')) {
      window.dispatchEvent(new HashChangeEvent('hashchange'));
    }
  });
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
