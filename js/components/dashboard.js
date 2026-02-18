import { modules } from '../../data/modules.js';
import { storage } from '../storage.js';
import { createProgressRing } from './progress-ring.js';

export function renderDashboard(container) {
  const overall = storage.getOverallProgress(modules);
  const streak = storage.getStreak();

  container.innerHTML = `
    <div class="dashboard">
      <header class="dashboard__hero">
        <div class="dashboard__hero-bg"></div>
        <div class="dashboard__hero-content">
          <p class="dashboard__overline">AI Product Manager Training</p>
          <h1 class="dashboard__title">The AI PM Digest</h1>
          <p class="dashboard__subtitle">Your accelerated path to becoming a world-class AI Product Manager at Google DeepMind. Master everything from neural network fundamentals to Gemini product strategy.</p>
          <div class="dashboard__stats">
            <div class="stat">
              <div class="stat__value">${overall.completed}</div>
              <div class="stat__label">Lessons Done</div>
            </div>
            <div class="stat">
              <div class="stat__value">${overall.total}</div>
              <div class="stat__label">Total Lessons</div>
            </div>
            <div class="stat">
              <div class="stat__value">${streak}</div>
              <div class="stat__label">Day Streak</div>
            </div>
            <div class="stat">
              <div class="stat__value">${overall.percent}%</div>
              <div class="stat__label">Complete</div>
            </div>
          </div>
          ${overall.percent > 0 ? `
            <div class="dashboard__progress-bar">
              <div class="progress-bar">
                <div class="progress-bar__fill" style="width: ${overall.percent}%"></div>
              </div>
            </div>
          ` : ''}
        </div>
      </header>

      <section class="dashboard__modules">
        <h2 class="dashboard__section-title">Your Learning Path</h2>
        <p class="dashboard__section-desc">12 modules taking you from AI fundamentals to DeepMind interview readiness.</p>
        <div class="dashboard__grid orchestrate">
          ${modules.map((mod) => {
            const modProgress = storage.getModuleProgress(mod);
            const nextLesson = mod.lessons.find(l => !storage.isLessonCompleted(`${mod.id}-${l.id}`));
            const href = nextLesson
              ? `#/module/${mod.id}/lesson/${nextLesson.id}`
              : `#/module/${mod.id}/lesson/${mod.lessons[0].id}`;

            return `
              <a href="${href}" class="card card--interactive card--module" style="--module-color: ${mod.color};">
                <div class="card__icon" style="background: ${mod.color}15; color: ${mod.color};">
                  <i data-lucide="${mod.icon}" style="width:22px;height:22px;display:block;"></i>
                </div>
                <div class="card__body">
                  <span class="card__number">${String(mod.number).padStart(2, '0')}</span>
                  <div class="card__title" style="color: ${mod.color};">
                    ${mod.title}
                  </div>
                  <div class="card__description">${mod.description}</div>
                  <div class="card__meta">
                    <span>${mod.lessons.length} lessons</span>
                    ${modProgress.completed > 0 ? `
                      <span style="color: var(--accent-primary);">${modProgress.completed}/${modProgress.total} complete</span>
                    ` : ''}
                  </div>
                  ${modProgress.percent > 0 ? `
                    <div class="progress-bar" style="margin-top: 8px;">
                      <div class="progress-bar__fill" style="width: ${modProgress.percent}%;"></div>
                    </div>
                  ` : ''}
                </div>
              </a>
            `;
          }).join('')}
        </div>
      </section>

      <section class="dashboard__target">
        <div class="card" style="border-color: var(--accent-purple-dim);">
          <h3 style="color: var(--accent-purple); margin-bottom: var(--space-3);">Target Role</h3>
          <h4 style="margin-bottom: var(--space-2);">Product Manager, Gemini App Integrated Assistance</h4>
          <p style="color: var(--text-secondary); font-size: var(--text-sm); margin-bottom: var(--space-4);">Google DeepMind \u2014 Zurich, Switzerland</p>
          <div class="dashboard__skills orchestrate--wave">
            <span class="tag tag--primary">LLMs</span>
            <span class="tag tag--blue">SDK Design</span>
            <span class="tag tag--purple">Diffusion Models</span>
            <span class="tag tag--warm">RAG</span>
            <span class="tag tag--primary">Product Vision</span>
            <span class="tag tag--blue">Cross-functional</span>
            <span class="tag tag--purple">AI Safety</span>
          </div>
        </div>
      </section>
    </div>
  `;

  if (window.lucide) window.lucide.createIcons();
}
