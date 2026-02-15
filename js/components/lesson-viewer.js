import { getModule, getLesson, modules } from '../../data/modules.js';
import { storage } from '../storage.js';
import { attachGlossary } from './glossary.js';
import { renderQuiz } from './quiz-engine.js';
import { renderSidebar } from './sidebar.js';

const diagramLoaders = {
  'neural-network': () => import('../diagrams/neural-network.js'),
  'transformer': () => import('../diagrams/transformer.js'),
  'diffusion': () => import('../diagrams/diffusion.js'),
  'rag-pipeline': () => import('../diagrams/rag-pipeline.js'),
  'training-loop': () => import('../diagrams/training-loop.js'),
};

export async function renderLesson(container, moduleId, lessonId) {
  const mod = getModule(moduleId);
  const lessonMeta = getLesson(moduleId, lessonId);

  if (!mod || !lessonMeta) {
    container.innerHTML = `
      <div class="empty-state animate-in">
        <div class="empty-state__icon">\u{1F50D}</div>
        <h3 class="empty-state__title">Lesson not found</h3>
        <p class="empty-state__text">This lesson doesn't exist.</p>
        <a href="#/" class="btn btn--primary" style="margin-top: var(--space-4);">Back to Dashboard</a>
      </div>
    `;
    return;
  }

  const lessonKey = `${moduleId}-${lessonId}`;
  const isCompleted = storage.isLessonCompleted(lessonKey);
  const prevScore = storage.getQuizScore(lessonKey);

  // Show loading state
  container.innerHTML = `
    <div class="lesson-loading animate-in" style="text-align: center; padding: var(--space-16);">
      <div class="spinner" style="margin: 0 auto var(--space-4);"></div>
      <p style="color: var(--text-secondary);">Loading lesson...</p>
    </div>
  `;

  // Load lesson content
  let lessonData;
  const fileName = `${moduleId}-${getModuleFileName(moduleId)}`;
  try {
    const lessonModule = await import(`../../data/lessons/${fileName}.js`);
    lessonData = lessonModule.lessons?.[lessonId];
  } catch (e) {
    // Module not yet created
    console.log(`Lesson file not found: ${fileName}.js`);
  }

  // Navigation links
  const lessonIndex = mod.lessons.findIndex(l => l.id === lessonId);
  const prevLesson = lessonIndex > 0 ? mod.lessons[lessonIndex - 1] : null;
  const nextLesson = lessonIndex < mod.lessons.length - 1 ? mod.lessons[lessonIndex + 1] : null;

  // Find next module's first lesson
  const modIndex = modules.findIndex(m => m.id === moduleId);
  const nextModule = modIndex < modules.length - 1 ? modules[modIndex + 1] : null;
  const nextModFirstLesson = nextModule ? nextModule.lessons[0] : null;

  container.innerHTML = `
    <article class="lesson animate-in">
      <header class="lesson__header">
        <div class="lesson__breadcrumb">
          <a href="#/" class="lesson__breadcrumb-link">Home</a>
          <span class="lesson__breadcrumb-sep">/</span>
          <span style="color: ${mod.color};">${mod.icon} Module ${mod.number}</span>
          <span class="lesson__breadcrumb-sep">/</span>
          <span>Lesson ${lessonIndex + 1}</span>
          ${isCompleted ? '<span class="tag tag--primary" style="margin-left: 8px;">Completed</span>' : ''}
          ${prevScore !== null ? `<span class="tag tag--blue" style="margin-left: 4px;">Score: ${prevScore}%</span>` : ''}
        </div>
        <h1 class="lesson__title">${lessonMeta.title}</h1>
        <div class="lesson__meta">
          <span style="color: ${mod.color};">${mod.title}</span>
        </div>
      </header>

      ${lessonMeta.diagram && lessonMeta.diagram !== 'none' ? `
        <div class="lesson__diagram" id="lesson-diagram">
          <div class="lesson__diagram-loading">
            <div class="spinner"></div>
          </div>
        </div>
      ` : ''}

      <div class="lesson__content" id="lesson-content">
        ${lessonData?.content || `
          <div class="lesson__coming-soon">
            <p style="color: var(--text-secondary); text-align: center; padding: var(--space-8);">
              \u{1F6A7} Lesson content is loading...
            </p>
          </div>
        `}
      </div>

      <div class="lesson__quiz" id="lesson-quiz"></div>

      <nav class="lesson__nav">
        ${prevLesson ? `
          <a href="#/module/${moduleId}/lesson/${prevLesson.id}" class="btn btn--ghost">
            \u2190 ${prevLesson.title}
          </a>
        ` : '<div></div>'}
        ${nextLesson ? `
          <a href="#/module/${moduleId}/lesson/${nextLesson.id}" class="btn btn--primary">
            ${nextLesson.title} \u2192
          </a>
        ` : nextModFirstLesson ? `
          <a href="#/module/${nextModule.id}/lesson/${nextModFirstLesson.id}" class="btn btn--primary">
            Next: ${nextModule.title} \u2192
          </a>
        ` : `
          <a href="#/" class="btn btn--primary">
            Back to Dashboard \u{1C6A}
          </a>
        `}
      </nav>
    </article>
  `;

  // Attach glossary tooltips
  const contentEl = document.getElementById('lesson-content');
  if (contentEl) {
    await attachGlossary(contentEl);
  }

  // Load diagram
  if (lessonMeta.diagram && lessonMeta.diagram !== 'none') {
    const diagramContainer = document.getElementById('lesson-diagram');
    try {
      const loader = diagramLoaders[lessonMeta.diagram];
      if (loader) {
        const diagramModule = await loader();
        diagramModule.render(diagramContainer);
      }
    } catch (e) {
      diagramContainer.innerHTML = '<p style="color: var(--text-dim); text-align: center; padding: var(--space-4);">Diagram loading...</p>';
    }
  }

  // Render quiz
  const quizContainer = document.getElementById('lesson-quiz');
  if (lessonData?.quiz) {
    renderQuiz(quizContainer, lessonData.quiz, lessonKey);
  }

  // Update sidebar active state
  renderSidebar();

  // Scroll to top
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function getModuleFileName(moduleId) {
  const names = {
    m01: 'ai-foundations',
    m02: 'deep-learning',
    m03: 'transformers',
    m04: 'llms',
    m05: 'diffusion',
    m06: 'rag',
    m07: 'ai-product-mgmt',
    m08: 'sdk-platforms',
    m09: 'ai-ethics',
    m10: 'leadership',
    m11: 'deepmind-gemini',
    m12: 'interview-prep'
  };
  return names[moduleId] || moduleId;
}
