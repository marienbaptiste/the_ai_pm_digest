import { Router } from './router.js';
import { renderSidebar, initSidebarToggle } from './components/sidebar.js';
import { renderDashboard } from './components/dashboard.js';
import { renderLesson } from './components/lesson-viewer.js';

const app = document.getElementById('app');
const router = new Router();

// Page transition wrapper
function withTransition(renderFn) {
  app.style.opacity = '0';
  app.style.transform = 'translateY(10px)';

  setTimeout(async () => {
    try {
      await renderFn();
    } catch (e) {
      console.error('Route render error:', e);
      app.innerHTML = `<div style="padding:2rem;color:#E8553A;">Error: ${e.message}</div>`;
    }
    requestAnimationFrame(() => {
      app.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
      app.style.opacity = '1';
      app.style.transform = 'translateY(0)';
    });
  }, 150);
}

// Routes
router
  .on('#/', () => {
    withTransition(() => renderDashboard(app));
  })
  .on('#/module/:moduleId/lesson/:lessonId', (params) => {
    withTransition(() => renderLesson(app, params.moduleId, params.lessonId));
  });

// On any navigation, update sidebar
router.onNavigate = () => {
  renderSidebar();
};

// Initialize — modules execute deferred, so DOM is already ready
function init() {
  renderSidebar();
  initSidebarToggle();
  router.start();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
