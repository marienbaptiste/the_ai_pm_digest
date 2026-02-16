import { storage } from '../storage.js';
import { renderSidebar, updateSidebarProgress } from './sidebar.js';

export function renderQuiz(container, quizData, lessonKey) {
  if (!quizData || !quizData.questions || quizData.questions.length === 0) {
    container.innerHTML = `
      <div class="quiz-placeholder">
        <p>No quiz available for this lesson yet.</p>
      </div>
    `;
    return;
  }

  const totalQ = quizData.questions.length;

  // Restore persisted state or start fresh
  const saved = storage.getQuizState(lessonKey);
  const state = saved
    ? { answers: saved.answers || {}, checked: saved.checked || {}, selfGraded: saved.selfGraded || {} }
    : { answers: {}, checked: {}, selfGraded: {} };

  // Convert string keys back to numbers (JSON stringify converts numeric keys to strings)
  function normalizeKeys(obj) {
    const out = {};
    for (const k of Object.keys(obj)) out[parseInt(k)] = obj[k];
    return out;
  }
  if (saved) {
    state.answers = normalizeKeys(state.answers);
    state.checked = normalizeKeys(state.checked);
    state.selfGraded = normalizeKeys(state.selfGraded);
  }

  function persistState() {
    storage.saveQuizState(lessonKey, {
      answers: state.answers,
      checked: state.checked,
      selfGraded: state.selfGraded
    });
  }

  function getScore() {
    let correct = 0;
    quizData.questions.forEach((q, i) => {
      const isChecked = !!state.checked[i];
      const isCorrect = isChecked && checkAnswer(q, state.answers[i], state.selfGraded[i]);
      if (isCorrect) correct++;
    });
    const percent = totalQ > 0 ? Math.round((correct / totalQ) * 100) : 0;
    return { correct, total: totalQ, percent };
  }

  function allChecked() {
    // Check each question individually (avoids key-type issues with Object.keys)
    for (let i = 0; i < totalQ; i++) {
      if (!state.checked[i]) {
        console.log(`[Quiz] allChecked: Q${i + 1} not yet checked`);
        return false;
      }
      const q = quizData.questions[i];
      if ((q.type === 'short' || q.type === 'scenario') && state.selfGraded[i] === undefined) {
        console.log(`[Quiz] allChecked: Q${i + 1} (${q.type}) awaiting self-grade`);
        return false;
      }
    }
    console.log(`[Quiz] allChecked: true — all ${totalQ} questions checked`);
    return true;
  }

  function render() {
    const score = getScore();
    const isComplete = allChecked();
    const isAlreadyCompleted = storage.isLessonCompleted(lessonKey);

    // Log completion state for debugging
    console.log(`[Quiz] render: lessonKey=${lessonKey}, isComplete=${isComplete}, score=${score.percent}%, alreadyCompleted=${isAlreadyCompleted}, checked=${JSON.stringify(state.checked)}`);

    // Save progress when all answered and score >= 70%
    if (isComplete && score.percent >= 70) {
      if (!isAlreadyCompleted) {
        console.log(`[Quiz] Marking lesson complete: ${lessonKey} with score ${score.percent}%`);
        // markLessonComplete saves to localStorage and dispatches 'progress-changed'
        // event, which updates sidebar progress ring, text, lesson link, and module count
        storage.markLessonComplete(lessonKey, score.percent);
        // Direct fallback update in case event handler missed it
        try { updateSidebarProgress(); } catch (e) { console.error('[Quiz] updateSidebarProgress error:', e); }
        // Deferred update to ensure DOM is stable after all synchronous work
        requestAnimationFrame(() => {
          try { updateSidebarProgress(); } catch (e) { /* silent */ }
        });
        const breadcrumb = document.querySelector('.lesson__breadcrumb');
        if (breadcrumb && !breadcrumb.querySelector('.tag--primary')) {
          breadcrumb.insertAdjacentHTML('beforeend',
            '<span class="tag tag--primary" style="margin-left: 8px;">Completed</span>'
          );
        }
        showToast(`Lesson completed! Score: ${score.percent}%`);
      } else {
        // Already completed — check for high score
        const prevScore = storage.getQuizScore(lessonKey);
        if (score.percent > (prevScore || 0)) {
          storage.markLessonComplete(lessonKey, score.percent);
          try { updateSidebarProgress(); } catch (e) { console.error('[Quiz] updateSidebarProgress error:', e); }
          showToast(`New high score: ${score.percent}%`);
        }
      }
    }

    const checkedCount = Object.keys(state.checked).length;
    const progressPercent = totalQ > 0 ? Math.round((checkedCount / totalQ) * 100) : 0;

    let subtitleText;
    if (isComplete) {
      subtitleText = `Score: ${score.correct}/${score.total} (${score.percent}%) ${score.percent >= 70 ? '\u2014 Passed!' : '\u2014 Need 70% to pass.'}`;
    } else if (checkedCount > 0) {
      const remaining = totalQ - checkedCount;
      subtitleText = `${checkedCount}/${totalQ} checked \u2014 ${remaining} question${remaining !== 1 ? 's' : ''} remaining`;
    } else {
      subtitleText = `${totalQ} questions \u2014 answer & check all to complete. Need 70% to pass.`;
    }

    container.innerHTML = `
      <div class="quiz animate-in">
        <div class="quiz__header">
          <div class="quiz__header-icon">\u{1F9EA}</div>
          <div style="flex: 1;">
            <h3 class="quiz__title">Knowledge Check</h3>
            <p class="quiz__subtitle">${subtitleText}</p>
          </div>
          <div class="quiz__progress-ring">
            <span class="quiz__progress-count">${checkedCount}/${totalQ}</span>
          </div>
        </div>
        <div class="quiz__progress-bar" style="height: 3px; background: var(--bg-elevated); border-radius: 2px; margin-bottom: var(--space-6); overflow: hidden;">
          <div style="height: 100%; width: ${progressPercent}%; background: linear-gradient(90deg, var(--accent-primary), ${isComplete && score.percent >= 70 ? 'var(--accent-green)' : 'var(--accent-blue)'}); border-radius: 2px; transition: width 0.5s cubic-bezier(0.16, 1, 0.3, 1);"></div>
        </div>

        <div class="quiz__questions">
          ${quizData.questions.map((q, i) => renderQuestion(q, i, state)).join('')}
        </div>

        ${isComplete ? `
          <div class="quiz__results animate-in">
            <div class="quiz__score-card ${score.percent >= 70 ? 'quiz__score-card--pass' : 'quiz__score-card--fail'}">
              <div class="quiz__score-value">${score.percent}%</div>
              <div class="quiz__score-label">
                ${score.percent >= 70 ? 'Passed! Great work.' : 'Not quite \u2014 review the explanations above and try again.'}
              </div>
              ${score.percent < 70 ? `
                <button class="quiz__action-btn quiz__action-btn--retry" id="quiz-retry">
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M2 8a6 6 0 1 1 1.76 4.24" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M2 12.5V8h4.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
                  Try Again
                </button>
              ` : `
                <p style="color: var(--text-secondary); font-size: var(--text-sm); margin-top: var(--space-3);">
                  This lesson has been marked as complete.
                </p>
              `}
            </div>
          </div>
        ` : ''}

        <div class="quiz__footer">
          <button class="quiz__action-btn quiz__action-btn--reset" id="quiz-reset-lesson">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M1.75 3.5h10.5M5.25 3.5V2.33a.58.58 0 0 1 .58-.58h2.34a.58.58 0 0 1 .58.58V3.5m1.75 0v8.17a1.17 1.17 0 0 1-1.17 1.16H4.67a1.17 1.17 0 0 1-1.17-1.16V3.5h7" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
            Reset Lesson Progress
          </button>
        </div>
      </div>
    `;

    // Attach choice listeners for unchecked questions
    attachChoiceListeners(quizData.questions, state, container, render, persistState);

    // Attach text input listeners
    attachTextListeners(quizData.questions, state, container, render, persistState);

    // Attach expert note toggles
    container.querySelectorAll('.quiz__expert-toggle').forEach(btn => {
      btn.addEventListener('click', () => {
        const content = btn.nextElementSibling;
        const isOpen = content.style.maxHeight !== '0px' && content.style.maxHeight;
        content.style.maxHeight = isOpen ? '0px' : content.scrollHeight + 'px';
        btn.classList.toggle('is-open');
      });
    });

    // Self-grade buttons for open-ended questions
    container.querySelectorAll('[data-self-grade]').forEach(btn => {
      btn.addEventListener('click', () => {
        const qi = parseInt(btn.dataset.selfGrade);
        const grade = btn.dataset.grade;
        console.log(`[Quiz] Self-grade Q${qi + 1}: grade=${grade}, isCorrect=${grade === 'correct'}`);
        state.selfGraded[qi] = grade === 'correct';
        persistState();
        render();
      });
    });

    // Per-question reset buttons
    container.querySelectorAll('[data-reset-question]').forEach(btn => {
      btn.addEventListener('click', () => {
        const qi = parseInt(btn.dataset.resetQuestion);
        delete state.answers[qi];
        delete state.checked[qi];
        delete state.selfGraded[qi];
        persistState();
        render();
      });
    });

    // Retry button (resets all questions)
    document.getElementById('quiz-retry')?.addEventListener('click', () => {
      state.answers = {};
      state.checked = {};
      state.selfGraded = {};
      persistState();
      render();
    });

    // Reset lesson button (resets progress + answers)
    document.getElementById('quiz-reset-lesson')?.addEventListener('click', () => {
      storage.resetLesson(lessonKey);
      storage.clearQuizState(lessonKey);
      state.answers = {};
      state.checked = {};
      state.selfGraded = {};
      render();
      renderSidebar();
      document.querySelector('.lesson__breadcrumb .tag--primary')?.remove();
      document.querySelector('.lesson__breadcrumb .tag--blue')?.remove();
    });
  }

  render();
}

function renderQuestion(q, index, state) {
  const isChecked = state.checked[index];
  const isOpenEnded = q.type === 'short' || q.type === 'scenario';
  const needsSelfGrade = isChecked && isOpenEnded && state.selfGraded[index] === undefined;
  const isCorrect = isChecked && checkAnswer(q, state.answers[index], state.selfGraded[index]);
  const stateClass = isChecked ? (needsSelfGrade ? 'quiz-pending' : (isCorrect ? 'quiz-correct' : 'quiz-incorrect')) : '';

  const difficultyColors = {
    foundational: 'tag--primary',
    applied: 'tag--warm',
    expert: 'tag--purple'
  };

  return `
    <div class="quiz__question ${stateClass}" data-question="${index}">
      <div class="quiz__question-header">
        <span class="quiz__question-num">Q${index + 1}</span>
        <span class="tag ${difficultyColors[q.difficulty] || 'tag--blue'}">${q.difficulty || 'applied'}</span>
        ${isChecked ? (needsSelfGrade ? `<span class="quiz__check-badge" style="background: var(--accent-blue); color: white;">\u2026</span>` : `<span class="quiz__check-badge ${isCorrect ? 'quiz__check-badge--correct' : 'quiz__check-badge--incorrect'}">${isCorrect ? '\u2713' : '\u2717'}</span>`) : ''}
        ${isChecked ? `
          <button class="quiz__question-reset" data-reset-question="${index}" title="Reset this question">
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none"><path d="M2 8a6 6 0 1 1 1.76 4.24" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M2 12.5V8h4.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
          </button>
        ` : ''}
      </div>
      <p class="quiz__question-text">${q.question}</p>

      ${q.type === 'mc' || q.type === 'multi' ? renderChoices(q, index, state) : ''}
      ${q.type === 'short' || q.type === 'scenario' ? renderTextInput(q, index, state) : ''}

      ${!isChecked && (q.type === 'mc' || q.type === 'multi') ? `
        <button class="quiz__action-btn quiz__action-btn--check" data-check="${index}">
          Check Answer
        </button>
      ` : ''}

      ${!isChecked && (q.type === 'short' || q.type === 'scenario') ? `
        <button class="quiz__action-btn quiz__action-btn--check" data-check="${index}">
          Show Answer
        </button>
      ` : ''}

      ${isChecked && needsSelfGrade ? `
        <div class="quiz__self-grade">
          <p class="quiz__self-grade-prompt">Compare your answer with the model answer above. How did you do?</p>
          <div class="quiz__self-grade-buttons">
            <button class="quiz__action-btn quiz__action-btn--check" data-self-grade="${index}" data-grade="correct">
              \u2705 I got it right
            </button>
            <button class="quiz__action-btn quiz__action-btn--retry" data-self-grade="${index}" data-grade="incorrect">
              \u274C I missed key points
            </button>
          </div>
        </div>
      ` : ''}
      ${isChecked && !needsSelfGrade ? `
        <div class="quiz__feedback ${isCorrect ? 'quiz__feedback--correct' : 'quiz__feedback--incorrect'}">
          <div class="quiz__feedback-icon">${isCorrect ? '\u2705' : '\u274C'}</div>
          <div class="quiz__feedback-text">
            <strong>${isCorrect ? 'Correct!' : 'Incorrect'}</strong>
            ${formatModelAnswer(q.explanation)}
          </div>
        </div>
        ${q.expertNote ? `
          <button class="quiz__expert-toggle">
            <span>\u{1F9E0} Expert Insight</span>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M4 6l4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
          </button>
          <div class="quiz__expert-content" style="max-height: 0px;">
            ${formatModelAnswer(q.expertNote)}
          </div>
        ` : ''}
      ` : ''}
    </div>
  `;
}

function renderChoices(q, index, state) {
  const isChecked = state.checked[index];

  return `
    <div class="quiz__choices" data-type="${q.type}">
      ${q.options.map((opt, oi) => {
        const selected = q.type === 'multi'
          ? (state.answers[index] || []).includes(oi)
          : state.answers[index] === oi;
        const isCorrectChoice = q.type === 'multi'
          ? q.correct.includes(oi)
          : q.correct === oi;

        let choiceClass = '';
        if (isChecked) {
          if (isCorrectChoice) choiceClass = 'quiz__choice--correct';
          else if (selected && !isCorrectChoice) choiceClass = 'quiz__choice--incorrect';
        }

        return `
          <label class="quiz__choice ${selected ? 'is-selected' : ''} ${choiceClass} ${isChecked ? 'is-locked' : ''}" data-choice="${oi}">
            <span class="quiz__choice-indicator">
              ${q.type === 'multi' ? `
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                  ${selected ? '<rect width="14" height="14" rx="3" fill="var(--accent-blue)"/><path d="M3.5 7l2.5 2.5L10.5 4" stroke="white" stroke-width="1.5" stroke-linecap="round"/>' : '<rect x="0.5" y="0.5" width="13" height="13" rx="2.5" stroke="var(--border-medium)"/>'}
                </svg>
              ` : `
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                  ${selected ? '<circle cx="7" cy="7" r="7" fill="var(--accent-blue)"/><circle cx="7" cy="7" r="3" fill="white"/>' : '<circle cx="7" cy="7" r="6" stroke="var(--border-medium)" stroke-width="1.5"/>'}
                </svg>
              `}
            </span>
            <span class="quiz__choice-text">${opt}</span>
          </label>
        `;
      }).join('')}
    </div>
  `;
}

function renderTextInput(q, index, state) {
  const isChecked = state.checked[index];
  // Escape HTML to prevent XSS
  const escapeHtml = (str) => {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  };

  return `
    <div class="quiz__text-input">
      <textarea
        class="quiz__textarea"
        data-question="${index}"
        placeholder="${q.type === 'scenario' ? 'Describe your approach...' : 'Type your answer...'}"
        rows="${q.type === 'scenario' ? 5 : 3}"
        ${isChecked ? 'disabled' : ''}
      >${escapeHtml(state.answers[index])}</textarea>
      ${isChecked ? `
        <div class="quiz__model-answer">
          <strong style="color: var(--accent-primary);">Model Answer:</strong>
          ${formatModelAnswer(q.correct)}
        </div>
      ` : ''}
    </div>
  `;
}

function formatModelAnswer(text) {
  // Check if text contains numbered points like (1), (2), (3) or 1., 2., 3.
  const numberedPattern = /\((\d+)\)\s+/g;
  const dotPattern = /(\d+)\.\s+/g;

  if (numberedPattern.test(text)) {
    // Split by (1), (2), etc.
    const parts = text.split(/\((\d+)\)\s+/).filter(p => p.trim() && !/^\d+$/.test(p));
    if (parts.length > 1) {
      const items = parts.map(part => `<li>${part.trim()}</li>`).join('');
      return `<ol>${items}</ol>`;
    }
  } else if (dotPattern.test(text)) {
    // Split by 1., 2., etc.
    const parts = text.split(/\d+\.\s+/).filter(p => p.trim());
    if (parts.length > 1) {
      const items = parts.map(part => `<li>${part.trim()}</li>`).join('');
      return `<ol>${items}</ol>`;
    }
  }

  // Check for bullet points (-, *, •)
  const lines = text.split('\n');
  const bulletPattern = /^[\-\*\•]\s+/;
  const bulletLines = lines.filter(line => bulletPattern.test(line.trim()));

  if (bulletLines.length > 0) {
    const formatted = lines.map(line => {
      const trimmed = line.trim();
      if (bulletPattern.test(trimmed)) {
        return `<li>${trimmed.replace(bulletPattern, '')}</li>`;
      } else if (trimmed) {
        return `<p>${trimmed}</p>`;
      }
      return '';
    }).filter(l => l).join('');

    return formatted.includes('<li>') ? formatted.replace(/(<li>.*<\/li>)+/g, match => `<ul>${match}</ul>`) : `<p>${text}</p>`;
  }

  // No special formatting needed
  return `<p>${text}</p>`;
}

function attachChoiceListeners(questions, state, container, rerender, persistState) {
  container.querySelectorAll('.quiz__choice:not(.is-locked)').forEach(choice => {
    choice.addEventListener('click', () => {
      const questionEl = choice.closest('.quiz__question');
      const qi = parseInt(questionEl.dataset.question);
      if (state.checked[qi]) return;

      const ci = parseInt(choice.dataset.choice);
      const q = questions[qi];

      if (q.type === 'multi') {
        if (!state.answers[qi]) state.answers[qi] = [];
        const idx = state.answers[qi].indexOf(ci);
        if (idx > -1) state.answers[qi].splice(idx, 1);
        else state.answers[qi].push(ci);
      } else {
        state.answers[qi] = ci;
      }

      // Update visual selection
      const choices = questionEl.querySelectorAll('.quiz__choice');
      choices.forEach((c, i) => {
        if (q.type === 'multi') {
          c.classList.toggle('is-selected', (state.answers[qi] || []).includes(i));
        } else {
          c.classList.toggle('is-selected', state.answers[qi] === i);
        }
      });

      // Re-render indicators
      choices.forEach((c, i) => {
        const indicator = c.querySelector('.quiz__choice-indicator');
        const selected = q.type === 'multi'
          ? (state.answers[qi] || []).includes(i)
          : state.answers[qi] === i;

        if (q.type === 'multi') {
          indicator.innerHTML = `<svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            ${selected ? '<rect width="14" height="14" rx="3" fill="var(--accent-blue)"/><path d="M3.5 7l2.5 2.5L10.5 4" stroke="white" stroke-width="1.5" stroke-linecap="round"/>' : '<rect x="0.5" y="0.5" width="13" height="13" rx="2.5" stroke="var(--border-medium)"/>'}
          </svg>`;
        } else {
          indicator.innerHTML = `<svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            ${selected ? '<circle cx="7" cy="7" r="7" fill="var(--accent-blue)"/><circle cx="7" cy="7" r="3" fill="white"/>' : '<circle cx="7" cy="7" r="6" stroke="var(--border-medium)" stroke-width="1.5"/>'}
          </svg>`;
        }
      });

      // Save selection (checked later via Check Answer button)
      persistState();
    });
  });

  // Check buttons for all question types
  container.querySelectorAll('.quiz__action-btn--check[data-check]').forEach(btn => {
    btn.addEventListener('click', () => {
      const qi = parseInt(btn.dataset.check);
      const q = questions[qi];
      console.log(`[Quiz] Check Q${qi + 1}: type=${q?.type}, answer=${JSON.stringify(state.answers[qi])}, correct=${JSON.stringify(q?.correct)}`);
      state.checked[qi] = true;
      persistState();
      rerender();
    });
  });
}

function attachTextListeners(questions, state, container, rerender, persistState) {
  container.querySelectorAll('.quiz__textarea').forEach(textarea => {
    let debounceTimer;
    textarea.addEventListener('input', () => {
      const qi = parseInt(textarea.dataset.question);
      state.answers[qi] = textarea.value;
      // Debounce persistence to avoid excessive writes
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => persistState(), 500);
    });
  });
}

function checkAnswer(q, answer, selfGraded) {
  if (q.type === 'mc') {
    if (answer === undefined || answer === null) return false;
    return answer === q.correct;
  }
  if (q.type === 'multi') {
    if (!Array.isArray(answer) || answer.length === 0) return false;
    return q.correct.length === answer.length &&
      q.correct.every(c => answer.includes(c));
  }
  if (q.type === 'short' || q.type === 'scenario') {
    // For open-ended questions, correctness depends entirely on self-grading
    return selfGraded === true;
  }
  return false;
}

function showToast(message) {
  const existing = document.getElementById('quiz-toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.id = 'quiz-toast';
  toast.textContent = message;
  toast.style.cssText = `
    position: fixed; bottom: 24px; right: 24px; z-index: 9999;
    background: var(--accent-primary, #F0B429); color: var(--bg-deep, #0C0A09);
    padding: 12px 20px; border-radius: 8px;
    font-family: var(--font-heading); font-weight: 600; font-size: 14px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    animation: toastIn 0.4s ease forwards;
  `;

  if (!document.getElementById('toast-styles')) {
    const style = document.createElement('style');
    style.id = 'toast-styles';
    style.textContent = `
      @keyframes toastIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
      @keyframes toastOut { from { opacity: 1; transform: translateY(0); } to { opacity: 0; transform: translateY(20px); } }
    `;
    document.head.appendChild(style);
  }

  document.body.appendChild(toast);
  setTimeout(() => {
    toast.style.animation = 'toastOut 0.4s ease forwards';
    setTimeout(() => toast.remove(), 400);
  }, 3000);
}
