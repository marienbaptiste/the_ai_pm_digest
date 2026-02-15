import { storage } from '../storage.js';
import { renderSidebar } from './sidebar.js';

export function renderQuiz(container, quizData, lessonKey) {
  if (!quizData || !quizData.questions || quizData.questions.length === 0) {
    container.innerHTML = `
      <div class="quiz-placeholder">
        <p>No quiz available for this lesson yet.</p>
      </div>
    `;
    return;
  }

  // Per-question checked state
  const state = {
    answers: {},
    checked: {},       // which questions have been checked
    selfGraded: {},    // self-grading for open-ended questions: { [index]: true/false }
  };

  const totalQ = quizData.questions.length;

  function getScore() {
    let correct = 0;
    quizData.questions.forEach((q, i) => {
      if (state.checked[i] && checkAnswer(q, state.answers[i], state.selfGraded[i])) correct++;
    });
    return { correct, total: totalQ, percent: Math.round((correct / totalQ) * 100) };
  }

  function allChecked() {
    if (Object.keys(state.checked).length !== totalQ) return false;
    // Open-ended questions must also be self-graded
    for (let i = 0; i < totalQ; i++) {
      const q = quizData.questions[i];
      if ((q.type === 'short' || q.type === 'scenario') && state.selfGraded[i] === undefined) {
        return false;
      }
    }
    return true;
  }

  function render() {
    const score = getScore();
    const isComplete = allChecked();
    const prevScore = storage.getQuizScore(lessonKey);
    const isAlreadyCompleted = storage.isLessonCompleted(lessonKey);

    // Save progress when all answered and score >= 70
    console.log(`[Quiz] isComplete=${isComplete}, score=${score.percent}%, isAlreadyCompleted=${isAlreadyCompleted}, checked=${Object.keys(state.checked).length}/${totalQ}`);
    if (isComplete && score.percent >= 70 && !isAlreadyCompleted) {
      storage.markLessonComplete(lessonKey, score.percent);
      renderSidebar();
      // Update breadcrumb tag
      const breadcrumb = document.querySelector('.lesson__breadcrumb');
      if (breadcrumb && !breadcrumb.querySelector('.tag--primary')) {
        breadcrumb.insertAdjacentHTML('beforeend',
          '<span class="tag tag--primary" style="margin-left: 8px;">Completed</span>'
        );
      }
      // Show save confirmation toast
      showToast(`Lesson completed! Score: ${score.percent}%`);
    }

    container.innerHTML = `
      <div class="quiz animate-in">
        <div class="quiz__header">
          <div class="quiz__header-icon">\u{1F9EA}</div>
          <div style="flex: 1;">
            <h3 class="quiz__title">Knowledge Check</h3>
            <p class="quiz__subtitle">${isComplete
              ? `Score: ${score.correct}/${score.total} (${score.percent}%) ${score.percent >= 70 ? '— Passed!' : '— Need 70% to pass.'}`
              : `Answer each question — feedback is instant. Need 70% to complete.`
            }</p>
          </div>
          <div class="quiz__progress-ring">
            <span class="quiz__progress-count">${Object.keys(state.checked).length}/${totalQ}</span>
          </div>
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
    attachChoiceListeners(quizData.questions, state, container, render);

    // Attach text input listeners
    attachTextListeners(quizData.questions, state, container, render);

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
        state.selfGraded[qi] = grade === 'correct';
        render();
      });
    });

    // Retry button
    document.getElementById('quiz-retry')?.addEventListener('click', () => {
      state.answers = {};
      state.checked = {};
      state.selfGraded = {};
      render();
    });

    // Reset lesson button
    document.getElementById('quiz-reset-lesson')?.addEventListener('click', () => {
      storage.resetLesson(lessonKey);
      state.answers = {};
      state.checked = {};
      state.selfGraded = {};
      render();
      renderSidebar();
      // Remove completed tag from breadcrumb
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
  const isIncorrect = isChecked && !needsSelfGrade && !isCorrect;
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
      </div>
      <p class="quiz__question-text">${q.question}</p>

      ${q.type === 'mc' || q.type === 'multi' ? renderChoices(q, index, state) : ''}
      ${q.type === 'short' || q.type === 'scenario' ? renderTextInput(q, index, state) : ''}

      ${!isChecked && (q.type === 'multi') ? `
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
            <p>${q.explanation}</p>
          </div>
        </div>
        ${q.expertNote ? `
          <button class="quiz__expert-toggle">
            <span>\u{1F9E0} Expert Insight</span>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M4 6l4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
          </button>
          <div class="quiz__expert-content" style="max-height: 0px;">
            <p>${q.expertNote}</p>
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

  return `
    <div class="quiz__text-input">
      <textarea
        class="quiz__textarea"
        data-question="${index}"
        placeholder="${q.type === 'scenario' ? 'Describe your approach...' : 'Type your answer...'}"
        rows="${q.type === 'scenario' ? 5 : 3}"
        ${isChecked ? 'disabled' : ''}
      >${state.answers[index] || ''}</textarea>
      ${isChecked ? `
        <div class="quiz__model-answer">
          <strong style="color: var(--accent-primary);">Model Answer:</strong>
          <p>${q.correct}</p>
        </div>
      ` : ''}
    </div>
  `;
}

function attachChoiceListeners(questions, state, container, rerender) {
  container.querySelectorAll('.quiz__choice:not(.is-locked)').forEach(choice => {
    choice.addEventListener('click', () => {
      const questionEl = choice.closest('.quiz__question');
      const qi = parseInt(questionEl.dataset.question);
      if (state.checked[qi]) return; // Already checked

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

      // For mc (single-choice), auto-check immediately
      if (q.type === 'mc') {
        state.checked[qi] = true;
        rerender();
      }
    });
  });

  // Check buttons for multi-select questions
  container.querySelectorAll('.quiz__action-btn--check').forEach(btn => {
    btn.addEventListener('click', () => {
      const qi = parseInt(btn.dataset.check);
      state.checked[qi] = true;
      rerender();
    });
  });
}

function attachTextListeners(questions, state, container, rerender) {
  container.querySelectorAll('.quiz__textarea').forEach(textarea => {
    textarea.addEventListener('input', () => {
      const qi = parseInt(textarea.dataset.question);
      state.answers[qi] = textarea.value;
    });
  });
}

function checkAnswer(q, answer, selfGraded) {
  if (answer === undefined || answer === null) return false;

  if (q.type === 'mc') {
    return answer === q.correct;
  }
  if (q.type === 'multi') {
    if (!Array.isArray(answer)) return false;
    return q.correct.length === answer.length &&
      q.correct.every(c => answer.includes(c));
  }
  // For short/scenario, use self-grading result
  if (q.type === 'short' || q.type === 'scenario') {
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

  // Add animation keyframes if not already present
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
