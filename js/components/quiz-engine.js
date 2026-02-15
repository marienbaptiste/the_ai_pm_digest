import { storage } from '../storage.js';

export function renderQuiz(container, quizData, lessonKey) {
  if (!quizData || !quizData.questions || quizData.questions.length === 0) {
    container.innerHTML = `
      <div class="quiz-placeholder">
        <p>No quiz available for this lesson yet.</p>
      </div>
    `;
    return;
  }

  const state = {
    answers: {},
    submitted: false,
    score: 0
  };

  function render() {
    container.innerHTML = `
      <div class="quiz animate-in">
        <div class="quiz__header">
          <div class="quiz__header-icon">\u{1F9EA}</div>
          <div>
            <h3 class="quiz__title">Knowledge Check</h3>
            <p class="quiz__subtitle">Test your understanding. You need 70% to complete this lesson.</p>
          </div>
        </div>

        <div class="quiz__questions">
          ${quizData.questions.map((q, i) => renderQuestion(q, i, state)).join('')}
        </div>

        ${!state.submitted ? `
          <div class="quiz__actions">
            <button class="btn btn--warm btn--lg quiz__submit" id="quiz-submit">
              Submit Answers
            </button>
          </div>
        ` : `
          <div class="quiz__results animate-in">
            <div class="quiz__score-card ${state.score >= 70 ? 'quiz__score-card--pass' : 'quiz__score-card--fail'}">
              <div class="quiz__score-value">${state.score}%</div>
              <div class="quiz__score-label">
                ${state.score >= 70 ? 'Passed! Great work.' : 'Not quite \u2014 review the explanations and try again.'}
              </div>
              ${state.score >= 70 ? `
                <p style="color: var(--text-secondary); font-size: var(--text-sm); margin-top: var(--space-3);">
                  This lesson has been marked as complete.
                </p>
              ` : `
                <button class="btn btn--secondary quiz__retry" id="quiz-retry" style="margin-top: var(--space-4);">
                  Try Again
                </button>
              `}
            </div>
          </div>
        `}
      </div>
    `;

    // Attach event listeners
    if (!state.submitted) {
      attachQuestionListeners(quizData.questions, state);
      document.getElementById('quiz-submit')?.addEventListener('click', () => {
        submitQuiz(container, quizData, lessonKey, state);
      });
    } else {
      document.getElementById('quiz-retry')?.addEventListener('click', () => {
        state.answers = {};
        state.submitted = false;
        state.score = 0;
        render();
      });

      // Attach expert note toggles
      container.querySelectorAll('.quiz__expert-toggle').forEach(btn => {
        btn.addEventListener('click', () => {
          const content = btn.nextElementSibling;
          const isOpen = content.style.maxHeight !== '0px' && content.style.maxHeight;
          content.style.maxHeight = isOpen ? '0px' : content.scrollHeight + 'px';
          btn.classList.toggle('is-open');
        });
      });
    }
  }

  render();
}

function renderQuestion(q, index, state) {
  const answered = state.answers[index] !== undefined;
  const isCorrect = state.submitted && checkAnswer(q, state.answers[index]);
  const isIncorrect = state.submitted && answered && !isCorrect;
  const stateClass = state.submitted ? (isCorrect ? 'quiz-correct' : (answered ? 'quiz-incorrect' : '')) : '';

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
      </div>
      <p class="quiz__question-text">${q.question}</p>

      ${q.type === 'mc' || q.type === 'multi' ? renderChoices(q, index, state) : ''}
      ${q.type === 'short' || q.type === 'scenario' ? renderTextInput(q, index, state) : ''}

      ${state.submitted ? `
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
        if (state.submitted) {
          if (isCorrectChoice) choiceClass = 'quiz__choice--correct';
          else if (selected && !isCorrectChoice) choiceClass = 'quiz__choice--incorrect';
        }

        return `
          <label class="quiz__choice ${selected ? 'is-selected' : ''} ${choiceClass}" data-choice="${oi}">
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
  return `
    <div class="quiz__text-input">
      <textarea
        class="quiz__textarea"
        data-question="${index}"
        placeholder="${q.type === 'scenario' ? 'Describe your approach...' : 'Type your answer...'}"
        rows="${q.type === 'scenario' ? 5 : 3}"
        ${state.submitted ? 'disabled' : ''}
      >${state.answers[index] || ''}</textarea>
      ${state.submitted ? `
        <div class="quiz__model-answer">
          <strong style="color: var(--accent-primary);">Model Answer:</strong>
          <p>${q.correct}</p>
        </div>
      ` : ''}
    </div>
  `;
}

function attachQuestionListeners(questions, state) {
  // Choice selection
  document.querySelectorAll('.quiz__choice').forEach(choice => {
    choice.addEventListener('click', () => {
      const questionEl = choice.closest('.quiz__question');
      const qi = parseInt(questionEl.dataset.question);
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
    });
  });

  // Text input
  document.querySelectorAll('.quiz__textarea').forEach(textarea => {
    textarea.addEventListener('input', () => {
      const qi = parseInt(textarea.dataset.question);
      state.answers[qi] = textarea.value;
    });
  });
}

function checkAnswer(q, answer) {
  if (answer === undefined || answer === null) return false;

  if (q.type === 'mc') {
    return answer === q.correct;
  }
  if (q.type === 'multi') {
    if (!Array.isArray(answer)) return false;
    return q.correct.length === answer.length &&
      q.correct.every(c => answer.includes(c));
  }
  // For short/scenario, always mark as "reviewed" (self-graded)
  if (q.type === 'short' || q.type === 'scenario') {
    return typeof answer === 'string' && answer.trim().length > 10;
  }
  return false;
}

function submitQuiz(container, quizData, lessonKey, state) {
  const questions = quizData.questions;
  let correct = 0;

  questions.forEach((q, i) => {
    if (checkAnswer(q, state.answers[i])) correct++;
  });

  state.score = Math.round((correct / questions.length) * 100);
  state.submitted = true;

  if (state.score >= 70) {
    storage.markLessonComplete(lessonKey, state.score);
  }

  // Re-render quiz with results
  renderQuiz(container, quizData, lessonKey);

  // Scroll to results
  setTimeout(() => {
    const results = container.querySelector('.quiz__results');
    if (results) results.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, 100);
}
