const STORAGE_KEY = 'aipm_digest_progress';

function getStore() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch (e) {
    console.warn('[Storage] Failed to read localStorage:', e);
  }
  return { progress: {}, streakDays: [], lastVisit: null };
}

function saveStore(store) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
  } catch (e) {
    console.error('[Storage] Failed to save to localStorage:', e);
  }
}

export const storage = {
  getLessonProgress(lessonKey) {
    const store = getStore();
    return store.progress[lessonKey] || null;
  },

  markLessonComplete(lessonKey, quizScore) {
    console.log(`[Storage] Marking lesson complete: ${lessonKey}, score: ${quizScore}`);
    const store = getStore();
    store.progress[lessonKey] = {
      completed: true,
      quizScore,
      timestamp: new Date().toISOString()
    };
    saveStore(store);
    this._updateStreak();
    // Verify save worked
    const verify = getStore();
    if (verify.progress[lessonKey]?.completed) {
      console.log(`[Storage] Save verified for ${lessonKey}`);
    } else {
      console.error(`[Storage] Save FAILED for ${lessonKey} — localStorage may be unavailable`);
    }
    window.dispatchEvent(new CustomEvent('progress-changed', { detail: { lessonKey, quizScore } }));
  },

  isLessonCompleted(lessonKey) {
    const store = getStore();
    return store.progress[lessonKey]?.completed === true;
  },

  getQuizScore(lessonKey) {
    const store = getStore();
    return store.progress[lessonKey]?.quizScore ?? null;
  },

  getOverallProgress(modules) {
    const store = getStore();
    let totalLessons = 0;
    let completedLessons = 0;

    for (const mod of modules) {
      for (const lesson of mod.lessons) {
        totalLessons++;
        const key = `${mod.id}-${lesson.id}`;
        if (store.progress[key]?.completed) {
          completedLessons++;
        }
      }
    }

    return {
      total: totalLessons,
      completed: completedLessons,
      percent: totalLessons > 0 ? Math.round((completedLessons / totalLessons) * 100) : 0
    };
  },

  getModuleProgress(moduleObj) {
    const store = getStore();
    let total = moduleObj.lessons.length;
    let completed = 0;

    for (const lesson of moduleObj.lessons) {
      const key = `${moduleObj.id}-${lesson.id}`;
      if (store.progress[key]?.completed) completed++;
    }

    return {
      total,
      completed,
      percent: total > 0 ? Math.round((completed / total) * 100) : 0
    };
  },

  getStreak() {
    const store = getStore();
    return store.streakDays?.length || 0;
  },

  _updateStreak() {
    const store = getStore();
    const today = new Date().toISOString().split('T')[0];

    if (!store.streakDays) store.streakDays = [];

    if (store.streakDays.includes(today)) {
      return; // Already recorded today, don't re-save
    }

    const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0];

    if (store.streakDays.length === 0 || store.streakDays[store.streakDays.length - 1] === yesterday) {
      store.streakDays.push(today);
    } else if (store.streakDays[store.streakDays.length - 1] !== today) {
      store.streakDays = [today];
    }

    store.lastVisit = today;
    saveStore(store);
  },

  getAllProgress() {
    return getStore().progress;
  },

  resetLesson(lessonKey) {
    console.log(`[Storage] Resetting lesson: ${lessonKey}`);
    const store = getStore();
    delete store.progress[lessonKey];
    saveStore(store);
    window.dispatchEvent(new CustomEvent('progress-changed', { detail: { lessonKey, reset: true } }));
  },

  resetAll() {
    localStorage.removeItem(STORAGE_KEY);
    window.dispatchEvent(new CustomEvent('progress-changed', { detail: { reset: true } }));
  }
};
