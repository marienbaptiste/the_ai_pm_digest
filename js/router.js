export class Router {
  constructor() {
    this.routes = new Map();
    this.currentRoute = null;
    this.onNavigate = null;
    window.addEventListener('hashchange', () => this._handleRoute());
  }

  on(pattern, handler) {
    this.routes.set(pattern, handler);
    return this;
  }

  start() {
    this._handleRoute();
  }

  navigate(hash) {
    window.location.hash = hash;
  }

  _handleRoute() {
    const rawHash = window.location.hash;
    const hash = rawHash && rawHash !== '#' ? rawHash : '#/';
    let matched = false;

    for (const [pattern, handler] of this.routes) {
      const params = this._match(pattern, hash);
      if (params !== null) {
        this.currentRoute = { pattern, hash, params };
        if (this.onNavigate) this.onNavigate(this.currentRoute);
        handler(params);
        matched = true;
        break;
      }
    }

    if (!matched) {
      this.navigate('#/');
    }
  }

  _match(pattern, hash) {
    // Convert pattern like '#/module/:moduleId/lesson/:lessonId' to regex
    const paramNames = [];
    const regexStr = pattern.replace(/:([^/]+)/g, (_, name) => {
      paramNames.push(name);
      return '([^/]+)';
    });

    const regex = new RegExp(`^${regexStr}$`);
    const match = hash.match(regex);

    if (!match) return null;

    const params = {};
    paramNames.forEach((name, i) => {
      params[name] = decodeURIComponent(match[i + 1]);
    });
    return params;
  }
}
