# The AI PM Digest

Training platform for AI Product Management. Static SPA with no framework - vanilla HTML/CSS/JS with ES modules.

## Project Structure

- `index.html` - App shell, served via `npx serve . -p 3000 -s`
- `css/` - Design system (variables, base, layout, components, animations, lesson, quiz)
- `js/` - SPA router, storage, components (sidebar, dashboard, lesson-viewer, glossary, quiz-engine, progress-ring), animated SVG diagrams
- `data/` - Module registry, 147-term glossary, 12 lesson content files (51 lessons total)

## Dev Server

```
npx serve . -p 3000 -s
```

Then open http://localhost:3000
