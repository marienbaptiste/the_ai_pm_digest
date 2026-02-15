# The AI PM Digest

Training platform for the Google DeepMind Product Manager role. Static SPA with no framework — vanilla HTML/CSS/JS with ES modules.

## Project Structure

- `index.html` — App shell, served via `npx serve . -p 3000 -s`
- `css/` — Design system (variables, base, layout, components, animations, lesson, quiz)
- `js/` — SPA router, storage, components (sidebar, dashboard, lesson-viewer, glossary, quiz-engine, progress-ring), animated SVG diagrams
- `data/` — Module registry, 147-term glossary, 12 lesson content files (51 lessons total)

## Dev Server

```
npx serve . -p 3000 -s
```

Then open http://localhost:3000

## Frontend Design Skill

<frontend_aesthetics>
You tend to converge toward generic, "on distribution" outputs. In frontend design, this creates what users call the "AI slop" aesthetic. Avoid this: make creative, distinctive frontends that surprise and delight.

Focus on:
- Typography: Choose fonts that are beautiful, unique, and interesting. Avoid generic fonts like Arial and Inter; opt instead for distinctive choices that elevate the frontend's aesthetics.
- Color & Theme: Commit to a cohesive aesthetic. Use CSS variables for consistency. Dominant colors with sharp accents outperform timid, evenly-distributed palettes. Draw from IDE themes and cultural aesthetics for inspiration.
- Motion: Use animations for effects and micro-interactions. Prioritize CSS-only solutions for HTML. Use Motion library for React when available. Focus on high-impact moments: one well-orchestrated page load with staggered reveals (animation-delay) creates more delight than scattered micro-interactions.
- Backgrounds: Create atmosphere and depth rather than defaulting to solid colors. Layer CSS gradients, use geometric patterns, or add contextual effects that match the overall aesthetic.

Avoid generic AI-generated aesthetics:
- Overused font families (Inter, Roboto, Arial, system fonts)
- Clichéd color schemes (particularly purple gradients on white backgrounds)
- Predictable layouts and component patterns
- Cookie-cutter design that lacks context-specific character

Interpret creatively and make unexpected choices that feel genuinely designed for the context. Vary between light and dark themes, different fonts, different aesthetics. You still tend to converge on common choices (Space Grotesk, for example) across generations. Avoid this: it is critical that you think outside the box!
</frontend_aesthetics>

## Current Design Choices

- **Headings**: Syne — angular, futuristic, distinctive
- **Body**: Newsreader — elegant editorial serif for long-form reading
- **Mono**: Fira Code — ligatures for code/formulas
- **Palette**: Warm obsidian (#0C0A09) with amber (#F0B429) primary accent, coral (#E8553A) warm accent, sky blue (#7EB8DA) links, lavender (#C4A7E7) expert content
- **Background**: Geometric grid overlay with radial color orbs, warm dark theme
