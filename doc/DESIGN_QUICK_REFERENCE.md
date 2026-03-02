# Design System Quick Reference

> Fast lookup for MedCHR.ai's Anthropic-inspired design tokens

---

## Colors (Copy & Paste)

```css
/* Backgrounds */
--bg-primary: #F5F3EE;    /* Main page background */
--bg-secondary: #FAF9F6;  /* Header, card backgrounds */
--bg-tertiary: #EFEEE9;   /* Hover states */
--bg-white: #FFFFFF;      /* Emphasis cards */

/* Text */
--text-primary: #191919;    /* Main text */
--text-secondary: #4A4A4A;  /* Secondary text */
--text-muted: #7A7A7A;      /* Muted text */

/* Accents */
--accent-primary: #D97757;   /* Buttons, links, primary actions */
--accent-hover: #CC6A4A;     /* Hover state */
--accent-lighter: #FBF4F1;   /* Light backgrounds */

/* Semantic */
--success: #2D7A54;     --success-light: #E6F3ED;
--warning: #CC8B2C;     --warning-light: #FDF6E8;
--error: #C74545;       --error-light: #FCF0F0;
--info: #4A7BA7;        --info-light: #EBF2F8;

/* Borders */
--border-primary: #D9D6D0;
--border-secondary: #E8E6E0;
```

---

## Spacing

```css
4px   8px   16px  24px  32px  48px  64px  96px
xs    sm    md    lg    xl    2xl   3xl   4xl
```

**Common uses:**
- Button padding: `md xl` (16px 32px)
- Card padding: `xl` (32px)
- Section margins: `2xl` (48px)
- Input padding: `md lg` (16px 24px)

---

## Typography

```css
/* Sizes (responsive) */
xs:   12-13px   /* Badges, captions */
sm:   14-15px   /* UI elements, labels */
base: 16-17px   /* Body text */
md:   17-18px   /* Subheadings */
lg:   20-23px   /* H3, large UI */
xl:   24-28px   /* H2 */
2xl:  32-40px   /* H1 */

/* Weights */
400  Normal body text
500  Medium UI elements
600  Semibold headings, buttons, labels
700  Bold emphasis, H1

/* Line heights */
1.2    Headings
1.5    Body text
1.625  Long-form content
```

---

## Components

### Button
```html
<button class="btn-primary">Primary Action</button>
<button class="btn-secondary">Secondary Action</button>
<button class="btn-ghost">Ghost Action</button>
```

### Card
```html
<div class="card">
  <div class="card-header">
    <h3 class="card-title">Title</h3>
  </div>
  <p>Content</p>
</div>
```

### Badge
```html
<span class="badge badge-success">Success</span>
<span class="badge badge-warning">Warning</span>
<span class="badge badge-error">Error</span>
<span class="badge badge-info">Info</span>
```

### Form
```html
<label>Label Text</label>
<input type="text" placeholder="Placeholder">
```

---

## Shadows

```css
xs:  0 1px 2px    /* Cards at rest */
sm:  0 1px 3px    /* Subtle elevation */
md:  0 4px 8px    /* Hover states */
lg:  0 8px 16px   /* Floating elements */
xl:  0 12px 24px  /* Modals */
```

---

## Border Radius

```css
sm:   6px     /* Small elements */
md:   10px    /* Inputs, small cards */
lg:   14px    /* Cards, main components */
xl:   18px    /* Large cards */
full: 9999px  /* Pills, badges, buttons */
```

---

## Utility Classes

```css
/* Text */
.text-primary    .text-secondary    .text-muted
.text-success    .text-warning      .text-error
.text-center     .text-left         .text-right

/* Spacing */
.mt-sm  .mt-md  .mt-lg
.mb-sm  .mb-md  .mb-lg

/* Flex */
.flex  .flex-col  .items-center  .justify-between
.gap-sm  .gap-md  .gap-lg

/* Layout */
.grid  .row  .card  .badge
```

---

## Transitions

```css
fast: 150ms    /* Hover, simple changes */
base: 250ms    /* Default animations */
slow: 400ms    /* Complex animations */

/* Easing: cubic-bezier(0.16, 1, 0.3, 1) */
```

---

## Accessibility

**Focus states:** 2px outline, 3px offset
**Contrast:** All combinations WCAG AA+
**Motion:** Respects prefers-reduced-motion

---

## Common Patterns

### Section Header
```html
<h2 class="mb-lg">Section Title</h2>
<p class="text-secondary mb-xl">Description text</p>
```

### Action Row
```html
<div class="flex justify-between items-center mb-lg">
  <h3>Title</h3>
  <button class="btn-primary">Action</button>
</div>
```

### Status Indicator
```html
<span class="badge badge-success">Active</span>
<span class="badge badge-warning">Pending</span>
<span class="badge badge-error">Inactive</span>
```

### Form Group
```html
<div class="mb-md">
  <label>Field Label</label>
  <input type="text" placeholder="Enter value">
</div>
```

---

## Responsive Breakpoints

```css
@media (max-width: 768px) { /* Tablet */ }
@media (max-width: 640px) { /* Mobile */ }
```

**Mobile-first approach:** Design for mobile, enhance for desktop

---

## File Location

`/frontend/templates/base.html` - All design tokens defined in `:root`

---

**Pro Tips:**
1. Use CSS custom properties: `color: var(--accent-primary);`
2. Combine utility classes: `class="flex gap-md items-center"`
3. Fluid spacing: Use clamp() for responsive padding
4. Semantic HTML: Use proper tags (header, main, article, etc.)
5. Test focus states: Always verify keyboard navigation

---

**Need more details?** See `/doc/DESIGN_SYSTEM.md`
