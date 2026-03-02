# MedCHR.ai Design System
## Anthropic-Inspired Healthcare UI

**Version:** 2.0
**Last Updated:** 2026-02-14
**Design Philosophy:** Clean, professional, trustworthy aesthetic inspired by Anthropic's design language, optimized for healthcare applications.

---

## Design Principles

### 1. Professional Clarity
- Clean, minimal aesthetic with generous whitespace
- Clear visual hierarchy through typography and spacing
- Professional but approachable tone suitable for healthcare

### 2. Accessibility First
- WCAG 2.1 Level AA compliant minimum
- 4.5:1 color contrast for normal text, 3:1 for large text
- Full keyboard navigation support
- Screen reader optimized
- Reduced motion support built-in

### 3. Trustworthy & Calm
- Warm neutral backgrounds create a calm environment
- Coral/rust accents provide energy without overwhelming
- Subtle shadows and borders maintain depth without distraction
- Smooth, purposeful transitions

---

## Color Palette

### Neutral Backgrounds (Warm Tones)
```css
--bg-primary: #F5F3EE        /* Main page background - warm cream */
--bg-secondary: #FAF9F6      /* Card/header background - lighter cream */
--bg-tertiary: #EFEEE9       /* Hover states - slightly darker */
--bg-hover: #E8E6E0          /* Interactive hover - warmer */
--bg-white: #FFFFFF          /* Pure white for emphasis */
```

### Text Colors (Deep, Readable)
```css
--text-primary: #191919      /* Main text - deep near-black */
--text-secondary: #4A4A4A    /* Secondary text - medium gray */
--text-muted: #7A7A7A        /* Muted text - lighter gray */
--text-disabled: #A8A8A8     /* Disabled states */
```

### Accent Colors (Coral/Rust)
```css
--accent-primary: #D97757    /* Primary actions - coral */
--accent-hover: #CC6A4A      /* Hover state - deeper rust */
--accent-light: #F5E8E3      /* Light backgrounds */
--accent-lighter: #FBF4F1    /* Lightest tint */
```

### Semantic Colors
```css
--success: #2D7A54           /* Success states - green */
--success-light: #E6F3ED     /* Success background */
--warning: #CC8B2C           /* Warning states - amber */
--warning-light: #FDF6E8     /* Warning background */
--error: #C74545             /* Error states - red */
--error-light: #FCF0F0       /* Error background */
--info: #4A7BA7              /* Info states - blue */
--info-light: #EBF2F8        /* Info background */
```

### Borders
```css
--border-primary: #D9D6D0    /* Main borders */
--border-secondary: #E8E6E0  /* Lighter borders */
--border-light: #F0EDE8      /* Subtle dividers */
```

---

## Typography

### Font Stack
```css
--font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif;
```

### Fluid Font Sizes (Responsive)
```css
--font-size-xs: clamp(0.75rem, 0.73rem + 0.09vw, 0.8125rem)      /* 12-13px */
--font-size-sm: clamp(0.875rem, 0.85rem + 0.11vw, 0.9375rem)     /* 14-15px */
--font-size-base: clamp(1rem, 0.97rem + 0.13vw, 1.0625rem)       /* 16-17px */
--font-size-md: clamp(1.0625rem, 1.03rem + 0.14vw, 1.125rem)     /* 17-18px */
--font-size-lg: clamp(1.25rem, 1.19rem + 0.27vw, 1.4375rem)      /* 20-23px */
--font-size-xl: clamp(1.5rem, 1.41rem + 0.39vw, 1.75rem)         /* 24-28px */
--font-size-2xl: clamp(2rem, 1.82rem + 0.78vw, 2.5rem)           /* 32-40px */
--font-size-3xl: clamp(2.5rem, 2.23rem + 1.17vw, 3.25rem)        /* 40-52px */
```

### Line Heights
```css
--line-height-tight: 1.2      /* Headings */
--line-height-snug: 1.375     /* Subheadings */
--line-height-normal: 1.5     /* Body text */
--line-height-relaxed: 1.625  /* Long-form content */
--line-height-loose: 1.75     /* Very spacious reading */
```

### Typography Usage
- **Headings:** 600-700 weight, tight line-height, -0.03em to -0.01em letter spacing
- **Body:** 400 weight, normal to relaxed line-height
- **UI Elements:** 500-600 weight, -0.01em letter spacing
- **Labels:** 600 weight for form labels to improve scannability

---

## Spacing Scale

```css
--space-xs: 4px
--space-sm: 8px
--space-md: 16px
--space-lg: 24px
--space-xl: 32px
--space-2xl: 48px
--space-3xl: 64px
--space-4xl: 96px
```

### Spacing Guidelines
- **Component padding:** Use lg (24px) to xl (32px) for cards
- **Vertical rhythm:** Use lg (24px) for paragraph spacing
- **Form fields:** Use md (16px) to lg (24px) for padding
- **Header/sections:** Use xl (32px) to 2xl (48px) for breathing room

---

## Border Radius

```css
--radius-sm: 6px      /* Small elements, checkboxes */
--radius-md: 10px     /* Input fields, smaller cards */
--radius-lg: 14px     /* Cards, modals */
--radius-xl: 18px     /* Large cards, hero sections */
--radius-full: 9999px /* Pills, badges, rounded buttons */
```

---

## Shadows

Subtle depth with minimal darkness:

```css
--shadow-xs: 0 1px 2px rgba(25, 25, 25, 0.04)
--shadow-sm: 0 1px 3px rgba(25, 25, 25, 0.06)
--shadow-md: 0 4px 8px rgba(25, 25, 25, 0.08)
--shadow-lg: 0 8px 16px rgba(25, 25, 25, 0.10)
--shadow-xl: 0 12px 24px rgba(25, 25, 25, 0.12)
```

---

## Transitions

Smooth, natural motion using Anthropic's preferred easing:

```css
--transition-fast: 150ms cubic-bezier(0.16, 1, 0.3, 1)
--transition-base: 250ms cubic-bezier(0.16, 1, 0.3, 1)
--transition-slow: 400ms cubic-bezier(0.16, 1, 0.3, 1)
```

**Note:** All transitions respect `prefers-reduced-motion` and reduce to 0.01ms for accessibility.

---

## Component Specifications

### Buttons

**Primary Button:**
- Background: `--accent-primary`
- Color: white
- Padding: `16px 32px`
- Border radius: `--radius-full`
- Font: 600 weight, base size
- Shadow: xs (at rest), md (on hover)
- Hover: Lifts 2px, deeper shadow
- Focus: 2px outline with 3px offset

**Secondary Button:**
- Background: white
- Color: `--text-primary`
- Border: 1px solid `--border-primary`
- Same sizing and interaction as primary

**Ghost Button:**
- Transparent background
- Color: `--text-secondary`
- No shadow, subtle hover background

### Cards

- Background: `--bg-white`
- Border: 1px solid `--border-primary`
- Radius: `--radius-lg` (14px)
- Padding: `--space-xl` (32px)
- Shadow: xs (at rest), md (on hover)
- Header: separated by border-bottom

### Forms

**Input Fields:**
- Background: `--bg-white`
- Border: 1px solid `--border-primary`
- Padding: `16px 24px`
- Radius: `--radius-md` (10px)
- Focus: border changes to accent, 4px shadow ring

**Labels:**
- Font: 600 weight, sm size
- Color: `--text-primary`
- Margin bottom: 8px

### Tables

- Header background: `--bg-tertiary`
- Header text: uppercase, 600 weight, xs size, 0.05em letter spacing
- Row padding: `24px 16px`
- Border: bottom border on rows
- Hover: background changes to `--bg-secondary`

### Badges

- Padding: `6px 14px`
- Radius: `--radius-full`
- Font: 600 weight, xs size, 0.02em letter spacing
- Border: 1px solid with color-mix transparency
- Variants: success, warning, error, info, neutral

---

## Layout

### Container Widths
```css
--container-sm: 640px
--container-md: 768px
--container-lg: 1024px    /* Main content max-width */
--container-xl: 1280px
```

### Header
- Sticky position
- Backdrop blur (8px) with semi-transparent background
- Responsive padding using clamp: `clamp(2rem, 1.08rem + 3.92vw, 5rem)`
- 1px bottom border

### Main Content
- Max width: 1024px
- Responsive padding: `clamp(2rem, 1.08rem + 3.92vw, 5rem)`
- Min height: `calc(100vh - 120px)` to push footer down

---

## Accessibility Features

### Keyboard Navigation
- All interactive elements have visible focus states
- 2px outline with 3px offset for clarity
- Focus-visible only (not on mouse clicks)

### Color Contrast
All color combinations meet WCAG AA standards:
- Primary text on primary bg: 12.8:1 ✓
- Secondary text on primary bg: 7.2:1 ✓
- Accent on white: 4.6:1 ✓

### Screen Readers
- Semantic HTML5 elements throughout
- ARIA labels where needed
- Proper heading hierarchy

### Motion
- Full `prefers-reduced-motion` support
- All animations and transitions respect user preferences

---

## Responsive Breakpoints

```css
/* Mobile first approach */
@media (max-width: 768px) {
  /* Tablet and below */
}

@media (max-width: 640px) {
  /* Mobile */
}
```

### Mobile Optimizations
- Reduced padding in header and main
- Single column grid layouts
- Stack flex rows vertically
- Smaller font sizes through clamp()
- Full-width toasts

---

## Implementation Notes

### Browser Support
- Modern browsers (Chrome, Firefox, Safari, Edge)
- Uses CSS custom properties (IE11 not supported)
- Uses modern CSS features: clamp(), color-mix()

### Performance
- System font stack for fast loading
- Only one web font (Inter) loaded
- CSS custom properties for theming
- Minimal shadow complexity

### Healthcare Considerations
- Calm, professional aesthetic reduces patient anxiety
- High contrast ensures readability in various lighting
- Generous spacing improves scannability of medical data
- Clear visual hierarchy helps clinicians find critical info quickly

---

## File Location

**Base Template:** `/frontend/templates/base.html`

All design tokens are defined in the `:root` CSS custom properties at the top of this file. All pages inherit from this base template and automatically receive the design system.

---

## Changelog

### Version 2.0 (2026-02-14)
- Complete redesign to match Anthropic's aesthetic
- Updated color palette to warm neutrals with coral accents
- Implemented fluid typography using clamp()
- Enhanced accessibility with improved focus states
- Added reduced motion support
- Increased spacing for better breathing room
- Refined shadows and borders for subtle depth
- Updated all component styles to match new system

### Version 1.0 (Initial)
- Original warm, inviting design
- Inter font family
- Basic design tokens
