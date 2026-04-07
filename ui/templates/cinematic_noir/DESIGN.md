```markdown
# Design System Document: The Cinematic Editorial

This design system is a high-end framework crafted to transform a standard movie recommendation interface into an immersive, editorial experience. We are moving away from "app-like" layouts and toward a "Cinematic Canvas"—where content is the protagonist and the UI is the sophisticated, invisible stage.

---

### 1. Overview & Creative North Star: "The Digital Curator"
Our Creative North Star is **The Digital Curator**. This system should feel like a premium film magazine brought to life. We avoid the rigid, boxy constraints of traditional grids in favor of **intentional asymmetry** and **tonal depth**. 

To break the "template" look, we utilize aggressive typographic scales and overlapping elements. Headers should bleed into the background, and movie posters should feel like they are floating in a dark, atmospheric void rather than being trapped in a grid.

---

### 2. Colors & Surface Philosophy
The palette is rooted in deep obsidian and charcoal, allowing the vibrant primary red to function as a surgical strike of attention.

*   **Primary (`#ffb4aa` / Container `#e50914`):** Use the red container for high-impact CTAs. 
*   **Neutral Surfaces:** We use a "Deep-to-Surface" hierarchy:
    *   `surface_container_lowest` (#0e0e0e) for the most recessed areas.
    *   `surface` (#131313) for the standard background.
    *   `surface_bright` (#3a3939) for elevated interactive elements.

#### The "No-Line" Rule
**Explicit Instruction:** Do not use 1px solid borders to section off content. In this design system, boundaries are defined strictly through background color shifts. To separate a sidebar from a main feed, transition from `surface` to `surface_container_low`. Hard lines break the cinematic immersion; tonal shifts preserve it.

#### The "Glass & Gradient" Rule
To add "soul" to the interface, use `surface_variant` with a 40% opacity and a 20px backdrop-blur for floating navigation bars or detail overlays. For Hero sections, apply a subtle linear gradient from `primary_container` (at 10% opacity) to `surface` to create a "glow" that anchors the typography.

---

### 3. Typography: Editorial Authority
We use a dual-font strategy to balance character with readability.

*   **Display & Headlines (Manrope):** These are your "vibe" setters. `display-lg` (3.5rem) should be used for featured titles with tight letter-spacing (-0.02em) to evoke a modern film poster feel.
*   **Body & Labels (Inter):** Used for metadata and descriptions. `body-md` (0.875rem) provides high legibility against dark backgrounds.

**Hierarchy Tip:** Always pair a `display-sm` movie title with a `label-md` category (e.g., "DRAMA • 2024") in all-caps with 0.1em letter spacing to create an authoritative, curated look.

---

### 4. Elevation & Depth: Tonal Layering
We reject traditional drop shadows in favor of **Tonal Layering**.

*   **The Layering Principle:** Depth is achieved by stacking. Place a movie card (`surface_container_highest`) on a background (`surface`). The contrast creates a natural "lift."
*   **Ambient Shadows:** For floating modals, use an ultra-diffused shadow: `box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5)`. Never use pure black shadows on non-black surfaces; instead, tint the shadow with the `on_surface` hue at 4% opacity.
*   **The "Ghost Border" Fallback:** If a container requires more definition, use the `outline_variant` token at **15% opacity**. It should be felt, not seen.

---

### 5. Components

#### Cards (The Hero Component)
*   **Styling:** Use `rounded-md` (0.75rem) for posters.
*   **Layout:** No dividers. Use 24px of vertical white space to separate "Trending" from "Recommended."
*   **Hover State:** On hover, the card should scale slightly (1.05x) and the background should shift to `surface_bright`. 

#### Buttons
*   **Primary:** `primary_container` (#e50914) with `on_primary_container` text. Use `rounded-full` for a modern, sleek feel.
*   **Tertiary:** Ghost style. No background, `on_surface` text. Only a subtle `surface_variant` background appears on hover.

#### Input Fields
*   **Style:** Minimalist. Use `surface_container_highest` with no border. Use `outline` only for the focus state.
*   **Typography:** Labels must use `label-sm` and be positioned 8px above the input area.

#### Immersive Sliders
Instead of standard scrollbars, use a "Bleed Effect." The last visible card on the right should fade into the background using a linear gradient mask, signaling that more content exists without using a scrollbar.

---

### 6. Do’s and Don’ts

**Do:**
*   **Do** use extreme scale. A "New Release" title can be 4x the size of the description.
*   **Do** embrace negative space. Let the movie posters breathe.
*   **Do** use `surface_tint` at 5% opacity over images to keep them visually consistent with the dark theme.

**Don’t:**
*   **Don’t** use pure #000000 black for anything other than absolute shadows. It kills the "Cinematic Gray" depth.
*   **Don’t** use icons with heavy strokes. Use light, 1.5px stroke icons to match the sophistication of Inter/Manrope.
*   **Don’t** use standard "Select" dropdowns. Create custom overlays using the Glassmorphism rules defined in Section 2.

---

### 7. Accessibility Note
While we are building a dark, cinematic experience, readability is non-negotiable. Ensure all `body-sm` text maintains at least a 4.5:1 contrast ratio against its respective `surface` tier. Use `on_surface_variant` for secondary info only if the contrast remains accessible.```