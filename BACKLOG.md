# Air Canvas — UX Analysis & Backlog

> Authored: 2026-03-21 | Perspective: UX Designer + Architect + Parent of a 3-year-old

---

## Current Gesture→Action Mapping Analysis

| Gesture | Action | Intuitive? | Problem |
|---------|--------|------------|---------|
| 👆 Point (index only) | Draw | ✅ Great | Natural, kids get it immediately |
| ✊ Fist | Stop drawing | ✅ Good | Passive — just stop extending finger |
| 🤏 Pinch | Cycle color | ❌ Bad | Invisible action → invisible result. Kid has no idea what changed or how to get the color they want. Cycling is slot-machine UX. |
| 🖐️ Open palm (hold 1.5s) | Clear canvas | ⚠️ Broken | See item #2. Also: destructive action on a "resting" gesture is dangerous. Kids naturally open their palms when confused. |
| ✌️ V-sign | Place sticker | ❌ Bad | Kids 3-8 can't reliably make V-sign. Even if they can, they don't know WHERE the stamp goes or WHICH stamp they'll get. No preview. |
| 🤏 Pinch (thumb+index) | Also checked before pointing | ⚠️ Conflict | Pinch threshold (0.06) is very tight. A kid reaching to point can accidentally trigger color change. |

**Core problem:** Too many gestures require fine motor control that kids 3-8 don't have. The app has 5 distinct hand poses but kids this age reliably do 2: point and open hand.

---

## UX Research: How Kids Drawing Apps Handle This

### Color Selection
- **Kids finger paint apps (iPad):** On-screen palette, always visible. Tap/touch a color = selected. No hidden gestures.
- **Draw Alive / Quiver:** Color palette as a toolbar strip, bottom or side of screen.
- **Tux Paint:** Big colorful squares on the side. Point and click.
- **Consensus:** **On-screen palette + point-to-select is the universal pattern.** Zero learning curve.

### Stamps/Stickers
- **Tux Paint:** Stamp tool selected from toolbar, then click to place. Preview shown at cursor.
- **Drawing Desk Kids:** Sticker panel opens, kid taps sticker, then taps canvas location.
- **Consensus:** **Select first, then place.** Never surprise the user with a random stamp. Always show a preview of what will be placed.

### Clear/Undo
- **Almost all kids apps:** Explicit button (trash can icon), sometimes with confirmation ("Are you sure?").
- **No app uses a gesture for destructive clear.** It's always a deliberate UI action.

---

## Backlog Items

### #1 — 📷 Camera as Default Background (AR-Style)
**Priority: P1** | **Complexity: S** | **Impact: High**

**Problem:** Kid draws on a black void. No sense of presence. The magic of "painting on yourself" is missing. Current themes (space/forest/ocean) are static dot patterns — decorative but not engaging.

**Proposed Solution:**
- Add `"camera"` as a theme in `THEMES` list, make it the default (`BACKGROUND_THEME = "camera"`)
- In the render loop, instead of `self.theme_bg_cache[current_theme]`, use the current (flipped) camera frame directly as the background
- The camera frame is already available — it's read every loop iteration but only used for hand detection. Just use `frame` as the background layer in `compose_art_layers()`
- Keep existing themes accessible via `b` key cycling: `["camera", "dark", "space", "forest", "ocean"]`

**Implementation approach:**
```python
# In run() render section:
if current_theme == "camera":
    background = frame  # Already flipped
else:
    background = self.theme_bg_cache[current_theme]
```

**Done criteria:**
- [ ] Camera feed is the default background
- [ ] Child sees themselves behind their drawing
- [ ] Drawing strokes render on top of camera feed with good contrast
- [ ] Theme cycling with `b` still works, camera is first in list
- [ ] Neon glow effect still looks good on camera background (may need slight darkening overlay)

---

### #2 — 🧹 Clear with Open Palm Not Working
**Priority: P0** | **Complexity: S** | **Impact: High (broken feature)**

**Problem:** `is_open_palm()` only checks 4 fingers (index, middle, ring, pinky) but NOT the thumb. This means any 4-fingers-extended pose triggers it — including the V-sign transition and natural "reaching" poses. But the real issue is the **stillness check is missing from the code.**

**Root cause analysis:**
- `config.py` defines `CLEAR_STILLNESS = 0.03` but **it's never used in `_process_hand()`**
- The `is_open_palm()` function works correctly (checks 4 fingers extended)
- BUT: `_process_hand()` checks `is_open_palm()` AFTER the pinch check. If thumb and index are close (pinch), the pinch branch fires first and returns early — **the palm check never runs**
- More critically: **gesture priority order matters.** The current order is: pinch → open_palm → fist → v_sign → pointing. When a kid opens their palm, the thumb-index distance might be small enough to trigger pinch first.
- The `CLEAR_STILLNESS` config was meant to require the hand to be still (not moving) during the hold, preventing accidental clears. **This was never implemented.**

**Proposed Solution:**
1. **Add stillness check:** In the open_palm branch, verify `state.speed < CLEAR_STILLNESS` before counting hold time
2. **Add thumb check to `is_open_palm()`:** Check that thumb tip is extended (thumb_tip.x significantly different from thumb_ip.x, accounting for handedness)
3. **Visual countdown:** Show a radial progress indicator around the palm when hold timer is active — kid sees "something is happening" and knows to keep holding
4. **Consider removing gesture-based clear entirely** in favor of an on-screen trash button (see UX research above). Gesture clear is inherently dangerous for kids.

**Done criteria:**
- [ ] Open palm held for 1.5s clears canvas (stillness enforced)
- [ ] Thumb must be extended (not just 4 fingers)
- [ ] Visual countdown/progress shown during hold
- [ ] No false triggers during normal drawing/gesturing
- [ ] `CLEAR_STILLNESS` config value actually used

---

### #3 — 💾 Save More Intuitive
**Priority: P2** | **Complexity: S** | **Impact: Medium**

**Problem:** Pressing `s` saves to the current directory as `drawing_<timestamp>.png`. The only feedback is a green "Saved!" overlay for 2 seconds (and that's only for `p`/export — `s` has NO visual feedback at all, only console print). Kid (and parent) has no idea where the file went.

**Current code issue:**
- `SAVE_KEY` handler: `cv2.imwrite(filename, self.canvas)` + `print()` — **zero on-screen feedback**
- `EXPORT_KEY` handler: shows "Saved!" overlay but only says "Saved!" with no path
- Save goes to CWD (unpredictable), export goes to `~/Desktop/air-canvas-art/`

**Proposed Solution:**
1. **Unify save behavior:** Both `s` and `p` should save to `~/Desktop/air-canvas-art/` (a known, visible location)
2. **Rich overlay feedback (3 seconds):**
   - Thumbnail of the saved drawing (small preview in corner)
   - "Saved! ✨" in large kid-friendly text
   - Path shown smaller underneath: `~/Desktop/air-canvas-art/art-2026-03-21.png`
   - Brief camera-flash white screen effect (100ms white overlay fade) for satisfying "snapshot" feel
3. **Sound:** Play a distinct "camera shutter" or "sparkle" sound on save
4. **Auto-create folder** with a fun name visible in Finder

**Done criteria:**
- [ ] Both `s` and `p` save to `~/Desktop/air-canvas-art/`
- [ ] On-screen overlay shows "Saved! ✨" + file path for 3 seconds
- [ ] Camera-flash visual effect on save
- [ ] Sound feedback on save
- [ ] Thumbnail preview of saved art shown in overlay

---

### #4 — ✨ Visual UX Polish
**Priority: P2** | **Complexity: M** | **Impact: Medium**

**Problem:** The UI is functional but looks like a developer tool, not a kids toy. FPS counter, keyboard shortcut text at the bottom, "L"/"R" hand labels — none of this is meaningful to a 4-year-old.

**Current UI elements that need work:**
- FPS counter (top-left) — remove or hide behind debug flag
- "THEME: DARK" text — unnecessary for kids
- "Point=Draw | V=Stamp | a=Alive..." bottom bar — keyboard shortcuts are meaningless to kids
- "L" and "R" color indicators — too small, too abstract
- No visual hierarchy, everything is monochrome text

**Proposed Solution:**

**Remove for kids:**
- FPS counter → only show if `DEBUG = True`
- Keyboard shortcut bar → remove entirely (keep in README)
- "THEME: X" label → remove

**Improve:**
- **Hand color indicators:** Replace "L"/"R" circles with larger, animated color blobs (pulsing gently). Place them at bottom-left and bottom-right corners. Make them 3x bigger (60px radius).
- **Cursor:** Instead of a thin circle, show a **filled semi-transparent circle** matching the current brush color + size. Kid sees exactly what they'll draw.
- **Avatars:** Make penguin/cat bigger (current 40px is tiny). 80px minimum. Add gentle bobbing animation.
- **Rainbow mode indicator:** Instead of text, show a rainbow arc across the top of the screen.
- **On-screen mode buttons** (for future, see #6): Instead of keyboard shortcuts, show large colorful icons the kid can point at.

**Visual style guide:**
- Rounded shapes everywhere (no sharp rectangles)
- Gentle pulsing animations (nothing fast or startling)
- Colors: bright but not neon-on-black harsh. Slightly muted pastels for UI elements, vibrant for drawing.
- Font: If text is needed, use large rounded sans-serif

**Done criteria:**
- [ ] FPS, theme label, shortcut bar removed from default view
- [ ] Color indicators are large, animated, clearly visible
- [ ] Cursor preview matches actual brush size and color
- [ ] Avatars are larger and have gentle animation
- [ ] Overall feel is "toy" not "tool"
- [ ] Optional `DEBUG` flag re-enables FPS and technical info

---

### #5 — 🎯 Stickers/Stamps More Intuitive
**Priority: P1** | **Complexity: M** | **Impact: High**

**Problem:** V-sign gesture is confusing for kids 3-8. They don't know:
1. How to make the gesture (fine motor control issue)
2. What stamp they'll get (random cycling, no preview)
3. Where exactly it'll be placed (between index+middle tips — not obvious)

The V-sign also conflicts with natural hand movements. Kids exploring often extend 2 fingers naturally.

**Proposed Solution: On-Screen Stamp Shelf**

Replace gesture-based stamping with a **visual stamp shelf:**

1. **Stamp shelf:** A row of 4-6 stamp icons displayed at the top of the screen (star ⭐, heart ❤️, circle 🔵, smiley 😊)
2. **Selection:** Point at a stamp icon → it highlights/enlarges → stamp becomes "loaded" on your cursor
3. **Placement:** With a stamp loaded, point anywhere on canvas → stamp is placed there
4. **Deselection:** Point away from shelf or make a fist → go back to drawing mode
5. **Preview:** While stamp is loaded, show a ghost/preview of the stamp following the fingertip
6. **Keep V-sign as optional shortcut** for power users but don't teach it to kids

**Shelf design:**
```
┌──────────────────────────────────────────────┐
│   ⭐    ❤️    🔵    😊    🌈    🦋          │  ← Top of screen, ~80px tall
└──────────────────────────────────────────────┘
│                                               │
│              Drawing canvas                   │
│                                               │
└───────────────────────────────────────────────┘
```

- Each icon: 60x60px, spaced evenly
- Active stamp: glowing border + 1.3x scale
- Hover: gentle scale-up animation
- The shelf can auto-hide after 5s of no interaction near it, reappear when finger approaches top

**Done criteria:**
- [ ] Stamp shelf visible at top of screen with all available stamps
- [ ] Point at stamp to select it
- [ ] Ghost preview follows fingertip when stamp is loaded
- [ ] Point on canvas to place stamp
- [ ] Fist or point-away deselects stamp mode
- [ ] V-sign still works as legacy shortcut
- [ ] No accidental stamps during normal drawing

---

### #6 — 🎨 Visual Color Palette
**Priority: P0** | **Complexity: M** | **Impact: Critical**

**Problem:** Pinch-to-cycle-color is the single worst UX in the app for kids:
1. **Invisible:** No visual affordance. Kid doesn't know colors can be changed.
2. **Unpredictable:** Cycling means you have to go through colors you don't want to get to the one you do.
3. **Motor skill barrier:** Pinch requires fine thumb-index coordination that 3-5 year olds struggle with.
4. **Accidental triggers:** Pinch threshold (0.06) is close to natural pointing distance, causing unwanted color changes.
5. **No visual map:** Kid can't see available colors or which one is selected.

**Proposed Solution: On-Screen Color Palette**

**Layout — Left Side Vertical Strip:**
```
┌──┐
│🔴│  ← Large color circles (50px diameter)
│🟠│     stacked vertically on the LEFT edge
│🟡│     
│🟢│     Currently selected: glowing ring + 1.5x size
│🔵│     
│🟣│     
│⚪│     
│🩷│     
└──┘
```

**Interaction model:**
1. **Point at color** → color highlights (hover state: grows + glows)
2. **Dwell 0.3s on color** → color selected for that hand (dwell prevents accidental selection)
3. **Visual feedback:** Selected color gets a bright ring + "pop" animation. The hand's color indicator updates.
4. **Both hands:** Left palette on left edge, right palette on right edge (matches current L/R hand color split)
5. **Always visible.** Never hide the palette. Kids need persistent affordances.

**Why dwell (0.3s) not instant:**
- Prevents color changes when finger sweeps past palette while drawing
- 0.3s is fast enough to feel instant but slow enough to prevent accidents
- Show a small radial progress ring during the dwell

**Color selection (8 colors, 2 palettes):**

Left hand palette (left edge):
| Color | BGR | Name |
|-------|-----|------|
| Cyan | (0,255,255) | Sky |
| Magenta | (255,0,255) | Berry |
| Green | (0,255,0) | Grass |
| Yellow | (255,255,0) | Sun |

Right hand palette (right edge):
| Color | BGR | Name |
|-------|-----|------|
| Orange | (255,100,0) | Fire |
| Pink | (255,0,100) | Flower |
| Light Blue | (100,100,255) | Cloud |
| White | (255,255,255) | Snow |

**Keep pinch as hidden power-user shortcut** but remove it from the taught gestures.

**Done criteria:**
- [ ] Color palettes visible on left and right edges of screen
- [ ] Point at color + dwell 0.3s = select color
- [ ] Visual hover feedback (grow + glow)
- [ ] Visual selection feedback (ring + pop animation)
- [ ] Selected color clearly indicated
- [ ] No accidental color changes while drawing near edges
- [ ] Pinch still works as legacy shortcut
- [ ] Palette doesn't interfere with drawing area

---

## Prioritized Backlog Summary

| # | Item | Priority | Size | Rationale |
|---|------|----------|------|-----------|
| 2 | 🧹 Open palm clear broken | **P0** | S | Broken feature. Regression. |
| 6 | 🎨 Visual color palette | **P0** | M | Core interaction is unusable for target age group. |
| 1 | 📷 Camera as default background | **P1** | S | Biggest "wow factor" for minimal effort. |
| 5 | 🎯 Stamp shelf | **P1** | M | Current stamp UX is unusable for kids. |
| 3 | 💾 Save feedback | **P2** | S | Functional but poor feedback. |
| 4 | ✨ Visual polish | **P2** | M | Important but cosmetic. Do after core interactions work. |

**Recommended execution order:** #2 → #1 → #6 → #5 → #3 → #4

Fix what's broken (#2), add the wow factor (#1), then fix the two core interaction patterns (#6, #5), then polish (#3, #4).

---

## Suggested Gesture Mapping (Post-Backlog)

| Gesture | Action | Why |
|---------|--------|-----|
| 👆 Point | Draw (or select from palette/shelf) | Universal, intuitive |
| ✊ Fist | Stop drawing / deselect stamp | Natural "stop" |
| 🖐️ Open palm (hold 1.5s, still) | Clear canvas (with visual countdown) | Keep but add safeguards |
| ~~🤏 Pinch~~ | ~~Cycle color~~ → Hidden legacy shortcut | Replaced by visual palette |
| ~~✌️ V-sign~~ | ~~Place stamp~~ → Hidden legacy shortcut | Replaced by stamp shelf |

**Net result:** Kid needs to learn exactly **2 gestures** — point and fist. Everything else is visual (palette, shelf, buttons). That's how it should be for ages 3-8.

---

## Architecture Notes

- **No new dependencies needed.** Everything is achievable with current OpenCV drawing primitives.
- **Performance concern:** Camera-as-background means compositing on every frame (already happening with theme backgrounds, so no regression).
- **State additions:** `selected_stamp: Optional[str]`, `palette_hover_start: float`, `stamp_shelf_hover_start: float`, `debug_mode: bool`
- **Config additions:** `DEBUG = False`, `PALETTE_DWELL_TIME = 0.3`, `PALETTE_CIRCLE_RADIUS = 25`, `STAMP_SHELF_HEIGHT = 80`
- **File structure:** No new files needed. All changes fit in `air_canvas.py` + `config.py`.

---

*"The best interface is no interface. The second best is one a 3-year-old can figure out in 5 seconds."*
