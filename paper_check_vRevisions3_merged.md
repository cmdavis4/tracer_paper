# Check of "Davis et al tracer paper vRevisions3 merged.pdf" (11 October)

Tick an item by changing `[ ]` to `[x]`. Sections 1–4 have a box on every action item; section 5 needs no action.

## What was compared

- **Paper:** `Davis et al tracer paper vRevisions3 merged.pdf` in the repo root, which is the version attached in chat on 11 October. Line numbers below follow it.
- **Notebook:** the 10 October rerun of `analysis/analysis-tracerpaper.ipynb`. All 48 code cells ran in order without errors, and the notebook was saved with their outputs at 12:15. I used those outputs, `entrainment_factor_stats.pkl` and the figure-data pickles.
- **Figures:** I compared every figure image embedded in the PDF with the rerun's PNG in `analysis/paper_figures/`.
- **IDs** such as P1-E1 refer to `paper_check_vRevisions3.md` and its checklist.

**Summary**

- Most checklist items are in the merged version. Section 1 lists two ticked items that the PDF does not fully implement, and two open items with something new to add.
- The air-mass fix moved two quoted numbers past the precision the text gives (section 2, items 1 and 2).
- The new text has three errors to fix before submission: a sentence that names the wrong storms, a sentence that compares the squall line with itself, and a "[citation]" placeholder (section 2, items 3–5).
- Six of the eleven figures in the PDF predate the rerun (section 4).

---

## 1. Checklist items that the merged version does not fully implement

- **Ticked but not fully in the PDF:** P1-E1 (the abstract), and four of the reference fixes in P1-E16 and P2-E5.
- **Open, with something new to add:** B (six figures still to replace) and P1-E15 (one more file for the archive).

Every other ticked item is in the PDF. Two of their numbers have since moved with the air-mass fix (section 2, items 1 and 2).

### P1-E1: the abstract still says "for all but the lightest rain"
*Line 19*

- [x] Fix the abstract's "for all but the lightest rain".

Lines 395–402 are fixed, but the abstract is unchanged. At 1 mm/hr, the organized storms' peak q̄_RS is 3.2–3.9 times the IC's, and their time mean is 7.1–9.3 times the IC's. So the organized storms entrain significantly more rainy air than the IC for the lightest rain too.

Lines 653–655 ("For light rainfall rates, the differences between the three storm morphologies are less pronounced") hold in direction, because the peak ratios grow from 3.2–3.9 at 1 mm/hr to 4.8–5.2 at 20 mm/hr.

### B: six figures in the PDF predate the rerun

- [x] Replace the six figures that section 4 lists.

### P1-E16 and P2-E5: four reference fixes are not in the PDF

- [x] **Grant et al. (2018):** the third author is still "Heever, S. C. van den" (line 811).
- [x] **Kaimal et al. (1976) and Wakimoto (1985):** neither entry has a journal, volume or pages yet (lines 850–851 and 995). From memory, they are J. Atmos. Sci., 33, 2152–2169, and Mon. Weather Rev., 113, 1131–1143. Confirm against the publisher pages.
- [x] **Maddox (1980):** the DOI field still holds the JSTOR URL (line 880).
- [x] **Foote and Fankhauser (1973):** the DOI still starts "https://doi.org/https://doi.org/" (line 790). Eighteen other entries now have the same doubled prefix (section 2, item 6).

Checklist item H, "Check that all bibliography references propagated", may cover these.

### P1-E15 (open, in H): the archive also lacks an input to Fig. 6

Both the current notebook and the archived one read the Mignani et al. (2025) fit parameters from `/moonbow/cmdavis4/projects/bl_transport/251006_INPconc-SumP_slope_intercept_results_forCharles.csv`. That file is not in `replication/data/`, whose `mignani_f6a.csv` holds different data. Without it, Fig. 6 and the Sect. 3.2 numbers cannot be reproduced from the archive.

- [ ] Add the CSV to the new archive version, and read it by a path relative to the repo.

---

## 2. Errors

### 1. The SC/IC ratio for the 1 mm/hr tracer is now 3.2, not 3.3
*Line 396*

- [x] Change "3.3x" to 3.2x.

After the air-mass fix, the SC's peak q̄_RS is 3.24 times the IC's. The SL's is 3.87 times, so "3.9x" still holds.

### 2. The SL and SC maximum q̄_FS now differ by 37%, not 35%
*Line 432*

- The SL's maximum is 2.20e6 # kg⁻¹ and the SC's is 1.60e6, so the SL's is 37% larger. Before the air-mass fix the difference was 35%.
- Measured from the SL's value, the SC's maximum is 27% smaller, which "within 35%" covers.

- [x] Name the reference, for example "the supercell's maximum is 27% smaller than the squall line's", or change 35% to 37%.

### 3. Line 441 names the wrong pair of storms
*Lines 440–442*

The text says "the squall line and supercell attain similar peak proportions of surface air in their updrafts, while the supercell's peak proportion of surface air is significantly less". The pair with similar normalized maxima is the IC and the SL (245 and 242 # kg⁻¹ s⁻¹). The SC's is 150. 

- [x] Begin the sentence "the isolated convective storm and squall line attain similar peak proportions".

### 4. Line 614 compares the squall line with itself, and its ratio disagrees with line 412
*Lines 614–615*

- [x] "the squall line reaches a maximum q̄_RS value ~6x that of the squall line's" should end "that of the supercell's".
- [x] The ratio is 5.4 (5.36), which line 412 quotes as "~5.4". "~6x" overstates it.

### 5. A citation placeholder remains
*Line 617*

- [x] Replace the placeholder: "owing to their strongly dynamically forced updrafts [citation]."

### 6. Reference-list problems not covered by the checklist

- [x] **Doubled DOI prefix.** These entries have DOIs that start "https://doi.org/https://doi.org/", so their links break:
  - Browning (1964), Davies-Jones (1984), Foote and Fankhauser (1973), Fujita and Grandoso (1968) and Hill (1974);
  - Jorgensen et al. (1997), Kingsmill and Wakimoto (1991), Klemp and Wilhelmson (1978), Lemon and Doswell (1979) and Newton (1950);
  - Rotunno et al. (1988), Schlesinger (1978), Smagorinsky (1963), Smull and Houze (1987) and Szoke and Zipser (1986);
  - Tompkins (2001), Weisman (1992), Weisman et al. (1988) and Wilhelmson (1974).
- [x] **Agard and Emanuel (2017)** has no journal, volume or pages (line 728). From memory, it is J. Atmos. Sci., 74, 3043–3054.
- [x] **Weisman and Klemp (1984):** the title reads "Convective Stormsin Directionally" (line 1003).

### 7. The Fig. 3 caption's citation is malformed
*Lines 275–276*

- [x] Fix the citation: "following (Seigel and van den Heever (2012)and Tompkins (2001)." has a stray opening parenthesis and a missing space.

---

## 3. Borderline

- [x] **Mixed-layer depth (line 139).** The text says θ is well mixed "below 900 hPa (~930 m)". All three namelists hold θ at 301.93 K from the surface to 1000 m, which is about 892 hPa. θ then rises to 302.55 K at 1250 m.
- [x] **"2–3x smaller" (line 431).** The IC's maximum q̄_FS is 2.5 times smaller than the SL's, but only 1.8 times smaller than the SC's.
- [x] **Lines 491–493 cite Fig. 9 (a–c) without saying which curves.** The IC's peak is "comparable to the largest proportion reached by either of the more organized storms" only for the normalized q̄_FS (dashed lines: 245 against 242 and 150 # kg⁻¹ s⁻¹). On the solid lines, the IC's peak is 40% of the SL's. Add "dashed lines".
- [x] **Lines 618–620** say that "isolated convective storms are far less effective at this transport than either of the other storms at any rain intensity". "This transport" refers back to air "from both rainy and non-rainy surface regions". For non-rainy air, the IC's peak N_FS-UL/F_upward is within 10% of the SL's, which line 604 says. The clause holds for rainy air only, where the SL's peak is 60–325 times the IC's.
- [x] **Line 242 writes N_RS for the upper-level amount.** It should be N_RS-UL, because Table 3 defines N_RS as the in-updraft total.
- [ ] **Right mover only (lines 249–251 and 614–617).** The notebook's "Sensitivity tests" section now computes this comparison (added 12 October). It limits the supercell to its right mover in two ways: a static line applied to the flux, the tracer aloft and the emission; and attribution where air crosses 10 km upward, which anvil spreading cannot bias. The two agree within 0.05 for every right-mover / full-domain ratio.

   | SC / SL, peak | 1 mm/hr | 5 | 10 | 20 | 40 |
   |---|---|---|---|---|---|
   | Per unit upward flux, full domain | 7.8 | 6.9 | 6.2 | 6.2 | 4.2 |
   | Per unit upward flux, right mover | 9.0 | 8.1 | 7.3 | 7.0 | 1.4 |
   | Per unit emitted, full domain | 13.4 | 12.0 | 10.8 | 9.6 | 3.6 |
   | Per unit emitted, right mover | 18.7 | 17.0 | 15.3 | 13.7 | 4.2 |

   - The fixed-source ratios do not change: 6.0 times the SL's end value and 6.6 times the IC's peak.
   - The SC leads the SL at every threshold with either domain, so lines 249–251 hold.
   - The "~4x for either normalization" at 40 mm/hr (line 615) holds only for the full domain. With the right mover alone it is 1.4 per unit upward flux. About three quarters of the supercell's 40 mm/hr tracer above 10 km comes from outside the right mover, so the explanation in lines 616–617 rests partly on the left mover.
   - Over 1–20 mm/hr, the "4x to 13x" range (lines 608–611) would widen to 7x–19x with the right mover alone.

- [x] **Fig. 9 caption units (lines 445–447; also lines 436–437).**
   - The dashed q̄_FS is described as "normalized by the amount of fixed-source tracer that has been emitted". Dividing by that amount would give a dimensionless number. The curves are divided by elapsed seconds, so the unit "# kg⁻¹ s⁻¹" is correct but the description is not. Lines 436–437 equate the two with "i.e.".
   - The caption gives f_RS in "[#]" and its normalized form in "[# s]". The axes say "fraction" and "s".

   No number changes either way, because the text quotes only ratios.

---

## 4. Figures

**PDF figures compared with the rerun's PNGs.**

- **Current:** Figs. 1, 2, 6, 8 and 9.
- **Predate the rerun, so replace them:**
  - [x] **Fig. 3:** the cold-pool boundary in (e) and (f) comes from the old liquid-only buoyancy. Nothing else differs.
  - [x] **Fig. 4:** the older layout, with the condensate contours drawn on the coarse arrow grid.
  - [x] **Fig. 5:** the older layout, with the same data.
  - [x] **Fig. 7:** the curves come from before the air-mass fix.
  - [x] **Fig. 10:** the updraft mass in (a–c) comes from before the air-mass fix. Nothing else differs.
  - [x] **Fig. 11:** the old version. It has the 10⁰ floor in (d–f), which hides the IC's 40 mm/hr curve, and the legend over the bottom row's x-axis labels.

- [x] **The rerun's Fig. 11** fixes both of those problems. However, in row (g–i), the y-axis label "Fraction" is drawn over the row header's "E_RS". In rows (a–c) and (d–f), the "#/kg" labels touch their headers.

- [x] **Notebook.** The old Fig. 1 cell (`224f8143`, index 20) still runs before the new one and writes the same files. The new cell overwrites them, so the output is right, but delete the old cell before archiving.

---

## 5. What matched

Every number in Sects. 3.2–3.4 and the conclusions that is not listed above matches the rerun at the precision the text gives:

| Section | Paper | Rerun |
|---|---|---|
| 3.2 | Medians 0.06, 0.24, 0.20 L⁻¹ STP; maxima 0.45–0.48 | Same; 0.450–0.478 |
| 3.2 | Peak rain at t = 70, 120, 180 min | Same |
| 3.3.2 | ~5x at 20 mm/hr | SL 5.2, SC 4.8 |
| 3.3.2 | IC peak q̄_RS at t = 65 min; SL and SC at t = 150–180 min | Same |
| 3.3.2 | SL/SC peak q̄_RS at 40 mm/hr ~5.4 (line 412) | 5.36 |
| 3.3.3 | IC peak q̄_FS at t = 60 min; SL and SC at t = 180 min | Same |
| 3.3.3 | Normalized IC and SL within 3%; SC ~40% less | 1.4%; 38–39% less |
| 3.3.4 | SL and IC peak f_RS within 25% (1–20 mm/hr) | Up to 22% |
| 3.3.4 | SC peak f_RS at least twice the others' | 2.1–2.6 times |
| 3.3.4 | SL normalized f_RS more than double the IC's | 2.2–10.4 times |
| 3.3.4 | SL f_RS ~30–55% below the SC's (1–20 mm/hr) | 29–57% below |
| 3.3.5 | t = 70 min is the rain peak and the last time with updraft below 100 m | Same |
| 3.3.5 | IC η_RP ~a quarter of the SL's and ~half of the SC's; 41% and 66% at 40 mm/hr | 0.25–0.28; 0.48–0.54; 41%; 66% |
| 3.3.6 | IC peak η_RSE at t = 65 min; rain rate peaks at t = 70 min; ~3.3 K | Same; 3.30 K at the lowest level |
| 3.3.6 | SL/SC η_RSE 37%–40%; 39% to 47% after t = 120 min; ~3x at 40 mm/hr | 36.6–40.4%; 38.6–47.4%; 3.3 |
| 3.4 | SC end ~6x the SL's and ~7x the IC's peak (fixed-source) | 6.0; 6.6 |
| 3.4 | IC and SL peaks within 20% | 10% |
| 3.4 | SC/SL 4x to 13x; ~4x at 40 mm/hr | 3.6–13.4; 4.2 and 3.6 |
