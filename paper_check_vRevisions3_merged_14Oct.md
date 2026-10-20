# Check of "Davis et al tracer paper vRevisions3 merged.pdf" (14 October)

Tick an item by changing `[ ]` to `[x]`. Section 6 needs no action.

## What was compared

- **Paper:** `Davis et al tracer paper vRevisions3 merged.pdf` in the repo root, saved 14 October at 10:45. Line numbers below follow it. They differ from the 11 October export's.
- **Notebook:** the current `analysis/analysis-tracerpaper.ipynb`. Its saved outputs come from several partial runs. So I reran, from the caches, every cell that the checked numbers come from: rain onset, the in-updraft statistics, the Fig. 11 data and the right-mover test. I wrote nothing into the notebook.
- **Figures:** I compared every figure in the PDF with its PNG in `analysis/paper_figures/`.
- **Scope:** every sentence that changed since 11 October, every number in Sects. 3.3.3, 3.3.4 and 3.4, and the items ticked in `paper_check_vRevisions3_merged.md`.

**Summary**

- Sect. 2.4 now says that the supercell's upper-level results use the right mover only. Fig. 11 and every Sect. 3.4 number still use the full domain (section 1).
- Fig. 9 no longer has dashed normalized lines. Its caption, one axis label and two sentences still describe them (section 2, items 2 and 3).
- Sect. 3.4 has three wrong statements: "~6x", "4x – 8x", and a claim about simulation-averaged q̄_RS (section 2, items 4–6).
- Line 483 writes q̄_FS where it means f_RS (section 2, item 1).
- Three reference-list problems remain (section 3).

---

## 1. Sect. 2.4 says right mover, but Fig. 11 and Sect. 3.4 use the full domain

- [ ] Make the method text, Fig. 11 and Sect. 3.4 agree.

**What disagrees**

- Line 260 says "we limit to the right-mover region in these calculations".
- Lines 224–226 say the opposite: "We do include the tracer emitted in the left-mover region in the results assessing upper-level transport of tracers (Section 3.4)".
- The PDF's Fig. 11 is identical to the notebook's PNG. The notebook computes it over the supercell's full domain (cells `01331029`, `2d0bfa59` and `7556113a`). For example, the supercell's 40 mm/hr curve in Fig. 11 (f) ends at 539 # kg⁻¹, which is the full-domain value. The right mover's is 180 # kg⁻¹.
- The Sect. 3.4 numbers are full-domain values too. For example, "4x – 8x" matches the full-domain ratios per unit upward flux, 4.2–7.8.

**If you switch Fig. 11 to the right mover, as Sect. 2.4 describes**

- Use the right mover for all three parts of each ratio: the tracer aloft, the upward mass flux and the emission. The notebook's sensitivity cells (`579747ac` to `188872de`) already compute all three at 10 km.
- These supercell / squall line ratios change:

  | SC / SL, end of run | Fixed-source | 1 mm/hr | 5 | 10 | 20 | 40 |
  |---|---|---|---|---|---|---|
  | Per unit upward flux, full domain (PDF) | 6.8 | 7.8 | 6.9 | 6.2 | 6.2 | 4.2 |
  | Per unit upward flux, right mover | 6.8 | 9.0 | 8.1 | 7.3 | 7.0 | 1.4 |
  | Per unit emitted, full domain (PDF) | – | 13.4 | 12.0 | 10.8 | 9.6 | 3.6 |
  | Per unit emitted, right mover | – | 18.7 | 17.0 | 15.3 | 13.7 | 2.4 |

  The right mover's 40 mm/hr ratio per unit emitted peaks at 4.2 before the end of the run.
- At 40 mm/hr, the right mover lofts only 1.4 times the squall line's tracer per unit upward flux. So three statements would need a 40 mm/hr exception:
  - the abstract, line 19: "significantly more effective as a transporter of rainy surface air";
  - lines 625–627: "significantly more effective … at transporting air from both rainy and non-rainy surface regions";
  - lines 653–655: "transports far more of this air … relative to its cumulative upward mass flux".
- Lines 224–226 need rewriting.
- Lines 264–266 claim the same results at 7, 8 and 9 km. The notebook has the right mover's tracer aloft only at 10 km, so that test needs it computed at 7–9 km.
- The comparisons with the isolated storm (lines 620–624) hold with either domain.

**If you keep the full domain:** rewrite lines 259–260 so that they say Sect. 3.4 uses the full domain, and keep lines 224–226. Section 2, items 4 and 5 still apply.

---

## 2. Errors

### 1. Line 483 names q̄_FS where it means f_RS
*Lines 482–484*

- [x] "the isolated convective storm's maximum q̄_FS value is still larger than those of the other storms" should say f_RS.

The ratios that follow ("~1.3x", "~4x") are f_RS ratios. The IC's maximum q̄_FS is about 5% of the others'.

### 2. Fig. 9 lost its dashed lines, but the text still describes them

- [x] **Caption, lines 455–459:** delete "and q̄_FS normalized by the simulation time that has elapsed [# kg-1 s-1] (dashed lines)" and "and its normalized values [s] (dashed lines)". The two "(solid lines)" can go too.
- [x] **Line 489:** delete "(for both normalized and non-normalized q̄_FS values)".
- [x] **Line 507:** "its larger normalized f_RS" should be "its larger f_RS". The since-rain f_RS supports the sentence: at 1–20 mm/hr, the IC's peak is 1.7–4.2 times the other storms'.

### 3. Fig. 9's labels are left over from the normalized version

- [x] The y-axis of panel (a) says "Normalized q̄_FS". The panels show the since-rain q̄_FS in # kg⁻¹, which is not normalized.
- [x] The y-axis of panel (d) says "f_RS (solid, fraction)", but no dashed line remains.
- [x] The caption's "Results in panels (d-i) are shown for each of the five rain-sourced tracers" should say (d-l), because panels (j-l) also show the five tracers.

### 4. "~6x more fixed-source tracer" is now ~7x
*Line 610*

- [x] With since-rain fixed-source tracer, the supercell's end-of-run N_FS-UL/F_upward is 6.8 times the squall line's. The ratio is the same with either domain. It is 6.6–6.9 at 7–10 km.

### 5. "4x – 8x more rain-sourced tracer" leaves out panels (h–i)
*Lines 610–611*

- [x] The sentence cites Fig. 11 (e-f) and (h-i). Per unit upward flux (e-f), the ratio is 4.2–7.8, which "4x – 8x" covers. Per unit emitted (h-i), it is 3.6–13.4. These are full-domain values. Section 1 gives the right mover's.

### 6. The supercell's simulation-averaged q̄_RS is larger than the squall line's
*Lines 611–612*

"This trend holds when considering simulation-averaged and/or end-of-simulation q̄_FS and q̄_RS values" says that the supercell's in-updraft values stay below the squall line's.

| SC / SL | Fixed-source | 1 mm/hr | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|---|
| Simulation average | 0.90 | **1.30** | **1.25** | **1.19** | **1.08** | 0.16 |
| End of simulation | 0.82 | 0.99 | 0.96 | 0.93 | 0.93 | 0.11 |

- [x] The claim fails for the simulation-averaged q̄_RS at 1–20 mm/hr (bold). Limit it to end-of-simulation values, or name the exception.

### 7. The new equation for F_t-RM has the wrong time step
*Line 250*

- [x] The equation multiplies by (2 s), as Eq. 2 does. Line 252 says that this flux is computed at 5-minute intervals, and the code uses the 300 s output interval. Replace (2 s) with (300 s). The ratio f_t-flux does not change.

---

## 3. Reference list and typos

- [x] **Peters et al. (2020)** has no journal, volume or pages (lines 941–942). From memory, it is J. Atmos. Sci., 77, 1475–1495. Confirm against the publisher page.
- [x] **Wakimoto (1985)** now has a journal and volume, but no pages (lines 1010–1011). From memory, they are 1131–1143.
- [x] **Markowski and Richardson (2014)** is in the list (line 896), but the text cites it nowhere. Cite it or delete it.
- [x] **Line 610:** a period is missing in "sections 3.3.2 and 3.3.3 However".
- [x] **Acknowledgements and Financial support** (lines 722–731) repeat the same text word for word. Check whether the journal wants the funding in both.

---

## 4. Borderline

- [x] **"within 5% for all six cases" (line 258).** The 40 mm/hr pair differs by 6%: f_t-flux is 0.223 and f_t-N is 0.237. The other five pairs agree within 3%. All six agree within 0.025 in absolute terms. "Within 6%" or "within 0.03" would be accurate.
  - Lines 261–263 could cite one more fact: the right mover's share of the upward air mass flux is 0.709 from the 2-second output and 0.703 from the 5-minute output.
- [x] **Rain onset (lines 400–402).** The text says the kept species are those "emitted since the storm began raining". It says this occurs at t=45 (IC), 55 (SL) and 40 min (SC), which are the first outputs with rain. The calculation also keeps the species emitted in the 5 minutes before each of those outputs, because rain began within that interval. So the kept emission starts at 40, 50 and 35 min. A reader who followed the text literally would drop one species per storm. A clause such as "including the species being emitted when rain began" would fix it.
- [x] **Table 3 and line 364** define N_FS and "fixed-source tracer" as the sum over all species. Lines 400–402 then restrict N_FS to the since-rain species. A reader of Table 3 or Table 4 alone would miss the restriction.
- [x] **"~another order of magnitude" (lines 620–622).** From in-updraft to upper-level maxima, the isolated storm's gap to the squall line grows by 5–28 times. Its gap to the supercell grows by 73–480 times. "One to two orders of magnitude" would cover both.
- [x] **"of the same order of magnitude as the other storms" (lines 505–506).** This holds at 1–20 mm/hr, where the organized storms' peak q̄_RS is 3.2–5.2 times the IC's. At 40 mm/hr, the squall line's is 16 times the IC's.
- [x] **Conclusion 4, line 653: "The supercell entrains less rainy and non-rainy surface air into its updraft than the squall line".** At 1–20 mm/hr, the supercell's peak q̄_RS is only 7–16% below the squall line's, and its simulation average is 8–30% above (section 2, item 6). Conclusion 2 (lines 640–645) calls the two comparable. The claim holds clearly for non-rainy air and for 40 mm/hr rain.

---

## 5. Figures and notebook

All PDF figures match the notebook's PNGs except Fig. 4. Figs. 1, 5 and 6 differ only in resolution.

- [x] **Fig. 4.** The PDF labels the cross sections "Cleft" and "Lobe". The notebook's Fig. 4 cell (`44ed0846`) labels them "Upper" and "Lower". The data agree. Change the cell's labels so that the archived notebook reproduces the PDF figure. The PNG on disk (12 October) also has a swapped legend, but the current cell code is correct, so a rerun fixes it.
- [x] **Fig. 11 label overlaps (ticked on 11 October)** are still in the PDF. In row (g–i), "Fraction" overlaps "E_RS". In rows (a–c) and (d–f), "#/kg" touches "F_upward".
- [x] **Notebook outputs.** Many cells have no saved output, and the execution counts restart several times. Rerun the notebook top to bottom before archiving.
- [x] **A source for line 258.** The method comparison (`27b685a7`) prints right-mover / full-domain ratios of tracer per unit upward air mass. It does not print f_t-flux and f_t-N as Sect. 2.4 defines them. Printing those would give line 258 a direct source.

---

## 6. What matched

Every other item ticked on 11 October is in the PDF. The rerun matches every number not listed above, at the precision the text gives:

| Lines | Paper | Rerun |
|---|---|---|
| 16–17 | SL and SC entrain significantly more rainy air than the IC, at all rain rates | Peak q̄_RS 3.0–16 times the IC's |
| 18 | SL entrains heavy-rain air more efficiently than the SC | 40 mm/hr: peak q̄_RS 5.4x; η_RSE ~3x after t=120 min |
| 401–402 | Rain onset at t=45, 55 and 40 min | Same |
| 412 | 3.2x and 3.9x at 1 mm/hr | 3.24 and 3.87 |
| 446–448 | IC max q̄_FS ~5% of the others'; SL and SC within 25% | 4.7% and 5.7%; SL 22% above SC |
| 463–465 | q̄_FS peaks at t=65 min (IC) and t=180 min (SL, SC) | Same |
| 466–468 | Without the limit, early species dominate the IC's q̄_FS | Since-rain species are at most 14% of it |
| 481–484 | IC max f_RS ~4x SL's, ~2x SC's (1–20 mm/hr); ~1.3x and ~4x at 40 mm/hr | 3.6–4.2; 1.7–2.1; 1.34; 3.80 |
| 487–490 | SL f_RS ~30–60% below SC's (1–20 mm/hr); SL's larger at 40 mm/hr | 40–58% (max), 39–44% (mean); yes |
| 503–505 | IC peak q̄_FS an order of magnitude below the others' | 18–21 times below |
| 609–610 | SC max q̄_FS and q̄_RS all below the SL's | 0.82; 0.84–0.93; 0.19 |
| 620 | IC max and mean q̄_FS and q̄_RS below the organized storms' | Yes, at every threshold |
| 622 | IC max N_FS-UL below 1% of either organized storm's | 0.41% (SL), 0.06% (SC) |
| 264–266 | No sensitivity to a 7–9 km threshold | SC/SL 6.6–6.9 (fixed-source), 4.5–7.7 (rain-sourced, per flux); IC/SL ≤ 3% |
