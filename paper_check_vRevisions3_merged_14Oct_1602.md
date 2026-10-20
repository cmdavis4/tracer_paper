# Check of "Davis et al tracer paper vRevisions3 merged.pdf" (14 October, 16:02 export)

Tick an item by changing `[ ]` to `[x]`. Section 5 needs no action.

## What was compared

- **Paper:** `Davis et al tracer paper vRevisions3 merged.pdf`, saved 14 October at 16:02. Line numbers below follow it.
- **Earlier check:** `paper_check_vRevisions3_merged_14Oct.md`, which covered the 10:45 export.
- **Numbers:** the notebook wrote Fig. 11's figure data at 15:52. At all four heights, that data is identical to my rerun of the edited cells from the caches. Fig. 9's figure data matches the in-updraft rerun from the 10:45 check. Every number below comes from these two sources.
- **Figures:** I compared every figure in the PDF with its PNG in `analysis/paper_figures/`.
- **Scope:** every sentence that changed since 10:45, every number in Sects. 2.4 and 3.4, and the 7–9 km sensitivity claim.

**Summary**

- Every new number in Sects. 2.4 and 3.4 matches the right-mover data.
- Two errors from the 10:45 check remain. Lines 224–226 still say that Sect. 3.4 includes the left mover's tracer, and Fig. 4's legend is still swapped (section 1).
- The abstract, lines 626–627 and Conclusion 4 state the supercell's advantage aloft without the 40 mm/hr exception (section 2).
- Fig. 8's new colour scale saturates for about a quarter of the organized storms' fixed-source values (section 2).

---

## 1. Errors

### 1. Lines 224–226 contradict Sect. 2.4

- [ ] Rewrite or delete "We do include the tracer emitted in the left-mover region in the results assessing upper-level transport of tracers (Section 3.4), the reasoning for which is discussed in Section 2.4."

- Sect. 3.4 now excludes the left mover from all three parts of each ratio: the tracer aloft, F_upward and E_RS. Cell `7556113a` sums E_RS over the right mover only.
- Lines 257–260 say that Sect. 3.4 uses the right mover only. Sect. 2.4 now justifies excluding the left mover, not including it.
- One fix: extend lines 220–221 ("We do not include this emitted tracer in any of the results … (Section 3.3) that are functions of the cumulative amount of tracer emitted") to Sections 3.3 and 3.4, and delete lines 224–226. Lines 221–224, on left-mover tracer that the right mover entrains, then cover both sections.

### 2. Fig. 4's legend swaps the two cross-section lines

- [ ] Rerun the Fig. 4 cell (`44ed0846`) and replace the figure.

- The PDF still has the 12 October PNG. The cell has not run since then.
- In the plan views (a–c), the orange dash-dot line is at y = 182 km. The gold dashed line is at y = 172 km.
- The row headers match the lines: "Upper cross section" (d–f) is orange, and "Lower cross section" (g–i) is gold.
- The legend swaps them. It labels the gold line "Upper cross-section location" and the orange line "Lower cross-section location".
- A reader who uses the legend would place panels (d–f) at y = 172 km. That section is the lobe, which lines 297–298 assign to panel (i).
- The current cell builds the legend from the plotted lines, so a rerun fixes it.

---

## 2. Borderline

- [ ] **The 40 mm/hr exception appears only in Sect. 3.4.** Sect. 3.4 now excludes the 40 mm/hr tracer from "7x - 9x" and "14x – 19x". Three summary statements have no exception:
  - the abstract, line 19: "the supercell is significantly more effective as a transporter of rainy surface air to the upper troposphere";
  - lines 626–627: "supercells are significantly more effective … at transporting air from both rainy and non-rainy surface regions";
  - Conclusion 4, line 656: "it transports far more of this air to the upper troposphere".

  At 40 mm/hr, the supercell lofts 1.4 times the squall line's tracer per unit upward flux, and 2.4 times per unit emitted. So "more" holds at every rain rate. "Significantly" and "far more" fit 1–20 mm/hr only.

- [ ] **Fig. 8 (e–f) saturate.** The fixed-source row's colour scale now ends at 10⁶ # kg⁻¹.
  - 21% of the squall line's in-updraft values exceed 10⁶, and so do 28% of the supercell's. The largest are 4×10⁷ and 3×10⁷.
  - The colour bars have no arrow for values above the scale, so the saturated orange reads as 10⁶.
  - The text quotes no value from these panels. A top of 10⁸, or `extend="max"`, would show the range.

---

## 3. Typo

- [ ] **Line 731 (Financial support):** "Postdoc. Mobility" has a space. The Acknowledgements (line 726) and the grant's name have none.

---

## 4. Still open in `paper_check_vRevisions3_checklist.md`

The Zenodo items in its section H are unchanged. The reference (lines 775–777) still names version "v1.0-acp". The text cites DOI …15659853, and the reference list cites …15659854.

---

## 5. What matched

- **Figures:** Figs. 2, 3, 7, 8, 9, 10 and 11 match the notebook's PNGs pixel for pixel. Figs. 1, 4, 5 and 6 differ only in resolution. Fig. 8 shows the since-rain q̄_FS, and Fig. 11 shows the right mover. Fig. 9's axis labels and Fig. 11's label spacing are fixed.
- **Earlier items:** apart from section 1, every item from the 10:45 check is in the PDF.

| Lines | Paper | Data |
|---|---|---|
| 250 | F_t-RM uses (300 s) | The 5-minute output interval |
| 257 | f_t-flux = f_t-N within 6% | Largest gap 6.0%, at 40 mm/hr |
| 363–364, Table 3, 397–401 | N_FS counts the species emitted since rain began, at t=40, 50 and 35 min | Same. These are the start times of the 5-minute emission windows in which rain began |
| 609–610, 654–655 | The SC's maximum and end-of-run q̄_FS and q̄_RS are all below the SL's | Maximum: 0.82 (FS), 0.84–0.93 (1–20 mm/hr), 0.19 (40). End: 0.82, 0.93–0.99, 0.11 |
| 609–611 | ~7x fixed-source and 7x–9x rain-sourced tracer per unit flux, excluding 40 mm/hr | 6.8; 7.0–9.0 |
| 611–613 | 14x–19x per unit emitted, excluding 40 mm/hr | 13.7–18.7 |
| 613 | The SC's maximum 40 mm/hr q̄_RS is ~20% of the SL's | 19% |
| 613–615 | At 40 mm/hr, the SC's maximum and end-of-run N_RS-UL exceed the SL's under both normalizations | Per unit flux: 1.40 (both). Per unit emitted: 4.2 (maximum), 2.4 (end) |
| 621–623 | From in-updraft to upper-level maxima, the IC's gap grows by one to two orders of magnitude | 5–28 times (SL), 94–222 times (SC) |
| 623–624 | The IC's maximum N_FS-UL is < 1% of either organized storm's | 0.41% (SL), 0.06% (SC) |
| 264–266 | The 7–9 km thresholds change no conclusion | SC/SL at the end of the run: fixed-source 6.9–7.1; rain-sourced 6.9–9.2 per unit flux and 8.7–15.4 per unit emitted (1–20 mm/hr); 40 mm/hr 1.40–1.47 per unit flux. Only "14x – 19x" is specific to 10 km |
| 941–942, 1011 | Peters et al. (2020), Wakimoto (1985) | Journal, volume and pages present |
