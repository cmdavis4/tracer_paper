# Fix checklist for the vRevisions 3 paper check

Each item comes from `paper_check_vRevisions3.md`. Tick an item by changing `[ ]` to `[x]`.

- **IDs** name the report item. For example, P1-E12 is Part 1, Errors, item 12. B marks a borderline item, N a number with no source, and F a figure or code note. Part 3 items keep their numbers.
- **Line numbers** follow the manuscript version you attached in chat on 7 October.
- **Order.** Do section A, then rerun the notebook (section B), then take replacement numbers from the rerun. The report's numbers predate the air-mass fix, so some will shift slightly (P3-6).

## A. Notebook and code (before the rerun)

- [x] **P1-E11, code.** Decide which Fig. 9 (a–c) to keep: the notebook's q̄_FS divided by elapsed time, or the PDF's undivided q̄_FS. For the PDF's version, remove the division in the Fig. 9 cell (index 41).
- [x] **P1-E5, code.** Decide whether the Fig. 3 cell (index 24) should compute buoyancy from all condensate, as Figs. 4–5 do, rather than from liquid only.
- [x] **P1-F.** Remove `ax.axvline(75)` from the Fig. 10 cell (index 43).
- [x] **P1-F.** For the Fig. 1 redraw: the hodograph axis labels in cell 19 are swapped.
- [x] **P2-F.** Fig. 11 (d): lower the y-axis floor in cell 53, so the IC's 40 mm/hr curve (peak 0.4) appears.
- [x] **P2-F.** Fig. 11: move the lower legend off the bottom row's x-axis labels.
- [x] **P2-F.** Fix the unused `ACC*_nc` columns in `blt.calculate_upper_level_tracer_dataframe`, which ships in the archive. Fixed on 8 October by removing the `ACC*` columns.
- [x] **P3-stale.** Delete or relabel cell 73, whose t = 100 min η_RP comparison is not the one the paper uses.
- [x] **P3-2 to P3-6.** Archive keys, the Fig. 4 condensate grid, `max_tracer_fraction_time` and the air mass. Fixed on 7 October; they take effect at the rerun.

## B. Rerun

- [x] Rerun `analysis/analysis-tracerpaper.ipynb` end to end.
- [ ] Replace the manuscript's figures with the rerun's output. Fig. 4's condensate contours change visibly, and the air-mass fix changes several other figures slightly.

## C. Manuscript numbers and claims (use the rerun's numbers)

- [x] **P1-E1.** Abstract lines 18–19, lines 397–402 and lines 645–646: the "12x" claim, the 40 mm/hr exception and "all but the lightest rain" no longer hold.
- [x] **P1-E9.** Lines 405–406: the IC's peak q̄_RS is at t = 65 min, not t = 90.
- [x] **P1-E10.** Lines 431–440: five of the six q̄_FS comparisons do not match (all but "within 3%").
- [x] **P1-E11, text.** After the Fig. 9 decision, update the text to match:
  - [x] Notebook's version: the caption, the (a–c) axis unit and line 433.
  - [x] PDF's version: lines 488–489.
  - [x] Either version: line 492, because no figure shows the normalized f_RS.
- [x] **P1-E12.** Lines 479–480 and 484: the comparable group is 1–20 mm/hr, and the heavy-rain group is 40 mm/hr alone. Conclusion line 626: "1-10" becomes 1–20.
- [x] **P3-1.** Lines 544–546: the 1.7 K is a 0–500 m mean. Say so, or quote 3.3 K for the lowest model level.
- [x] **P3-6.** Lines 551–553: "36%–40%" becomes 37%–40%, and "38%" becomes 39%.
- [x] **P1-E13.** Line 554: the 40 mm/hr η_RSE ratio over t = 120–180 min is 3.3, not ~4.
- [x] **P2-E1.** Lines 604–607: the "5x to 9x" range does not match the 10 km ratios. It probably comes from the 7 km block.
- [x] **P1-E14.** Lines 633–635: the third conclusion's η_RP half is wrong, because the SL/SC η_RP ratio is smallest at 40 mm/hr.
- [x] **P2-E2.** Lines 641–643: the claim that the SL lofts non-rainy air much better than the IC contradicts lines 599–601.
- [x] **P2-E3.** Lines 656–658: aloft, the SC lofts more heavy-rain air than the SL. The sentence holds only for entrainment into the updraft.

## D. Methods, captions, figure references and typos

- [x] **P1-E2.** Fig. 1 caption: only the winds differ between storms, not the low-level moisture.
- [x] **P1-E3.** Cite Weisman and Klemp (1982) for the thermodynamic profile and (1984) for the supercell's curved hodograph, consistently in the Fig. 1 caption, Table 1, lines 135–138, line 141 and lines 684–687.
- [x] **P1-E4.** Line 218: the dividing line ends at y = 162.9 km, not 162.5 km.
- [x] **P2-E4.** Eq. 2 (line 241): Δt is the 2-s output interval, but Table 2 defines it as the 1-s model timestep.
- [x] **P1-E5, text.** Fig. 3 caption (line 275): the buoyancy unit is m s⁻², not m² s⁻¹.
- [x] **P1-E6.** Fig. 4 caption: swap "lobe" and "cleft". Lines 286–287: swap the panel letters (f) and (i).
- [x] **P1-E7.** Line 345, Fig. 6 caption: the right-mover boundary appears in Fig. 5, not Fig. 3.
- [x] **P1-E8.** Line 381: "Table ." becomes "Table 4".

## E. References and data availability


- [x] **P1-E16.** Add Saleeby and Cotton (2004), Smagorinsky (1963), Lilly (1962) and Hill (1974), cited in Table 1.
- [x] **P1-E16.** Add Tompkins (2001), cited in the Fig. 3 caption as "Tompkins et al.", and check its author list.
- [x] **P1-E16.** Fix the third author of Grant et al. (2018), listed as "Heever, S. C. van den".
- [x] **P1-E16.** Add the journal, volume and pages for Kaimal et al. (1976) and Wakimoto (1985).
- [x] **P1-E16.** Fix the Maddox (1980) DOI, which holds a JSTOR URL.
- [x] **P2-E5.** Add Fan et al. (2010), Saleeby et al. (2016), Li et al. (2017) and Srivastava et al. (2018), cited at lines 578–580.
- [x] **P2-E5.** Remove the doubled "https://doi.org/" from the Foote and Fankhauser (1973) DOI.

## F. Borderline: decide whether to change

- [x] **P2-B3.** Lines 232–233: the upper-level sums are not "analogous" to N_FS and N_RS, because those count only updraft grid points.
- [x] **P1-B1.** Lines 305–306: at t = 170 min, the low-level and mid-level vorticity maxima are equal within 0.3%, not "greater than".
- [x] **P1-B2.** Line 397: the IC's 1 mm/hr peak q̄_RS is 26% of the SL's and 30% of the SC's ("~25%").
- [x] **P1-B3.** Lines 461–462: at 20 mm/hr, "within 20%" holds only with the SL as the reference.
- [x] **P1-B4.** Lines 537–540: Fig. 10 (d) shows rain area, not total rain rate.
- [x] **P2-B1.** Lines 591, 604–609 and 639–640: at 40 mm/hr, the SL–SC order reverses between the updraft and upper levels, and the text does not say so.
- [x] **P2-B2.** Lines 637–638: the SC entrains the least surface air only after q̄_FS is divided by elapsed time.
- [x] **P1-B5.** The text and Table 1 say RAMS v6.3.04, but the cited archive is v6.3.02.

## G. Numbers with no source

- [x] **P2-N.** Lines 249–251: the notebook no longer has the right-mover-only comparison. Remove the claim, or recompute it (about 2 hours).
- [x] **P1-N.** Lines 297–299: no output shows the couplet at t ≈ 60 min or the split at t ≈ 75 min.

## H. Final edits
- [ ] Incorporate all changes from proofs
- [x] Fix citation format in new lapse rate paragraph
- [ ] Check that all bibliography references propagated
- [ ] - [ ] **P1-E15.** Lines 693–697 and the Davis and van den Heever (2025) reference:
  - [ ] Publish a new archive version that holds the revised runs' namelists and the revised notebook.
  - [ ] Update the reference's version ("v1.0-acp").
  - [ ] Make the two DOIs consistent: …15659853 in the text and …15659854 in the reference list.
  - [ ] Point "Interactive computing environment" at the revised notebook.
-
- [ ] Clean up tracked changes
- [ ] Draft email