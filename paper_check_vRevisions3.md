# Check of "Davis et al tracer paper vRevisions 3 pre check.pdf" against the analysis notebook

Part 1 covers everything except upper-level transport. Part 2, at the end, covers upper-level transport.

# Part 1: Everything except upper-level transport (6 October)

## What was compared

- **Paper:** I checked the PDF in the repository root. The version you attached in chat on 7 October is newer, and its Sect. 3.4 and fourth conclusion are rewritten. Every passage that Part 1 cites reads the same in both versions. Page and line numbers below follow the newer version.
- **Notebook:** `analysis/analysis-tracerpaper.ipynb` at commit `aa5dab7`. I used its saved outputs and its figure-data pickles in `analysis/paper_figures/figure_data/`. I did not run or edit the notebook.
- **Model setup:** each run's RAMSIN namelist and the RAMS source in `replication/RAMS_tracer_source`.
- **Earlier versions:** the notebooks at git tags `v1.0-acp` and `vRevisions1-acp`. Both tags use the same original runs and give identical statistics.

**Abbreviations.** IC is the isolated convective storm, SL the squall line, and SC the supercell.

**"Stale"** marks a paper value or statement that matches the original runs but not the current ones. The IC and SL runs changed and the SC run did not, so a supercell-only number cannot be stale.

**Skipped:**

- Sections 2.4 and 3.4, Fig. 11, the fourth conclusion and the upper-level sentences at the end of Sect. 2.3. Part 2 covers these.
- The citation format in the boundary-layer paragraph.
- The layout of Fig. 1.
- Numbers quoted from other papers.

**Not listed means it matched.** Everything not listed below matched at the precision the paper gives. That includes Table 1, the Sect. 2 setup numbers, Sect. 3.2, Sect. 3.3.5 and most of Sect. 3.3.6.

**No large-scale mismatch.** No storm or figure failed to match as a whole. The one exception is Fig. 9 (a–c), which item 11 diagnoses.

---

## 1. Errors

### 1. The gap between the isolated storm and the organized storms barely grows with rain rate any more
*Abstract lines 18–19; Sect. 3.3.2 lines 397–402; Conclusions lines 645–646*

**The 12x claim.** Lines 397–399 say the organized storms' maximum q̄_RS grows relative to the IC's "by factors as large as 12x" as the threshold rises to 20 mm/hr. The notebook gives at most 5.2:

| Max q̄_RS ratio | 1 mm/hr | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|
| SL / IC | 3.9 | 4.2 | 4.6 | 5.2 | 16.1 |
| SC / IC | 3.3 | 3.7 | 4.2 | 4.9 | 3.0 |

The 12x might come from the SL/IC time-mean ratio at 20 mm/hr, which is 12.9. But the sentence describes maxima, and the SC/IC time-mean ratio there is 14.1.

**The 40 mm/hr exception.** Line 399 says the trend "does not hold for the 40 mm hr-1 tracer". That is true only for the SC. The SL ratio rises to 16.1.

**"All but the lightest rain."** The abstract says the organized storms entrain "significantly more rainy air than the isolated convective storm for all but the lightest rain". Lines 645–646 say the differences are "less pronounced" for light rain. The current numbers do not support either statement:

- At 1 mm/hr the organized storms reach 3.3–3.9 times the IC's maximum and 7.1–9.4 times its time-mean.
- The smallest ratio of all is SC/IC at 40 mm/hr (3.0).

Lines 400–402 still hold in direction, but the change from 1 to 20 mm/hr is small.

**Stale.** In the original runs the 1 mm/hr ratios were 1.9 (SL/IC) and 2.1 (SC/IC), and SL/IC rose to 6.1 at 20 mm/hr.

**Source:** the "Tracer entrainment statistics" cell (index 67), rows `max_mean` and `mean_mean`, saved in `entrainment_factor_stats.pkl`.

### 2. The Fig. 1 caption says low-level moisture varies between storms
*Fig. 1 caption, p. 5*

The caption says "The low-level moisture and shear profile were varied between storms". All three RAMSINs use the same thermodynamic profile: 13 g/kg in the lowest 1250 m, with θ well mixed up to 1000 m. Only the winds differ, which is what lines 136–137 and Table 1 say.

**Stale.** The original SL run used a different sounding.

### 3. The Weisman–Klemp profile is dated 1984 in some places and 1982 in others
*Fig. 1 caption; Table 1; lines 135–138; line 141; lines 684–687*

- The Fig. 1 caption, Table 1 and lines 135–138 attribute the profile to Weisman and Klemp (1984).
- Line 141 calls it "WK82".
- Lines 684–687 cite it as "(Weisman and Klemp, 1982)".

Pick one year for the profile. The analytic thermodynamic profile comes from the 1982 paper.

### 4. The right-mover dividing line ends at the wrong y
*Line 218*

The paper says the line runs to "(x=424.5km, y=162.5km)". `blt.sc_wk_right_mover_boundary` defines it as y = 78 km + 0.2x, which gives y = 162.9 km at x = 424.5 km. This formula has not changed since `v1.0-acp`.

### 5. The cold-pool buoyancy has the wrong units, and Fig. 3 computes it differently from Figs. 4–5
*Fig. 3 caption, line 275*

- **Units.** The caption reads "-0.04 m2s-1 buoyancy contour". The code computes B = g(θ_ρ − ⟨θ_ρ⟩)/⟨θ_ρ⟩, whose unit is m s⁻².
- **Calculation.** The two figures compute θ_ρ from different condensate:
  - The Fig. 3 cell (index 24) uses liquid condensate only: `R_liquid`, cloud plus rain.
  - Figs. 4 and 5 use the `blt` buoyancy, which subtracts all condensate (`RTP − RV`).
  - Both captions say the boundary is "defined as in Figure 3".

  The contour probably moves little, but the two definitions are not the same.

### 6. The Fig. 4 caption swaps cleft and lobe, and lines 286–287 cite the wrong panels
*Fig. 4 caption, p. 16; lines 285–287*

**What the figure shows.** In the code (cell index 26), the cleft is at y = 182 km and plots in row (d–f). The lobe is at y = 172 km and plots in row (g–i). The figure's row labels and legend agree with this.

**The caption** says the reverse: "(d-f) ... lobe ..., (g-i) ... cleft".

**Lines 286–287** cite "(Fig. 4 (f))" for the continuous updraft in the lobe. They cite "(Fig. 4 (i))" for the cleft updraft that a downdraft nearly breaks. The panels show the opposite:

- (i) has the continuous updraft.
- (f) has the updraft that a downdraft interrupts.

The physical descriptions are right. Only the panel letters and the caption labels need swapping.

### 7. The Fig. 6 caption points to the wrong figure for the right-mover boundary
*Line 345*

The caption says the right mover "is shown as in Fig. 3". The right-mover boundary appears in Fig. 5. Fig. 3 is the isolated storm.

### 8. A table number is missing
*Line 381*

"in Table ." should read "in Table 4".

### 9. The IC peak q̄_RS time is stale
*Lines 405–406*

The paper says the IC reaches its maximum q̄_RS "for all rain-sourced tracer species at t=90 minutes". The notebook gives t = 65 min for all five tracers, and line 569 also says "about t=65 minutes".

**Stale.** The original runs gave 90–95 min.

### 10. Most q̄_FS comparisons in Sect. 3.3.3 do not match
*Lines 431–440*

| Paper | Notebook |
|---|---|
| The IC maximum is "a factor of ~3 smaller" than the SL's and the SC's | 2.5 times smaller than the SL's; 1.85 times smaller than the SC's |
| The SC and SL maxima are "within 25% of each other" | The SC's is 26% below the SL's (the SL's is 35% above the SC's) |
| The IC reaches its maximum "at t=65 minutes" | t = 60 min |
| After dividing by elapsed time, the IC and SL maxima are "within 3%" | 2.2% (matches) |
| After that division, the SC is "~25% less than both" | 38% less than the IC and 36% less than the SL |
| "all three ... within ~25% of each other" | The SC is 38% below the IC |

**Values behind the table:**

- Undivided maxima: IC 8.65e5 (t = 60), SL 2.16e6 (t = 180), SC 1.60e6 (t = 180), in # kg⁻¹.
- Time-divided maxima: 1.44e4, 1.41e4 and 0.90e4, in # kg⁻¹ min⁻¹.

These values do not match the original runs either, so they are not simply stale.

**Source:** cell index 67, `total_fs_tracer` column, rows `max_mean`, `max_mean_time` and `max_mean_normalized`.

### 11. Fig. 9 (a–c) in the PDF and in the notebook show different quantities, and the text cites both
*Fig. 9; lines 433, 488–489, 492*

**The PDF version.** Fig. 9 (a–c) in the PDF shows q̄_FS. Its peaks are about 0.9e6, 2.2e6 and 1.6e6 # kg⁻¹, and this matches the caption.

**The notebook version.** The notebook's Fig. 9 cell (index 41, `rs_factors_fs_entrainment`) divides q̄_FS by elapsed minutes. Its peaks are 1.44e4, 1.41e4 and 0.90e4, but its axis is still labelled "# kg⁻¹". The notebook has done this since its first commit (`8ecffa1`). The PDF figure therefore comes from earlier code, which I could not identify. Rows (d–l) are the same in both versions.

**The text cites both versions:**

- **Line 433** cites Fig. 9 (a–c) for the undivided maxima.
- **Lines 488–489** cite it for the IC's peak being "comparable to the largest proportion reached by either of the more organized storms". That is true only after dividing by time (1.44e4 against 1.41e4). The undivided values are 0.86e6 and 2.16e6.
- **Line 492** cites Fig. 9 (d–f) for the IC's "smaller normalized f_RS". Fig. 9 (d–f) shows the undivided f_RS, in which the IC and SL are similar: SL/IC is 0.94–1.25 up to 20 mm/hr. No figure shows the normalized f_RS.

**Fix.** Decide which version of (a–c) to keep:

- **The notebook's version.** Update the caption, the axis unit (# kg⁻¹ min⁻¹) and line 433.
- **The PDF's version.** Revert the division in cell 41, and reword lines 488–489.

Line 492 needs a change either way, because no figure shows the normalized f_RS.

### 12. Sect. 3.3.4 and the second conclusion group the 20 mm/hr tracer with heavy rain
*Lines 479–487; Conclusions line 626*

**How lines 479–487 group the tracers.** Lines 479–480 say the SL and SC reach comparable q̄_RS for "the three rain-sourced tracers corresponding to the rainfall rates less than 20 mm hr-1". Those three tracers are 1, 5 and 10 mm/hr. Line 484 then contrasts them with "the tracers emitted at heavier rainfall rates", which leaves 20 and 40 mm/hr. Lines 484–487 make two claims about those two tracers:

1. The SL reaches a greater q̄_RS than the SC.
2. The SL has the larger f_RS, because "a larger fraction of this surface air originates in regions of heavy rain".

**What the notebook gives at 20 mm/hr.** Neither claim holds:

1. The SL/SC q̄_RS ratio is 1.06 at the maximum, 1.06 at the end of the simulation and 0.91 averaged over time. The storms are comparable.
2. The SL/SC f_RS ratio is 0.48, so the SL's fraction is about half the SC's.

Both claims hold only for the 40 mm/hr tracer.

**Three other passages already put 20 mm/hr in the comparable group:**

- Lines 411–412 say q̄_RS is similar for the 1, 5, 10 and 20 mm/hr tracers.
- Lines 471–475 say the SL's f_RS is ~30–55% less than the SC's for 1–20 mm/hr, and larger only at 40 mm/hr.
- Lines 558–559 say q̄_RS is comparable for ≤ 20 mm/hr.

**Fix.** Lines 479–480 should name the four tracers at 1–20 mm/hr, and line 484 should name only the 40 mm/hr tracer. The second conclusion (line 626) says "1-10 mm hr-1", and it should say 1–20 mm/hr for the same reason.

**Stale.** In the original runs the SL/SC q̄_RS ratio at 20 mm/hr was 1.9 at the maximum and 2.5 averaged over time. The 20 mm/hr tracer then belonged with the heavy-rain tracer, and these passages still describe those runs.

### 13. The 40 mm/hr η_RSE ratio is 3.3, not ~4
*Line 554*

The paper says the SL's η_RSE for the 40 mm/hr tracer, averaged from t = 120 min onward, "is ~4x that of the supercell's". The notebook gives 3.3 for t = 120–180. The value 3.9 is the t = 150–180 average (row `t150_to_t180_mean_entrainment_efficiency`). The 38% and 47% earlier in the same paragraph use t = 120–180.

The original runs gave 9.0.

### 14. The third conclusion's η_RP claim is stale
*Lines 633–635*

The paper says that at 40 mm/hr, the SL's η_RP and its entrained fraction are both "more pronounced" relative to the SC's.

**The η_RP half is wrong.** The SL/SC η_RP ratio is smallest at 40 mm/hr:

- At the end of the simulation it is 1.59, against 1.82–1.95 for 1–20 mm/hr.
- Averaged over time it is 1.88, against 1.94–2.08 for 1–20 mm/hr.

**The η_RSE half holds.** Over t = 120–180 the ratio is 3.3, against 0.38–0.47.

**Stale.** In the original runs the SL/SC η_RP ratio was 3.3 at 40 mm/hr and 0.9–1.5 below it.

### 15. The code archive does not contain the revised runs
*Code and data availability, lines 693–697; reference Davis and van den Heever (2025)*

**The archive version.** The reference pins version "v1.0-acp".

**The namelists.** The README says the archive ships namelists in `data/{ic,sl,sc}_wk/`. The local copies are `replication/data/*/RAMSIN.*`, dated 13 June 2025, and they describe the original runs:

- The IC namelist has BTHP = 4.0, but Table 1 says 2 K.
- The SL namelist has a different input sounding.

The revised runs' namelists are in neither place: `ic_wk/validation/other_forcing_bthp-2.0_radius-2x/RAMSIN` and `sl_wk/validation/sc_profile/RAMSIN`. Until a new archive version is published, the statement that the output "can be reproduced from the available materials" is not true.

**The notebook.** The "Interactive computing environment" section points to the archive's `analysis-tracerpaper.ipynb`, which is the original notebook. The revised notebook is `analysis/analysis-tracerpaper.ipynb`.

**The DOIs.** The text cites DOI `…15659853` and the reference list cites `…15659854`. These are probably the concept DOI and the version DOI. Make both point at the new version, or make them consistent on purpose.

### 16. Reference-list problems

**Cited but missing from the list:**

- Saleeby and Cotton (2004), Smagorinsky (1963), Lilly (1962) and Hill (1974), all cited in Table 1.
- Tompkins et al. (2001), cited in the Fig. 3 caption. I believe this paper has a single author (Tompkins, 2001, J. Atmos. Sci.), so the in-text form may also need to change.

**Malformed entries:**

- **Grant et al. (2018):** the third author appears as "Heever, S. C. van den". This is the same author-parsing problem you described earlier.
- **Kaimal et al. (1976) and Wakimoto (1985):** neither entry has a journal, volume or pages.
- **Maddox (1980):** the DOI field holds a JSTOR URL ("https://doi.org/http://www.jstor.org/…"), so the link is broken.

---

## 2. Borderline approximations

1. **Fig. 5 (f) vorticity, lines 305–306.** The paper says low-level vorticity is "greater than" mid-level vorticity at t = 170 min. Within the panel's area, the maxima are 0.0333 s⁻¹ for the 0–1 km mean and 0.0332 s⁻¹ at 5 km, equal to within 0.3%. Source: the `f05` pickle.
2. **IC peak q̄_RS share, line 397.** The paper says the IC's maximum q̄_RS for the 1 mm/hr tracer is "~25%" of either organized storm's. The notebook gives 26% of the SL's and 30% of the SC's.
3. **"Within 20%", lines 461–462.** The paper says the SL and IC maximum f_RS values are "within 20%" for tracers up to 20 mm/hr.
   - At 20 mm/hr the SL's value is 25% above the IC's, which is the same as the IC's being 20% below the SL's.
   - For 1–10 mm/hr the two are within 16%.
4. **Fig. 10 (d) as evidence for total rainfall rate, lines 537–540.** These lines cite Fig. 10 (d) for the IC's "total rainfall rate", but that panel shows rain area.
   - The total rain rate is not plotted. It peaks at t = 70 min (cell index 76).
   - The 1 mm/hr rain area peaks at t = 75 min. The 5 and 10 mm/hr areas peak at t = 70 min, and the 20 and 40 mm/hr areas at t = 65 min.
5. **RAMS version.** The text and Table 1 say RAMS v6.3.04, but the cited archive, van den Heever et al. (2022), is v6.3.02. This may be intentional.

---

## 3. Numbers with no source found

- **Lines 297–299.** The SC's vorticity couplet forms "at around t=60 minutes" and the storm splits "at about t=75 minutes". The paper marks both "not shown", and the notebook has no output for either.

---

## 4. Figure files: PDF compared with the notebook's current output

I compared the images embedded in the PDF with the PNGs in `analysis/paper_figures/` pixel by pixel. Where the sizes differed, I trimmed margins first and then compared by eye.

- **Identical:** Figs. 2, 3, 7 and 8. Fig. 6 also matches, at the PDF's lower resolution.
- **Same data, older layout:** Figs. 1, 4 and 5. Only the tick spacing and panel sizes differ.
- **Fig. 9:** rows (d–l) are identical. Panels (a–c) differ, as item 11 describes.
- **Fig. 10:** the notebook draws a vertical line at t = 75 min in panels (d–f), from `ax.axvline(75)` in cell index 43. The PDF version has no such line. Remove that call before you regenerate the figure.
- **Fig. 1, for the planned redraw:** cell index 19 labels the hodograph x-axis "V (m/s)" and the y-axis "U (m/s)". MetPy's `plot_colormapped(u, v, …)` puts u on the x-axis, so the labels are swapped.

---

# Part 2: Upper-level transport (7 October, newer version)

## What was compared

- **Paper:** the version you attached in chat on 7 October. This part covers:
  - the end of Sect. 2.3 and all of Sect. 2.4;
  - Sect. 3.4 and the Fig. 11 caption;
  - the fourth conclusion and the transport claims in lines 644–658.

  As you asked, I did not compare the attached figure images with the notebook's output.
- **Notebook:** `analysis/analysis-tracerpaper.ipynb` as of 7 October, which is commit `1220749` plus uncommitted edits. I used:
  - its saved outputs;
  - the per-storm upper-level caches;
  - `analysis/paper_figures/figure_data/f11_tracer_entrainment_{7,8,9,10}km.pkl`.
- **Part 1 still stands.** The rerun left the entrainment statistics and Figs. 1–10 unchanged.

All values are for z = 10 km, unless a line says otherwise. A storm's "peak" is its maximum over the simulation. For the SL and the SC, every upper-level quantity peaks at the end of the simulation.

**Overview.** Every number in the rewritten Sect. 3.4 matches the notebook, except one range (item 1). Two sentences in the Conclusions contradict Sect. 3.4 (items 2 and 3).

---

## 1. Errors

### 1. The "5x to 9x" range does not match z = 10 km
*Lines 604–607*

The text says the SC's upper-level transport of rain-sourced tracer peaks at "5x to 9x" the SL's, "across all rainfall rate thresholds". It says this holds both per unit upward flux and per unit emitted. At 10 km, the ratios are:

| SC / SL, peak values | 1 mm/hr | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|
| Per unit upward flux (Fig. 11 e, f) | 7.8 | 6.9 | 6.2 | 6.2 | 4.2 |
| Per unit emitted (Fig. 11 h, i) | 13.4 | 12.0 | 10.7 | 9.6 | 3.6 |

- At 40 mm/hr, both ratios are below 5.
- Per unit emitted, every ratio for 1–20 mm/hr is above 9.
- **Probable source:** at z = 7 km, the ratios for 1–20 mm/hr span 5.7–9.3. Cell index 69 prints the 7 km block first.

Source: the "Supercell / Squall line" tables in the z = 10 km block of cell index 69.

### 2. The fourth conclusion's claim about the squall line contradicts lines 599–601
*Lines 641–643; lines 599–601*

- **The conclusion** calls the SL "a significantly more effective transporter of rainy and non-rainy surface air to the upper troposphere" than the IC.
- **Lines 599–601** say the two storms' peak N_FS-UL/F_upward values are within 20% of each other. The notebook gives 10%: 1.18e5 # kg⁻¹ for the SL and 1.08e5 for the IC.
- **For rainy air, the claim holds.** Depending on the rain rate, the SL's peak N_RS-UL/F_upward is 60–325 times the IC's.
- **For non-rainy air, the SL leads in only two ways:**
  - **At the end of the simulation, by 3.6 times.** This lead mostly reflects the IC's growing F_upward. From the IC's peak at t = 70 min to the end, its F_upward grows 3.8-fold, but its N_FS-UL grows only 13%. More than a third of that F_upward accrues after t = 120 min, when the IC's updraft mass is under 1% of its peak.
  - **In absolute amount, by 17 times.** Fig. 11 does not show absolute amounts.

### 3. Lines 656–658 contradict Sect. 3.4
*Lines 655–658; lines 604–612 and 636–643*

- **What the paper says.** Lines 656–658 say the SL's "potential for the transport of air containing rain-aerosolized bioaerosols is the greatest of the three storm morphologies if this aerosolization requires heavy rain".
- **What Sect. 3.4 and the fourth conclusion show.** The SC lofts more rainy air than the SL at every rain rate. At 40 mm/hr, the SC's peak is 4.2 times the SL's per unit flux, 3.6 times per unit emitted and 2.45 times in absolute amount.
- **When the sentence holds.** It holds if "transport" there means entrainment into the updraft. There, the SL's peak 40 mm/hr q̄_RS is 5.4 times the SC's (Part 1).

### 4. Eq. 2 uses the wrong Δt
*Eq. 2, line 241; Table 2*

Table 2 defines Δt as the 1-s model timestep. Eq. 2 sums over the 2-s output, so its Δt must be 2 s. The code uses 2 s (`LITE_OUTPUT_INTERVAL_SECONDS`). With Table 2's Δt, F_upward would come out half as large.

### 5. Reference-list problems in these sections

- **Missing from the list:** Fan et al. (2010), Saleeby et al. (2016), Li et al. (2017) and Srivastava et al. (2018), all cited at lines 578–580.
- **Broken DOI:** the entry for Foote and Fankhauser (1973) starts its DOI field with "https://doi.org/" twice.

---

## 2. Borderline

1. **The heavy-rain reversal goes unmentioned (lines 591, 604–609 and 639–640).**
   - **What the paper says.** Line 591 says the rain-sourced results aloft "broadly parallel trends in q̄_RS". Lines 639–640 say these trends "differ less" from the in-updraft trends than the fixed-source trends do.
   - **The reversal.** At 40 mm/hr, the order of the SL and SC reverses. In the updraft, the SL's peak q̄_RS is 5.4 times the SC's. Aloft, the SC's peak is 4.2 times the SL's per unit flux.
   - **Line 607** says the organized storms' peak q̄_RS values "were much closer than this". That holds at 1–20 mm/hr, where they are within 17%, but not at 40 mm/hr.
   - **The consequence.** From the updraft to upper levels, the SC/SL ratio rises 22-fold at 40 mm/hr. For fixed-source tracer it rises 8- to 9.5-fold. So for heavy rain, the rain-sourced trend changes more, not less.
2. **"The supercell entrains the least proportion of surface air" (lines 637–638).**
   - This holds after dividing q̄_FS by elapsed time, as lines 592–593 do. The SC's time-normalized peak is then 38% below the IC's and 36% below the SL's.
   - Undivided, the IC has the lowest peak q̄_FS (8.6e5 # kg⁻¹, against 1.6e6 for the SC) and the lowest time mean.
   - Which reading a reader takes depends on Fig. 9 (a–c). Part 1 item 11 describes the two versions of those panels.
3. **N_FS-UL and N_RS-UL are said to be calculated "analogously to N_FS and N_RS" (lines 232–233).**
   - **The definitions.** Table 3 sums N_FS and N_RS over in-updraft gridpoints only. Eq. 1 and the code sum every gridpoint at z ≥ 10 km.
   - **The size of the difference.** At t = 180 min, in-updraft gridpoints hold 17% of the SC's fixed-source tracer above 10 km, and none of the IC's.

---

## 3. No source found

- **The right-mover-only comparison (lines 249–251).** The notebook no longer contains this comparison. The note at the top of its "Sensitivity tests" section says those cells now repeat the full-domain calculation.
  - **Why it matters more now.** The new conclusions rest on the SC's lead over the SL. That lead could depend on what the left mover lofts.
  - **The cost.** The right-mover flux requires rereading the SC's 5401 lite files, which takes about 2 hours.

---

## 4. Claims that still hold

- **Methods.** The end of Sect. 2.3 and Sect. 2.4 describe the code correctly, apart from Error 4 and Borderline 3.
- **Fixed-source comparisons (lines 592–601):**
  - The SC has the lowest time-normalized peak q̄_FS.
  - The SC's end value is 6.0 times the SL's ("~6x") and 6.6 times the IC's peak ("~7x").
  - The IC and SL peaks are within 10% ("within 20%").
- **The IC's rain-sourced transport (lines 602–604, 610–612, 646–648 and 653–654):**
  - Its N_RS-UL/F_upward peaks briefly, at t = 100 min.
  - It stays far below both organized storms at every rain rate. At 40 mm/hr, 0.001% of the IC's emitted tracer reaches 10 km, against 0.37% for the SL and 1.3% for the SC.
- **The SL and SC in the updraft (lines 604–608).** Their peak q̄_RS values are within 17% at 1–20 mm/hr. Their peak η_RSE values differ by a factor of 1.8–2.4.
- **The SC's lead (lines 610–612 and 636–641):**
  - The SC lofts the most surface air per unit flux.
  - It also lofts the most relative to its own entrainment. At 1 mm/hr, its fraction aloft (11.6%) exceeds its η_RSE (8.2%). The SL's fraction aloft (0.87%) is a fifth of its η_RSE (4.2%).
- **Height sensitivity (lines 252–254).** Every claim in this list holds at 7, 8, 9 and 10 km. Across those heights, the SL's peak N_FS-UL/F_upward exceeds the IC's by 10–20%.

---

## 5. Figure and code notes

- **Fig. 11 (d) hides one curve.**
  - The IC's 40 mm/hr N_RS-UL/F_upward peaks at 0.40 # kg⁻¹, which is below the axis floor of 1. The floor is set by `ax.set_ylim([1e0, 1e5])` in cell index 53.
  - The caption says (d–i) show all five tracers.
- **Fig. 11 legend overlap.** In the current output, the lower legend overlaps the bottom row's x-axis labels. The attached version shows the same overlap.
- **Wrong, but unused, `ACC*_nc` columns.**
  - **The bug.** `blt.calculate_upper_level_tracer_dataframe` converts the accumulated emission with the air mass at `isel(z=1)`, but only after it has cut the dataset to z ≥ threshold. That index is the second level above the threshold. The docstring says it is the lowest level above ground.
  - **The effect.** Fig. 11 does not use these columns, because cell index 51 takes E_RS from the full dataset. No number in the paper changes.
  - **Why it matters.** The function ships in the code archive.

---

# Part 3: Code review (7 October)

## What was reviewed

- **Code:** `analysis/analysis-tracerpaper.ipynb` (commit `1220749` plus uncommitted edits) and `src/blt_utils/__init__.py`.
- **What I looked for:** errors that change a number in the paper, a figure, or the archived figure data.
- **Already reported:** Parts 1 and 2 list several code problems, so this part does not repeat them:
  - the time division in Fig. 9 (a–c) (Part 1 item 11);
  - the liquid-only buoyancy in Fig. 3 (Part 1 item 5);
  - the stray t = 75 min line in Fig. 10 and the swapped hodograph labels in Fig. 1 (Part 1 section 4);
  - the clipped curve and legend overlap in Fig. 11, and the unused `ACC*_nc` columns (Part 2 section 5).

---

## 1. Errors that affect the paper

### 1. The "surface" temperature difference is a 0–500 m average
*Lines 544–546; cell index 65 ("IC surface temperature by rain intensity")*

- **What the paper says.** At t = 65 min, the "average surface potential temperature" under 40 mm/hr rain is ~1.7 K lower than under 1 mm/hr rain.
- **What the code computes.** The cell averages the potential temperature deficit over z = 0–500 m. That gives 1.74 K, the paper's number.
- **At the lowest model level (12 m)**, where tracer is emitted, the difference is 3.3 K. The deficit there is 7.2 K under 40 mm/hr rain and 3.9 K under 1 mm/hr rain.
- **Fix.** Either describe the number as the lowest-500-m mean, or quote 3.3 K for the surface.

### 2. The archived data for Fig. 9 (d–l) holds only the 40 mm/hr curve
*Cell index 41*

- **The bug.** In rows (d–l), the loop over the five rain-sourced tracers saves every curve under the same key, such as `storm-sl_wk_rs_fraction`. Each tracer overwrites the previous one, so each storm keeps only the 40 mm/hr curve.
- **Why it matters.** The paper says the archive includes "the underlying data presented in all figures".
- **Status: fixed.** Each key now includes the tracer name, for example `storm-sl_wk_tracer-TRACERP001_rs_fraction`.

### 3. The archived data for Figs. 3 and 4 is labelled with the wrong storm
*Cell indices 24 and 26*

- **The bug.** Both cells build their keys from a variable `storm` that they never set. It keeps the value "sc_wk" from an earlier loop, so every key in the Fig. 3 and Fig. 4 data files starts with `storm-sc_wk`. The data themselves are the right storms'.
- **Related gaps:**
  - The plan-view cold-pool contours in Fig. 4 are drawn without `plot_and_save`, so they are not archived.
  - The Fig. 8 key `storm-*_max_updraft_speed` holds the mean updraft speed, not the maximum.
- **Status: fixed.** The Fig. 3, 4 and 5 cells now build their keys from `storm_name`. The Fig. 4 plan-view cold pool is saved under `storm-sl_wk_t-<minutes>_pv_cp`. The Fig. 8 key is now `storm-*_mean_updraft_speed`.

### 4. Fig. 4's condensate contours are drawn on the coarse arrow grid
*Cell index 26*

- **The bug.** The condensate contour receives `this_xc_ds_quiver`, the grid thinned for the wind arrows, instead of `this_xc_ds`.
- **The effect.** The contours come from every 6th x point (3 km apart) and every 3rd level above 1.5 km (up to 750 m apart). Panels (d–i) therefore show the condensate outlines much more coarsely than the model resolves them.
- **Status: fixed.** The contour now uses `this_xc_ds`, the full cross-section grid.

---

## 2. Errors with little or no effect on the paper

### 5. `max_tracer_fraction_time` is assigned twice
*Cell index 67, the cell that computes the tracer statistics*

- **The bug.** The second assignment overwrites the first with the time of the peak of f_RS / t, not of f_RS.
- **The effect.** For the SL and SC, the stored times are wrong by up to 65 min. For example, the SC's 40 mm/hr f_RS peaks at t = 165 min, but the stored time is 100 min.
- **Why the paper is unaffected.** The IC's stored times are right by coincidence (65 min). Only the IC's times feed a later cell, the "IC max tracer fraction comparison". The paper quotes none of these times.
- **Status: fixed.** `max_tracer_fraction_time` now keeps the time of the peak of f_RS. The second calculation is stored as `max_tracer_fraction_normalized_time`, and it now uses the same expression as `max_tracer_fraction_normalized`.

### 6. The air mass uses the spacing between level centres, not the cell thickness
*`blt.calculate_thermodynamic_variables`*

- **The bug.** Δz is `z.diff("z")`, the distance between neighbouring T-levels. In the stretched layer below about 3 km, that is 4.7% smaller than the cell thickness in the RAMS header (`__zmn01`). For example, the lowest cell is 25.0 m thick, but the code uses 23.8 m. Above about 3 km, where the spacing is a constant 250 m, the two agree.
- **The effect.** E_RS comes from the lowest cell, so it is 4.7% low. η_RSE and the fractions aloft are therefore up to 5% high, and η_RP up to 5% low.
- **Status: fixed.** The air mass now uses the cell thickness from the run's header file (`blt.read_t_level_cell_thicknesses`). A header from a run with a different vertical grid raises an error.
- **Measured effect on the paper (correcting my earlier claim that no comparison changes).** The bias depends on how each storm's tracer is spread in height, so ratios between storms shift by up to 3%. Two quoted values move by one percentage point, both in lines 551–553:
  - the SL's time-averaged η_RSE goes from "36%–40%" of the SC's to 37%–40%;
  - the 1 mm/hr ratio from t = 120 min goes from "38%" to 39%.

  Every other value the paper quotes is unchanged at the precision it gives. Parts 1 and 2 quote values from before the fix, so their in-updraft ratios can differ from a rerun by up to 3%. Their fractions of emitted tracer that reach upper levels fall by 4.7%: for example, the SC's 11.6% at 1 mm/hr becomes 11.1%.

---

## 3. Stale cell

- **"IC rain production efficiency comparison" (cell index 73)** compares the IC's η_RP at t = 100 min with the other storms. The paper uses t = 70 min instead, which cell index 67 computes as `effective_rain_production_efficiency`. The t = 100 min output could be mistaken for the paper's numbers.

---

## Fixes made on 7 October

- **Code:** `src/blt_utils/__init__.py` (the air mass) and six notebook cells: Figs. 3, 4, 5, 8 and 9, and the statistics cell. I edited the notebook file on disk, because the notebook tools only work on the focused notebook.
- **Caches.** The tracer-profile and development-dataframe caches depend on the air mass, so the code now reads them from new files ending in `_cellthickness`. The old files are untouched.
  - **How I made the new files.** I rescaled the old caches instead of recomputing them from raw output, which would take hours. Every cached quantity that uses the air mass is a sum over a single model level, so multiplying each level by (cell thickness / old spacing) is exact.
  - **Check.** A from-scratch computation with the new code, at one time each for the IC and the SC, matches the rescaled caches to within 1 part in 10⁷.
  - **Provenance.** Each new file records that it was rescaled, in its `provenance` attribute.
- **Not yet done:** the notebook needs a full rerun. That regenerates the figures, the figure data, the statistics tables and the upper-level fractions with the fixes.
