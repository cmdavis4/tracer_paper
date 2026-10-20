# Final pre-submission check of the vRevisions3 manuscript (19 October)

Tick an item by changing `[ ]` to `[x]`. Section 8 needs no action.

## What was checked

- **Files:** the tracked-changes PDF `Davis et al tracer paper vRevisions3 merged_unlinked_ttcleaned.pdf` (16 October, 14:36) and the clean PDF `Davis et al tracer paper vRevisions3 merged.pdf` (14 October, 17:42). With all changes accepted, the two files have the same text, apart from Eq. 2 (item 1).
- **Line numbers** are given as "clean / tracked".
- **Text:** every sentence, read for errors, contradictions, typos and undefined terms.
- **Numbers:** every number in Sects. 3.2 and 3.3.2–3.3.6 and in the Conclusions that the figure data can test, recomputed from the figure-data pickles behind Figs. 6, 7, 9 and 10. Earlier checks covered Sects. 2.4, 3.3.3, 3.3.4 and 3.4.
- **Cross-references:** every Figure, Table, Section and Equation reference, including panel letters and the times printed on the panels.
- **Citations:** all 104 reference-list entries against the in-text citations, in both directions.
- **Journal rules:** ACP's [submission guidelines](https://www.atmospheric-chemistry-and-physics.net/submission.html) and [AI policy](https://www.atmospheric-chemistry-and-physics.net/policies/ai_policy.html), as read on 19 October 2026.
- **Not checked, as agreed:** numbers quoted from other papers, the content of the equations and of the Table 3–4 definitions, and whether the tracked file marks every change since the original submission.

---

## 1. Error in the tracked-changes file only

- [x] **Eq. 2 has no replacement (tracked 246–247).** The tracked file shows the old Eq. 2, which uses Δt, as struck through. The new Eq. 2, which uses (2 s), is missing, but line 239 still cites "Equation 2". The clean file has the new equation. Re-insert it in the marked-up document as a tracked insertion, then export again. This item is carried over from `paper_check_vRevisions3_tracked_16Oct.md`.

---

## 2. Text errors

- [x] **Conclusion 1 refers to a maximum it no longer states (633–634 / 719–720).**
  - The paragraph says "This maximum mixing ratio of rain-sourced tracers is short-lived in the isolated convective storm".
  - The revision rewrote the sentence before it, which used to describe the isolated storm's brief maximum. It now says only that the isolated storm "entrains less air from rainy surface regions".
  - "This maximum" therefore has nothing to refer to. Name it, for example "The isolated convective storm's maximum rain-sourced tracer mixing ratio is short-lived because…".
- [x] **Sect. 3.3.2 describes an updraft mean as a gridpoint value (407 / 425).** The section opens: "The rain-sourced tracer mixing ratio (q̄_RS) conveys information about the product of the mass of air in that gridpoint…". But q̄_RS is the mean over the updraft, and no gridpoint has been named. The sentence repeats line 370, which is about one gridpoint. Replace "in that gridpoint" with "in the updraft".
- [x] **Code and data availability repeats itself (707–711 / 799–803).** The section repeats two things:
  - The same Zenodo DOI appears in two consecutive sentences.
  - The section says twice that the RAMS output is excluded because of its file size.

  Merging the old "Interactive computing environment" paragraph into this section caused the repeats. Delete one DOI and one of the two file-size sentences.
- [x] **The RAMS version and the RAMS reference disagree (113 and 823–824 / 115 and 907–908).**
  - The text and Table 1 say RAMS 6.3.04, which matches the archived source in `replication/RAMS_tracer_source`.
  - The RAMS reference that the text cites, van den Heever et al. (2022), is the Zenodo record for v6.3.02.

  Cite a 6.3.04 record if one exists. Otherwise, add Davis and van den Heever (2025) as the source of the modified 6.3.04 code.
- [ ] **Two symbols for vertical velocity (266 / 279).** Line 266 says "we will use the symbols U, V, and W" for the wind components. But Table 2, Eq. 2, Table 3 and Sect. 3.3 use lowercase w, and the text never uses W. Only the figure legends do. The simplest fix is to say at line 266 that W in the figures is the w of Table 2.

---

## 3. Typos

| | Clean / tracked | Now | Should be |
|---|---|---|---|
| [ ] | 342 / 359 | "the median rain-sourced concentrations of INPs … is" | "are" |
| [ ] | 408–409 / 427 | "for the 1 mm hr-1 in the supercell" | "for the 1 mm hr-1 tracer in the supercell" |
| [ ] | 461 / 501–502 | "the storm’s different initial rain onset times" | "the storms’" |
| [ ] | 605 / 679 | "supercell‘s", with an opening quotation mark | "supercell’s" |
| [ ] | Table 1 | "Rayleigh dampening layer" | "Rayleigh damping layer" |
| [ ] | 270 / 283 | "(e.g. Byers and Braham 1949; Wilhelmson 1974; Schlesinger 1978)" | A comma before each year, as in every other citation |

---

## 4. Reference list

Every in-text citation has an entry, and every entry is cited. ACP asks for the volume, the full page range and the DOI of each entry.

| | Clean / tracked | Entry | Problem |
|---|---|---|---|
| [ ] | 856 / 936 | Kaimal et al. (1976) | No DOI. It is https://doi.org/10.1175/1520-0469(1976)033%3C2152:TSITCB%3E2.0.CO;2 |
| [ ] | 890 / 968 | Maddox (1980) | No DOI. It is https://doi.org/10.1175/1520-0477(1980)061%3C1374:MCC%3E2.0.CO;2 |
| [ ] | 1008 / 1068 | Wakimoto (1985) | No DOI. It is https://doi.org/10.1175/1520-0493(1985)113%3C1131:FDMAOT%3E2.0.CO;2 |
| [ ] | 914 / 987 | Morris et al. (2014) | One author is written "Alex Huffman, J.". It should be "Huffman, J. A." |
| [ ] | 784 / 874 | Diehl et al. (2002) | The title has a double colon: "pollen:: Part II" |
| [ ] | 781, 806, 998 / 871, 893, 1057 | Després et al. (2012), Garcia et al. (2012), Tobo et al. (2013) | No page range or article number |
| [ ] | 885–886 / 963 | Lilly (1982) | A book chapter with no editors or publisher |

---

## 5. ACP rules

- [ ] **Undefined abbreviations.** ACP says abbreviations "need to be defined in the abstract and then again at the first instance in the rest of the text."
  - "RAMS" is not defined in the abstract (line 14). The abstract is exactly 250 words, which is ACP's limit. Spelling out RAMS therefore means cutting four words elsewhere in the abstract.
  - "INP" is defined only in the abstract. Its first use in the main text is the heading of Sect. 3.2 (322 / 339).
  - "CIN" and "WK82" are never defined (140 / 145).
- [ ] **Units in the figures (optional before review).** ACP asks figures to use "h" rather than "hr", and exponents rather than slashes, for example "m s⁻¹".
  - Every figure legend and colour bar uses "mm/hr", "m/s" or "#/kg".
  - The copy editors fix units in the text, but changing the figures needs a rerun of the notebook.
  - The typesetters may ask for this change after acceptance.
- **No action needed, because the copy editors handle these:**
  - the forms "Fig.", "Sect." and "Eq. (1)" in running text;
  - "hr" in the text;
  - the missing spaces between numbers and units ("25m", "~15km", "z=10km");
  - journal-name abbreviations and all-caps titles in the reference list;
  - colour and line-style words in the captions;
  - the unnumbered F_t-RM equation in Sect. 2.4.

---

## 6. Decide: AI-use statement

- [x] **The manuscript has no AI-use statement.** ACP's rules for one are as follows:
  - Authors must declare any AI tool used beyond grammar and readability checks, in the Methods or the Acknowledgements.
  - The declaration names the tool and describes "the purpose and extent of this use".
  - Authors certify this declaration at manuscript registration.
  - The policy does not say whether AI-assisted analysis code counts.

  Claude Code was used to edit parts of the analysis notebook and to run these manuscript checks. If you decide a statement is needed, one sentence in the Acknowledgements is enough. For example: "Claude Code (Anthropic) was used to help edit the analysis code and to check the manuscript against the analysis output; the authors reviewed all changes."

---

## 7. Still open in `paper_check_vRevisions3_checklist.md`, section H

- The Zenodo reference still names version "v1.0-acp" (775 / 865).
- The text cites DOI …15659853, but the reference list cites …15659854 (775 / 866).
- **If the new archive version is built from `replication/`, refresh that folder first.**
  - The folder matches the layout that the README describes.
  - But its notebook dates from 18 March, and its figure data date from 2025.
  - The figure-data files also carry the old figure numbers, so `f06_tracer_entrainment.pkl` holds today's Fig. 7.

---

## 8. What matched

- **Cross-references:** every one points to an item that exists. Each panel letter and each time agrees with the figure, for example Fig. 3 at t = 30, 65 and 85 min, and Fig. 5 at t = 90, 140 and 170 min.
- **Figures:** all 11 agree with their captions. All are identical to the notebook PNGs, which the 16 October check established.
- **Required sections:** all are present and in ACP's order. Page and line numbers run consecutively in both PDFs.
- **Numbers:**

| Claim | Paper | Figure data |
|---|---|---|
| Sect. 3.3.2: peak q̄_RS at 1 mm/hr, SC and SL relative to IC | 3.2x, 3.9x | 3.24, 3.87 |
| Same, at 20 mm/hr | ~5x in both | 4.83 (SC), 5.18 (SL) |
| Peak q̄_RS at 40 mm/hr, SL relative to SC | ~5.4 | 5.36 |
| Peak times of q̄_RS: IC; SL and SC | t = 65; t = 150–180 | t = 65; t = 150–180 |
| Sect. 3.3.4: peak f_RS at 1–20 mm/hr, IC relative to SL; to SC | ~4x; ~2x | 3.6–4.2; 1.7–2.1 |
| Same, at 40 mm/hr | ~1.3x; ~4x | 1.34; 3.80 |
| SL's f_RS relative to SC's, 1–20 mm/hr (peak and time mean) | 30–60% less | 40–58% less (peak), 39–44% less (mean) |
| Sect. 3.3.5: η_RP, IC at t = 70 relative to SL's and SC's end values, 1–20 mm/hr | ~a quarter; ~half | 0.25–0.28; 0.48–0.54 |
| Same, at 40 mm/hr | 41%; 66% | 41.4%; 65.5% |
| SL's η_RP relative to SC's (time mean, peak, end), all thresholds | SL larger | 1.6–2.1 |
| Sect. 3.3.6: η_RSE, SL's time mean relative to SC's, 1–20 mm/hr | 37–40% | 36.6–40.4% |
| Same, from t = 120 min | 39% rising to 47%; ~3x at 40 mm/hr | 38.6% rising to 47.4%; 3.3x |
| IC's surface θ, 40 mm/hr regions relative to 1 mm/hr regions, t = 65 | ~3.3 K colder | 3.30 K |
| Sect. 3.2: maximum −10 °C INP concentration | 0.45–0.48 L⁻¹ | 0.450–0.478 L⁻¹ |
| Conclusion 2: SL's time-averaged q̄_RS relative to SC's, 1–20 mm/hr | comparable | 0.77–0.93 |
| Sect. 3.4 and Conclusions: SC's peak and end q̄_FS and q̄_RS below SL's | all below | all below |
