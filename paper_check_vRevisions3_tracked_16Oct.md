# Check of "Davis et al tracer paper vRevisions3 merged_unlinked_ttcleaned.pdf" (16 October, 14:36 export)

Tick an item by changing `[ ]` to `[x]`.

## What was compared

- **Paper:** the tracked-changes PDF, 59 pages, saved 16 October at 14:36. Line numbers below follow it.
- **Reference:** the clean export `Davis et al tracer paper vRevisions3 merged.pdf` (14 October, 17:42), which the previous check (`paper_check_vRevisions3_merged_14Oct_1602.md` and its follow-up) found free of errors.
- **Text:** I rebuilt the accepted-changes text by dropping every struck-through character, then diffed it sentence by sentence against the 17:42 clean text.
- **Figures:** I compared every embedded figure with the 17:42 clean PDF and with the notebook PNGs in `analysis/paper_figures/`.

## 1. Error

- [ ] **Eq. 2 (lines 246–247) is deleted with no replacement.** The tracked PDF shows the old Eq. 2, which uses Δt, struck through in red. It does not show the new Eq. 2, which uses (2 s). Once the changes are accepted, the paper has no Eq. 2, but line 239 still cites "Equation 2". The clean 17:42 PDF has the new equation, so only the tracked-changes file is affected. Word's compare often drops an edited equation object. Re-inserting the new equation as a tracked insertion fixes it.

## 2. Still open in `paper_check_vRevisions3_checklist.md`

The Zenodo items in its section H are unchanged:

- The reference (lines 865–866) still names version "v1.0-acp".
- The text (lines 799–800) cites DOI …15659853, and the reference list cites …15659854.

## 3. What matched

- **Text:** apart from Eq. 2, the accepted-changes text matches the 17:42 clean text word for word. The remaining diffs are overbar glyphs, spacing and text that Word marks as moved (green). The moved text is the "Interactive computing environment" paragraph (line 810), which now appears in Code and data availability.
- **Figures:** all 11 figures are identical to the 17:42 clean PDF. The file also contains the superseded versions of Figs. 3, 4 and 6–11, each on the page before its caption, as expected in a tracked-changes file.
