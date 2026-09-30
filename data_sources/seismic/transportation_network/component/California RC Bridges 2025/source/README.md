# Source data

`Sa_1_Fragility Database.csv` is a copy of the file of the same name in
DesignSafe project PRJ-5910, version 1 (published 2025-04-29,
doi:10.17603/ds2-c73m-nj37, Open Data Commons Attribution License):

Chen, S., Y. Xie, C. Wu, H. V. Burton, J. E. Padgett, and Á. Zsarnóczay. 2025.
Second-Generation Component and System-Level Seismic Fragility Models for
Reinforced Concrete Bridges in California. DesignSafe-CI, PRJ-5910.

The data paper describing it is Chen et al. (2025), Earthquake Spectra 41(4):
3234–3253, doi:10.1177/87552930251343634.

The same project version also publishes `EDP_Fragility Database.csv`, which
lists the median capacities of each component for its damage states, in terms of
the engineering demand parameter (EDP) of the component. That file is kept in
`../../California RC Bridges 2025 EDP/source/`. The generator in
`../../California RC Bridges 2025 EDP/` builds the EDP-based dataset
`seismic/transportation_network/component/California RC Bridges 2025 EDP` from
it. The generator of this folder, `../generate_library_files.py`, reads it only
to confirm the capacities quoted in the damage-state descriptions (see
[Damage-state descriptions](#damage-state-descriptions)); it changes no
fragility parameter here.

This file is the record of every change the generator makes to the published
labels and values and of every published row it omits. The same changes are
recorded in the `correction` column of `../id_crosswalk.csv` and in the Comments
of each affected model.

## File format

`Sa_1_Fragility Database.csv`:

- UTF-8 with a byte-order mark, 1,018 data rows; CRLF line endings in the
  DesignSafe file, stored here with LF endings.
- Columns: `Bridge group`, `Fragility notation`, the lognormal median (`_λ`, in
  g) and dispersion (`_ζ`) of Sa(1.0 s) for the `Slight`, `Moderate`,
  `Extensive`, and `Complete` damage states, `Notes`, and `Reference`.
- `N/A` marks a damage state with negligible probability of damage from ground
  shaking; an empty cell marks a state the source does not define.

Row numbers used in the generator, the crosswalk, the model metadata, and this
file are 0-based data rows of `Sa_1_Fragility Database.csv`; the spreadsheet row
is the data row plus 2.

Keep the file unchanged. Corrections are applied by
`../generate_library_files.py`.

## Number formatting

The source values have at most two decimals and no trailing zeros (3,968 cells
with two decimals, 489 with one, and 47 with none); the generator stops if a
cell has more. Every value read from the source is written to `fragility.csv`
exactly as published, with only the float noise of the conversion removed. The
recomputed values of data rows 0 to 7 (see [Recomputed single-span
diaphragm-abutment curves](#recomputed-single-span-diaphragm-abutment-curves))
are rounded to four decimals, and trailing zeros are dropped, so 32.5740 is
written 32.574. The Comments of those models quote each published value as
printed and each recomputed value as stored.

## Changes to the published labels and values

### Column-shape labels

Three rows carry the wrong column shape in their `Notes`. The values are
unchanged; only the label, and with it the shape token of the model ID, is
corrected.

| Row | Published shape | Stored shape |
|---|---|---|
| 298 | circular | oblong |
| 299 | oblong | circular |
| 741 | circular | rectangular |

The corrected models are `CRCB.E2.S3P.C1.D.AbPass.BoxGirder.Obl.M17` (row 298),
`CRCB.E2.S3P.C1.D.AbPass.BoxGirder.Span5.MF.Circ.MCP18` (row 299), and
`CRCB.E1.S3P.C1.S.System.BoxGirder.Rect.M17` (row 741).

Evidence:

- Row 298: the published values match the oblong-column class S-E2-S34-O-D of
  the Mangalathu (2017) thesis.
- Row 299: Mangalathu et al. (2018) model circular columns only (Fig. 2c, Fig.
  3, and Table 1 of the study). The source exchanges the shape labels of rows
  298 and 299; the values are in the correct rows.
- Row 741: the published values match the rectangular-column class S-E1-S34-R-S
  of the thesis and the omitted rectangular-column Soleimani (2017) entry of the
  same group (data row 744).

### Abutment directions of the single-span seat-abutment curves

The four Cavalcante et al. (2022) active and passive abutment curves of group
E123-S1-C0-S carry the values that Cavalcante et al. (2022, Table 9) give for
the other direction. The notation is exchanged; the values are unchanged.

| Data row | Deck | Published notation | Stored notation | Model ID |
|---|---|---|---|---|
| 26 | slab | AbAct | AbPass | `CRCB.E123.S1.C0.S.AbPass.Slab.CPR22` |
| 27 | T-beam | AbAct | AbPass | `CRCB.E123.S1.C0.S.AbPass.TBeam.CPR22` |
| 28 | slab | AbPass | AbAct | `CRCB.E123.S1.C0.S.AbAct.Slab.CPR22` |
| 29 | T-beam | AbPass | AbAct | `CRCB.E123.S1.C0.S.AbAct.TBeam.CPR22` |

The published values of these rows, as median in g and dispersion for the slight
and the moderate state:

| Row | Slight λ | Slight ζ | Moderate λ | Moderate ζ |
|---|---|---|---|---|
| 26 | 24.8 | 0.95 | 104 | 0.95 |
| 27 | 27.7 | 0.78 | 117 | 0.78 |
| 28 | 10.9 | 0.95 | 35.6 | 0.95 |
| 29 | 12.9 | 0.77 | 43 | 0.77 |

In the tables of this file, λ is the median in g and ζ the dispersion, as in the
column names of the source file.

### Recomputed single-span diaphragm-abutment curves

In the literature they surveyed, Chen et al. (2025) found no viable set of
fragility models for single-span bridges with diaphragm abutments. They derived
the active and passive abutment curves of group E123-S1-C0-D (data rows 0 to 7)
from the seat-abutment curves of E123-S1-C0-S with modification factors. A
modification factor is the ratio of a diaphragm-abutment to the corresponding
seat-abutment fragility parameter in the two-span, single-column groups of one
era: Era 1, or Eras 2 and 3, as the `Notes` of each row state. The published
values show the factors applied to the medians and the dispersions; the paper
describes the adjustment of the medians only. The database does not list the
factors.

The factors were applied by the labels of the seat rows. The slight-median
factors of Eras 2 and 3 can be computed from the source file. The Era 2 and Era
3 two-span, single-column groups publish identical AbAct and AbPass curves, so
the Era 2 rows serve for both: E2-S2-C1-D (diaphragm abutments) and E2-S2-C1-S
(seat abutments), each with a circular-column and an oblong-column curve. F is
the mean of the two diaphragm-to-seat ratios:

AbAct:  F = mean(0.50 / 0.73, 0.42 / 0.66) = 0.6606   (data rows 114, 115 over
536, 537) AbPass: F = mean(1.05 / 1.31, 0.94 / 1.24) = 0.7798   (data rows 116,
117 over 538, 539)

Applied to the seat rows as labeled, these factors reproduce the published
slight medians of rows 1 and 5: 0.6606 × 24.8 = 16.38 g (seat row 26, labeled
AbAct) and 0.7798 × 10.9 = 8.50 g (seat row 28, labeled AbPass). Applied to the
seat rows that hold the direction of each row, they give 0.6606 × 10.9 = 7.20 g
and 0.7798 × 24.8 = 19.34 g. The AbAct diaphragm curves were therefore built on
the seat rows labeled AbAct (26 and 27), which hold the passive-direction
values, and the AbPass curves on rows 28 and 29, which hold the active-direction
values, so the swap corrected above propagates into rows 0 to 7.

Each published value is F × s_labeled, where F is the factor for the direction
and parameter of the row and s_labeled is the value of the seat row with the
same deck and the same published label. Keeping F and using the seat row that
holds the direction of the row, s_correct, gives

recomputed = F × s_correct = published × s_correct / s_labeled

so the factor itself need not be known. Every populated value is recomputed this
way and rounded to four decimals. The ratios s_correct / s_labeled are listed
below; the seat rows are given as the labeled row, then the correct row:

| Rows | Curve | Seat rows | Slight λ | Slight ζ | Moderate λ | Moderate ζ |
|---|---|---|---|---|---|---|
| 0, 1 | active, slab | 26, 28 | 0.4395 | 1.0000 | 0.3423 | 1.0000 |
| 2, 3 | active, T-beam | 27, 29 | 0.4657 | 0.9872 | 0.3675 | 0.9872 |
| 4, 5 | passive, slab | 28, 26 | 2.2752 | 1.0000 | 2.9213 | 1.0000 |
| 6, 7 | passive, T-beam | 29, 27 | 2.1473 | 1.0130 | 2.7209 | 1.0130 |

The ratios are shown to four decimals; the generator uses them unrounded. The
eight recomputed models are:

| Row | Model ID |
|---|---|
| 0 | `CRCB.E123.S1.C0.D.AbAct.Slab.Mod1.CPR22` |
| 1 | `CRCB.E123.S1.C0.D.AbAct.Slab.Mod23.CPR22` |
| 2 | `CRCB.E123.S1.C0.D.AbAct.TBeam.Mod1.CPR22` |
| 3 | `CRCB.E123.S1.C0.D.AbAct.TBeam.Mod23.CPR22` |
| 4 | `CRCB.E123.S1.C0.D.AbPass.Slab.Mod1.CPR22` |
| 5 | `CRCB.E123.S1.C0.D.AbPass.Slab.Mod23.CPR22` |
| 6 | `CRCB.E123.S1.C0.D.AbPass.TBeam.Mod1.CPR22` |
| 7 | `CRCB.E123.S1.C0.D.AbPass.TBeam.Mod23.CPR22` |

Their published and stored values (each cell gives the published value, then the
stored value):

| Row | Slight λ | Slight ζ | Moderate λ | Moderate ζ |
|---|---|---|---|---|
| 0 | 17.56 → 7.7179 | 0.95 → 0.95 | 78.73 → 26.9499 | 0.95 → 0.95 |
| 1 | 16.38 → 7.1993 | 1.17 → 1.17 | 95.16 → 32.574 | 1.17 → 1.17 |
| 2 | 19.62 → 9.1371 | 0.78 → 0.77 | 88.57 → 32.5514 | 0.78 → 0.77 |
| 3 | 18.3 → 8.5224 | 0.96 → 0.9477 | 107.06 → 39.3468 | 0.96 → 0.9477 |
| 4 | 7.75 → 17.633 | 1 → 1 | 26.3 → 76.8315 | 1 → 1 |
| 5 | 8.5 → 19.3394 | 1.19 → 1.19 | 37.22 → 108.7326 | 1.19 → 1.19 |
| 6 | 9.17 → 19.6906 | 0.81 → 0.8205 | 31.77 → 86.444 | 0.81 → 0.8205 |
| 7 | 10.06 → 21.6017 | 0.96 → 0.9725 | 44.96 → 122.333 | 0.96 → 0.9725 |

The extensive and complete states of these rows are not defined. The generator
repeats the calculation of F above from the source rows and stops unless F ×
s_labeled equals the published slight median of rows 1 and 5, and F × s_correct
equals the recomputed one, each within 0.01 g.

### Moderate dispersion of the Shao et al. (2022) 10 ft slope-offset curves

Ten rows repeat one Shao et al. (2022) configuration, an abutment on piles with
a 10 ft approach slab and damage through slope offset, in different groups. The
source prints a large dispersion of 1.99, as does Shao et al. (2022, Table 6);
the dataset stores 0.99. Shao et al. (2022, Sec. 4.4) defines the dispersion of
the integrated model as the square root of the sum of squares of the dispersions
of its three methods, and for this configuration their Tables 3 to 5 give 0.41,
0.62, and 0.65, which combine to 0.987. The moderate median, 0.92 g, and the
slight state are unchanged.

| Data row | Group | Model ID |
|---|---|---|
| 17 | E123-S1-C0-D | `CRCB.E123.S1.C0.D.Approach.Pile.Slope10.SXR22` |
| 41 | E123-S1-C0-S | `CRCB.E123.S1.C0.S.Approach.Pile.Slope10.SXR22` |
| 63 | E1-S2-C1-D | `CRCB.E1.S2.C1.D.Approach.Pile.Slope10.SXR22` |
| 103 | E1-S2-C2P-D | `CRCB.E1.S2.C2P.D.Approach.Pile.Slope10.SXR22` |
| 247 | E1-S3P-C1-D | `CRCB.E1.S3P.C1.D.Approach.Pile.Slope10.SXR22` |
| 286 | E1-S3P-C2P-D | `CRCB.E1.S3P.C2P.D.Approach.Pile.Slope10.SXR22` |
| 466 | E1-S2-C1-S | `CRCB.E1.S2.C1.S.Approach.Pile.Slope10.SXR22` |
| 522 | E1-S2-C2P-S | `CRCB.E1.S2.C2P.S.Approach.Pile.Slope10.SXR22` |
| 739 | E1-S3P-C1-S | `CRCB.E1.S3P.C1.S.Approach.Pile.Slope10.SXR22` |
| 795 | E1-S3P-C2P-S | `CRCB.E1.S3P.C2P.S.Approach.Pile.Slope10.SXR22` |

### Filled lower states of nine Unseating curves

Nine Unseating rows define the extensive and complete states only: the source
studies define unseating capacity for those two states, and the slight and
moderate cells are empty. The generator sets the slight and moderate median and
dispersion equal to the extensive ones. The extensive and complete curves are
unchanged and keep the numbers DS3 and DS4. Note that with this setup, the
slight and moderate states cannot occur which is in line with the intent of
model developers. The LS1 and LS2 descriptions of these models say that the
state is not defined by the source.

The nine models are:

| Row | Model ID |
|---|---|
| 24 | `CRCB.E123.S1.C0.S.Unseating.Slab.CPR22` |
| 25 | `CRCB.E123.S1.C0.S.Unseating.TBeam.CPR22` |
| 249 | `CRCB.E1.S3P.C1.D.Unseating.BoxGirder.Span5.MF.Disc.Circ.MCP18` |
| 316 | `CRCB.E2.S3P.C1.D.Unseating.BoxGirder.Span5.MF.Disc.Circ.MCP18` |
| 400 | `CRCB.E3.S3P.C1.D.Unseating.BoxGirder.Span5.MF.Disc.Circ.MCP18` |
| 851 | `CRCB.E2.S3P.C2P.S.Unseating.BoxGirder.TCB.Span3.Obl.JML19` |
| 852 | `CRCB.E2.S3P.C2P.S.Unseating.BoxGirder.TCB.Span3.Flr1.JML19` |
| 853 | `CRCB.E2.S3P.C2P.S.Unseating.BoxGirder.TCB.Span3.Flr1T.JML19` |
| 854 | `CRCB.E2.S3P.C2P.S.Unseating.BoxGirder.TCB.Span3.Flr2.JML19` |

The extensive median and dispersion copied to the slight and moderate states:

| Row | Extensive λ | Extensive ζ |
|---|---|---|
| 24 | 0.94 | 0.73 |
| 25 | 0.8 | 0.64 |
| 249 | 3.87 | 1.12 |
| 316 | 9.8 | 0.99 |
| 400 | 11.72 | 1.04 |
| 851 | 7.79 | 0.9 |
| 852 | 7.66 | 0.9 |
| 853 | 7.18 | 0.87 |
| 854 | 7.28 | 0.85 |

## Omitted rows

The generator omits 101 of the 1,018 rows; the crosswalk lists them with
`omitted` in its `dataset` column.

### Soleimani (2017) entries that repeat Mangalathu (2017) curves

In three single-column, multi-span seat-abutment groups the source lists six
Mangalathu (2017) System curves a second time, attributed to Soleimani (2017),
with the same column shape. All eight parameters of each repeated row equal
those of its Mangalathu (2017) partner, which the generator checks. The repeated
rows are omitted, and the Mangalathu (2017) models are kept.

| Omitted | Partner | Shape | Kept model ID |
|---|---|---|---|
| 742 | 740 | circular | `CRCB.E1.S3P.C1.S.System.BoxGirder.Circ.M17` |
| 744 | 741 | rectangular | `CRCB.E1.S3P.C1.S.System.BoxGirder.Rect.M17` |
| 835 | 833 | circular | `CRCB.E2.S3P.C1.S.System.BoxGirder.Circ.M17` |
| 837 | 834 | oblong | `CRCB.E2.S3P.C1.S.System.BoxGirder.Obl.M17` |
| 957 | 955 | circular | `CRCB.E3.S3P.C1.S.System.BoxGirder.Circ.M17` |
| 959 | 956 | oblong | `CRCB.E3.S3P.C1.S.System.BoxGirder.Obl.M17` |

The shape of row 741 is its corrected label; the source labels it circular.

### Rows without parameters

Ninety-five component rows are N/A in the slight and moderate median and
dispersion, the only states the source defines for these components (the
extensive and complete cells are empty). The database uses N/A to mark a
negligible probability of damage from ground shaking. They hold no curve and are
omitted; no System row is among them. They are 71 column foundation rotation
(FndRot) rows, in every group where the component occurs except for one row of
E1-S2-C2P-S (data row 504, kept as
`CRCB.E1.S2.C2P.S.FndRot.BoxGirder.MCB.Rect.M17`), and 24 shear key (Key) rows,
in the Era 2 and Era 3 multi-span seat-abutment groups.

| Component | Group | Rows | Data rows |
|---|---|---|---|
| FndRot | E1-S2-C1-D | 2 | 48, 49 |
| FndRot | E1-S2-C1-S | 2 | 453, 454 |
| FndRot | E1-S2-C2P-D | 4 | 90, 91, 92, 93 |
| FndRot | E1-S2-C2P-S | 3 | 501, 502, 503 |
| FndRot | E1-S3P-C1-D | 2 | 238, 239 |
| FndRot | E1-S3P-C1-S | 2 | 726, 727 |
| FndRot | E1-S3P-C2P-D | 4 | 273, 274, 275, 276 |
| FndRot | E1-S3P-C2P-S | 4 | 774, 775, 776, 777 |
| FndRot | E2-S2-C1-D | 2 | 122, 123 |
| FndRot | E2-S2-C1-S | 2 | 546, 547 |
| FndRot | E2-S2-C2P-D | 4 | 153, 154, 155, 156 |
| FndRot | E2-S2-C2P-S | 4 | 593, 594, 595, 596 |
| FndRot | E2-S3P-C1-D | 2 | 305, 306 |
| FndRot | E2-S3P-C1-S | 2 | 816, 817 |
| FndRot | E2-S3P-C2P-D | 4 | 356, 357, 358, 359 |
| FndRot | E2-S3P-C2P-S | 4 | 891, 892, 893, 894 |
| FndRot | E3-S2-C1-D | 2 | 181, 182 |
| FndRot | E3-S2-C1-S | 2 | 636, 637 |
| FndRot | E3-S2-C2P-D | 4 | 209, 210, 211, 212 |
| FndRot | E3-S2-C2P-S | 4 | 683, 684, 685, 686 |
| FndRot | E3-S3P-C1-D | 2 | 392, 393 |
| FndRot | E3-S3P-C1-S | 2 | 938, 939 |
| FndRot | E3-S3P-C2P-D | 4 | 424, 425, 426, 427 |
| FndRot | E3-S3P-C2P-S | 4 | 989, 990, 991, 992 |
| Key | E2-S2-C1-S | 2 | 550, 551 |
| Key | E2-S2-C2P-S | 4 | 601, 602, 603, 604 |
| Key | E2-S3P-C1-S | 2 | 820, 821 |
| Key | E2-S3P-C2P-S | 4 | 899, 900, 901, 902 |
| Key | E3-S2-C1-S | 2 | 640, 641 |
| Key | E3-S2-C2P-S | 4 | 691, 692, 693, 694 |
| Key | E3-S3P-C1-S | 2 | 942, 943 |
| Key | E3-S3P-C2P-S | 4 | 997, 998, 999, 1000 |

## Soleimani (2017) models

Six Soleimani (2017) System curves remain after the repeated entries are
omitted. Each is the regular base model of the unbalanced-frame study of that
thesis (Tables 7.10 and I.5).

| Data row | Model ID |
|---|---|
| 743 | `CRCB.E1.S3P.C1.S.System.BoxGirder.Circ.S17` |
| 745 | `CRCB.E1.S3P.C1.S.System.BoxGirder.Rect.S17` |
| 836 | `CRCB.E2.S3P.C1.S.System.BoxGirder.Circ.S17` |
| 838 | `CRCB.E2.S3P.C1.S.System.BoxGirder.Obl.S17` |
| 958 | `CRCB.E3.S3P.C1.S.System.BoxGirder.Circ.S17` |
| 960 | `CRCB.E3.S3P.C1.S.System.BoxGirder.Obl.S17` |

The author states in the Acknowledgements of the thesis that its curves "are
illustrative and should not be used for deployment in ShakeCast, or other risk
analysis software". Chen et al. (2025) included the curves in the database. This
library distributes models and is not a risk-analysis application, but its
models are meant for damage and loss assessments such as pelicun's, which is the
kind of use the author cautions against. Users must judge for themselves whether
these curves suit their use before deploying them. The Comments of each of the
six models carry the same caution.

## Damage-state descriptions

The LimitStates descriptions of the models are written for this library from the
wording in `../metadata_text.json` (`limit_states`). Their sources are:

- System: the Hazus verbal definitions of the four bridge damage states.
- Column, Unseating, and the secondary components (abutments, bearings, deck
  displacement, column foundations, shear keys, and joint seals): Mangalathu
  (2017) Sec. 6.1 and Table 6.13, Ramanathan (2012) Chapter 5, and Chen et al.
  (2025) Sec. 3. The two-state unseating capacities of 6 and 9 in. come from the
  EDP-file row for groups E1-S2-C2P-S, E1-S3P-C2P-S, E123-S1-C0-D, and
  E123-S1-C0-S, which cites Mangalathu, Soleimani, and Jeon (2017) and
  Cavalcante et al. (2022).
- Approach: Shao et al. (2022) Table 2.
- Cavalcante et al. (2022) bearings: Cavalcante et al. (2022) Table 8.

The median capacities that the descriptions quote come from `EDP_Fragility
Database.csv` in `../../California RC Bridges 2025 EDP/source/`, where they are
expressed in the EDP of each component (curvature ductility, column drift in
percent, displacement or settlement in inches, or rotation in radians). The
generator keeps them in its `CAPACITIES` table and stops unless every quoted
value equals the value of the matching row of the EDP file; its function
`edp_row_key` documents how rows are matched by component, EDP, and the era or
group in `Notes`. Some values of the EDP file are deliberately not quoted:

- The moderate joint seal description quotes no capacity. The EDP file gives 5
  in., which the published seal curves do not reproduce (see the seal sentence
  in the model Comments).
- The Era 3 row of unseating capacities is not quoted for the Era 3 Mangalathu
  (2017) curves. Their descriptions quote the Era 2 set, because those curves
  equal the Era 2 curves of the thesis (Appendix C).
- The two-state unseating models (the filled rows above) quote only the
  extensive and complete capacities of their row.
- No description quotes the `Dispersion` column.

The capacities are text only; no fragility parameter is derived from or changed
by the EDP file.
