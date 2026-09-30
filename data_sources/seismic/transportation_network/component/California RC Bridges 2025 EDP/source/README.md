# Source data

`EDP_Fragility Database.csv` is a copy of the file of the same name in
DesignSafe project PRJ-5910, version 1 (published 2025-04-29,
doi:10.17603/ds2-c73m-nj37, Open Data Commons Attribution License):

Chen, S., Y. Xie, C. Wu, H. V. Burton, J. E. Padgett, and Á. Zsarnóczay. 2025.
Second-Generation Component and System-Level Seismic Fragility Models for
Reinforced Concrete Bridges in California. DesignSafe-CI, PRJ-5910.

The data paper describing it is Chen et al. (2025), Earthquake Spectra 41(4):
3234–3253, doi:10.1177/87552930251343634.

The file lists the median capacities of each bridge component for its damage
states, in terms of the engineering demand parameter (EDP) of the component.
These are the median component capacities that the database associates with its
Sa(1.0 s) curves; for the joint seal and Era 3 unseating values, which the
published curves do not reflect, see [Values stored as
published](#values-stored-as-published).
`../generate_library_files.py` turns each row of `../models.csv` (below) into a
lognormal fragility model in the EDP of the component and writes the dataset
`seismic/transportation_network/component/California RC Bridges 2025 EDP`.

This is the only copy of the file in the repository. The generator of the
Sa(1.0 s)-based datasets in `../../California RC Bridges 2025/` also reads it,
to confirm the capacities quoted in its damage-state descriptions.

This README is the record of every change made to the published rows: the
split of row 6 in `../models.csv` and the changes the generator makes to the
published values. The value changes are also recorded in the `change` column
of `../id_crosswalk.csv` and in the Comments of each affected model.

## File format

- ASCII text without a byte-order mark, 24 data rows; CRLF line endings in the
  DesignSafe file, stored here with LF endings.
- Columns: `Fragility notation`, `EDP (unit)`, the median capacity of each
  damage state (`Slight_median` to `Complete_median`, empty where a component
  has fewer states), a `Dispersion`, `Notes` that name the design era, the
  bridge groups, or the approach configuration a row applies to (empty in the
  nine rows that apply to every group), and `References` that name the source
  studies.

Row numbers in this file are 0-based data rows of `EDP_Fragility
Database.csv`; the spreadsheet row is the data row plus 2. The model metadata
give the data row of this file; the crosswalk gives it and the 0-based row of
`../models.csv`.

Keep the file unchanged. Changes are applied in `../models.csv` and by
`../generate_library_files.py`, which stops if `../models.csv` no longer
matches this file.

## The model table

`../models.csv` is the table the generator reads. It differs from
`EDP_Fragility Database.csv` in three ways: row 6 appears twice, so that each
row gives one model; the `Notes` and `References` of the two copies name only
the two bridge groups and the study of their model, as described under [Row
6](#row-6-split-into-two-models-with-filled-lower-states); and a first column,
`Model ID`, is added. Every other value, including whitespace and spelling, is
as in the published file. Rows 0 to 6 of `../models.csv` are data rows 0 to 6
of the published file, row 7 is the second copy of data row 6, and rows 8 to 24
are data rows 7 to 23.

A model ID has the form `CRCB.<component>[.<qualifier>].<scope>.<source>`. The
component is the `Fragility notation` of the row; the qualifier, used only for
columns, is `CD` for curvature ductility or `DR` for drift ratio. The scope is
a design era (`E1`, `E2`, `E3`), `SingleSpan`, `MultiSpan` for the capacities
the file assigns to all six multi-span studies, or an approach configuration
(`Step`, `Slope10`, `Slope30`, `Slope45`). The source token names the study the
capacities come from, and it gives the model its `Reference` key:

| Token | Study | Reference key |
|---|---|---|
| `M17` | Mangalathu (2017), including the rows the file credits to all six multi-span studies or to five of them, whose capacities are those of its Table 6.13 | `mangalathu2017` |
| `MCP18` | Mangalathu et al. (2018) | `mangalathu2018` |
| `MSJ17` | Mangalathu, Soleimani, and Jeon (2017) | `mangalathu2017b` |
| `CPR22` | Cavalcante et al. (2022) | `cavalcante2022` |
| `SXR22` | Shao et al. (2022) | `shao2022` |

Every model also cites `chen2025`. Other works that a model draws on, such as
Ramanathan (2012), are cited in its Comments.

Before it builds any metadata, the generator checks `../models.csv` against the
published file. The model IDs must be unique and of the form above. The
notation of each row must be the component named in its ID. A column ID must
carry the qualifier of its EDP, `CD` for "Curvature ductility" and `DR` for
"Column drift (%)", and no other ID may carry one. The component and qualifier
must name a demand type, and the source token must name the study that the
`References` of the row credit, read without an "except for" clause ("All six
listed in Table 3 of the paper" for `M17`). With the repeated row removed, the
notation, EDP, four medians, and dispersion of `../models.csv` must equal those
of the published file, row for row. For each published row, the `Notes` and
`References` of its copies must together name the same items as the published
strings, where an item is the text between commas, semicolons, and colons; each
copy of row 6 must name only some of them. The generator also stops if a row
has a populated damage state above an empty one, unless the row defines only
the extensive and complete states and is an unseating row, and if the unit of a
row does not fit the unit type of its demand type in `src/dlml/vocabulary.py`.

## metadata_text.json

`../metadata_text.json` holds the text of the model metadata that the
published file does not contain: what was taken from the source publications
(the capacity sources and table numbers, the EDP definitions, the bridges each
study covers) and the sentence templates the generator fills. Its keys are
fragments of the model ID, such as a component, a scope token, or
`<component>.<source>`, so the text of a model follows from its ID. Some keys
match one model only, such as `Unseating.E3.M17`. Below, `<kind>` is the
component with its qualifier (`Column.CD`, `Seal`), and the neighbor folder is
`../../California RC Bridges 2025/`, which generates the Sa(1.0 s)-based
datasets.

- `component_phrases` and `component_groups`, by component: the noun of the
  model Description and the label of the component group.
- `scopes`, by scope token: the scope phrase of the Description. Where one
  scope token needs a different phrase by source, the key is
  `<scope>.<source>`, which the generator tries before `<scope>` (`E1.MSJ17`
  and `E1.MCP18` before `E1`). The `MCP18`, `E1.MSJ17`, and `SingleSpan`
  phrases name the bridge groups of the row with `{group}`.
- `capacity`: the capacity paragraph of the Comments. For a model, the
  generator joins with a space, in this order, the entries that exist among
  `<kind>`, `<kind>.<source>`, and `<kind>.<scope>.<source>`, and, for a model
  of the `MultiSpan` scope only, the entry `MultiSpan`. It stops if none of
  the first three exists. Text that belongs to one study is keyed
  `<kind>.<source>` (`Key.M17`, `Column.CD.M17`, `Column.DR.MCP18`,
  `Unseating.MSJ17`, `Bearing.CPR22`, `Approach.SXR22`). A bare `<kind>` key
  is for text true for every source; the file has none. A
  `<kind>.<scope>.<source>` entry adds what applies to one scope of a study:
  `CRCB.Unseating.E3.M17` takes `Unseating.M17` (the era capacity sets) and
  `Unseating.E3.M17` (the notes on Era 3). `MultiSpan` is the sentence shared
  by the nine models of that scope.
- `use`: the Use paragraph of the Comments. Its `template` is filled with the
  words of `default`, replaced by those of the source token in `sources` and
  then by those of the component in `components`.
- `comments`: the provenance paragraph, the `row_split` sentence added to the
  models of a published row that appears twice in `../models.csv`, and the
  paragraph on filled states.
- `limit_states`: the changes to the damage-state wording of the neighbor
  folder, described under [Damage-state
  descriptions](#damage-state-descriptions).
- `description`, `eras`, and `group_root`: the template of the model
  Description, the era years of the dataset description, and the root of the
  component groups.

The generator derives these placeholder values from the ID and the row:

| Placeholder | Value | Applies to |
|---|---|---|
| `{era}` | "Era 1" | an era scope (`E1`, `E2`, `E3`) |
| `{group}` | "group E1-S3P-C1-D", "groups E123-S1-C0-D and E123-S1-C0-S" | a row whose `Notes` name bridge group codes |
| `{length}` | "10", the slab length in feet | a `Slope` scope; used by the neighbor's slab sentences |
| `{conversions}` | "0.7% as 0.007, 1.5% as 0.015, 2.5% as 0.025, and 5% as 0.05" | a column drift row |
| `{scope}` | the scope phrase from `scopes` | every model |
| `{percent}` | the published median of the state, "0.7" | every state with a published median; the column drift notes use it |

`scopes` entries may use the first four, `capacity` entries the first five,
and the `limit_states.thresholds` notes all six. A template that uses a
placeholder the model lacks, such as `{era}` for a `MultiSpan` model, stops
the run with a message naming the models.csv row and the text key. So does a
placeholder that occurs twice in one template. After it builds all models,
the generator also stops if a key of `scopes`, `capacity`, `use.sources`, or
`use.components` was used by no model, which catches a misspelled key.

Adding a model takes one row of `../models.csv` with its ID, copied from a row
of the published file. The text file and the generator need more only when
the ID brings something new. A new component and source pair needs a
`capacity` entry; without one, the generator stops, so no model takes the
text of another study. A new source also needs a `use.sources` variant and a
token in the generator's two source tables, `SOURCE_KEYS` and
`SOURCE_STUDIES`, and a new scope token needs a `scopes` entry. A new
component also needs its `component_phrases`, `component_groups`, and
`DEMAND_TYPES` entries and damage-state wording in the neighbor folder.

## Rows and models

Each published row gives one model, except row 6, which gives two. Every model
has the dispersion of 0.35 that the file assigns to all rows.

| Row | Model ID | Demand type | Unit | States |
|---|---|---|---|---|
| 0, 1, 2 | `CRCB.Column.CD.E1.M17`, `CRCB.Column.CD.E2.M17`, `CRCB.Column.CD.E3.M17` | Peak Column Curvature Ductility | unitless | 4 |
| 3, 4, 5 | `CRCB.Column.DR.E1.MCP18`, `CRCB.Column.DR.E2.MCP18`, `CRCB.Column.DR.E3.MCP18` | Peak Column Drift Ratio | unitless | 4 |
| 6 | `CRCB.Unseating.E1.MSJ17` | Peak Joint Opening | inch | 4 (2 filled) |
| 6 | `CRCB.Unseating.SingleSpan.CPR22` | Peak Joint Opening | inch | 4 (2 filled) |
| 7, 8, 9 | `CRCB.Unseating.E1.M17`, `CRCB.Unseating.E2.M17`, `CRCB.Unseating.E3.M17` | Peak Joint Opening | inch | 4 |
| 10 | `CRCB.AbAct.MultiSpan.M17` | Peak Abutment Active Displacement | inch | 2 |
| 11 | `CRCB.AbPass.MultiSpan.M17` | Peak Abutment Passive Displacement | inch | 2 |
| 12 | `CRCB.AbTran.MultiSpan.M17` | Peak Abutment Transverse Displacement | inch | 2 |
| 13 | `CRCB.Bearing.MultiSpan.M17` | Peak Bearing Deformation | inch | 2 |
| 14 | `CRCB.Bearing.SingleSpan.CPR22` | Peak Bearing Deformation | inch | 2 |
| 15 | `CRCB.DeckMax.MultiSpan.M17` | Peak Deck Displacement | inch | 2 |
| 16 | `CRCB.FndRot.MultiSpan.M17` | Peak Foundation Rotation | rad | 2 |
| 17 | `CRCB.FndTran.MultiSpan.M17` | Peak Foundation Translation | inch | 2 |
| 18 | `CRCB.Key.MultiSpan.M17` | Peak Shear Key Deformation | inch | 2 |
| 19 | `CRCB.Seal.MultiSpan.M17` | Peak Joint Opening | inch | 2 |
| 20 | `CRCB.Approach.Step.SXR22` | Permanent Approach Settlement | inch | 2 |
| 21, 22, 23 | `CRCB.Approach.Slope10.SXR22`, `CRCB.Approach.Slope30.SXR22`, `CRCB.Approach.Slope45.SXR22` | Permanent Approach Settlement | inch | 2 |

Rows 0 to 2, 7 to 13, and 15 to 19 cite "All six listed in Table 3 of the
paper"; rows 0 to 2 and 7 to 9 each exclude one study. These are the six studies
from which Chen et al. (2025) extracted curves for multi-span bridges:
Mangalathu (2017); Soleimani (2017); Soleimani, Mangalathu, and DesRoches
(2017); Mangalathu, Soleimani, and Jeon (2017); Mangalathu et al. (2018); and
Jeon, Mangalathu, and Lee (2019). Rows 0 to 2 exclude Mangalathu et al. (2018),
whose column curves use column drift (rows 3 to 5), and rows 7 to 9 exclude
Mangalathu, Soleimani, and Jeon (2017), whose unseating capacities are those of
row 6. The file cites that work as "Mangalathu, Soleimani, et al. (2017)". The
IDs with the scope `MultiSpan` carry the capacities the file assigns to all six
studies; they do not cover the single-span bridge groups.

## Changes to the published values

### Column drift converted from percent to ratio

Rows 3 to 5 give column drift in percent. The demand type Peak Column Drift
Ratio is a ratio, so the generator divides each value by 100 and writes it
without trailing zeros.

| Row | Model ID | Published (%) | Stored |
|---|---|---|---|
| 3 | `CRCB.Column.DR.E1.MCP18` | 0.7, 1.5, 2.5, 5 | 0.007, 0.015, 0.025, 0.05 |
| 4 | `CRCB.Column.DR.E2.MCP18` | 1, 2.5, 5, 7.5 | 0.01, 0.025, 0.05, 0.075 |
| 5 | `CRCB.Column.DR.E3.MCP18` | 1, 2.5, 7.5, 10 | 0.01, 0.025, 0.075, 0.1 |

The damage-state descriptions quote both numbers, for example "0.007 (0.7%)".

### Row 6: split into two models, with filled lower states

Row 6 lists four bridge groups, E1-S2-C2P-S, E1-S3P-C2P-S, E123-S1-C0-D, and
E123-S1-C0-S, and cites two studies. Its capacities of 6 and 9 in. are those of
Mangalathu, Soleimani, and Jeon (2017, Table 4), 152 and 229 mm, for the two Era
1 groups, and those of Cavalcante et al. (2022, Table 8), also 152 and 229 mm,
for the two single-span groups. The row is represented here by two models with
identical parameters: `CRCB.Unseating.E1.MSJ17` for the Era 1 groups and
`CRCB.Unseating.SingleSpan.CPR22` for the single-span groups. In
`../models.csv` the row appears twice. The `Notes` of each copy name the two
groups of its model, and the `References` name its study: "Mangalathu,
Soleimani, et al. (2017)" for `CRCB.Unseating.E1.MSJ17` and "Cavalcante et al.
(2022)" for `CRCB.Unseating.SingleSpan.CPR22`. The Comments of both models
quote the published row.

The row defines the extensive and complete states only; both studies define
unseating capacity for those two states. The slight and moderate cells are
empty. The generator sets the slight and moderate medians equal to the 
extensive median of 6 in. The extensive and complete states keep the numbers 
DS3 and DS4, and the slight and moderate states cannot occur. The LS1 and LS2 
descriptions of the two models say that the state is not defined by the source.

## Values stored as published

The following values are stored as the file prints them, although a source
study gives a different value or the evidence suggests a different one. Each
model's Comments give the same evidence.

- Foundation rotation (row 16): 1.5 and 6 rad, as in Ramanathan (2012) Table
  5.15 and Mangalathu (2017) Table 6.13. Ramanathan (2012, p. 181) states the
  basis as an axial pile movement of ±0.5 in. at the opposite edges of a 20 ft
  wide pile cap, about 0.004 rad. The published Sa(1.0 s) curves are consistent
  with the values having been applied in radians.
- Shear key (row 18): slight capacity 1 in., as in Mangalathu (2017) Table
  6.13; Ramanathan (2012) Table 5.16 gives 1.5 in.
- Joint seal (row 19): 2 and 5 in. With the demand model of the unseating
  curves of the same bridge class, the Sa(1.0 s) seal curves of Mangalathu
  (2017) reproduce the slight state with a 2 in. capacity but imply about 75 in.
  for the moderate state.
- Unseating, Era 3 (row 9): 1.5, 4.5, 14, and 21 in. The Era 3 Sa(1.0 s)
  unseating curves of Mangalathu (2017, Appendix C) equal its Era 2 curves, so
  these capacities were not used to build them.
- Bearing (row 13): 1 and 4 in. Mangalathu, Soleimani, and Jeon (2017, Table
  4) give a moderate capacity of 76 mm (3 in.) for the bearings of the Era 1
  bridges they model.
- Column curvature ductility (rows 0 to 2): the file assigns these capacities
  to Soleimani, Mangalathu, and DesRoches (2017), among others, but that study
  uses column displacement ductility as the column EDP (p. 469). Its curves in
  the database are system curves only.
- Approach settlement (rows 20 to 23): the file gives inches. Shao et al.
  (2022, Table 2) give the capacities in cm, for configurations they label STO
  (step offset) and SLO-S, SLO-M, and SLO-L (slope offset with a 10, 30, or 45
  ft slab). The stored values are those of the file, which equal the cm values
  divided by 2.54 and rounded to 0.1 in. The Minor (LS2) and Moderate (LS3)
  states of the study are the slight and moderate states of the models. The
  Comments of these models name the source table but not the cm values.

| Row | Model ID | Shao et al. label | Minor, LS2 (cm) | Moderate, LS3 (cm) | Stored (in.) |
|---|---|---|---|---|---|
| 20 | `CRCB.Approach.Step.SXR22` | STO | 3.56 | 6.35 | 1.4, 2.5 |
| 21 | `CRCB.Approach.Slope10.SXR22` | SLO-S | 5.33 | 10.67 | 2.1, 4.2 |
| 22 | `CRCB.Approach.Slope30.SXR22` | SLO-M | 16.26 | 32.26 | 6.4, 12.7 |
| 23 | `CRCB.Approach.Slope45.SXR22` | SLO-L | 24.13 | 48.51 | 9.5, 19.1 |

## Capacity sources

The Comments of each model name the table its capacities come from. The column
curvature ductility and the four-state unseating capacities are those of
Mangalathu (2017) Table 6.13, which Soleimani (2017) Table 7.5 repeats. The
two-state capacities of the abutments, bearings, deck, column foundations,
shear keys, and joint seals are those of Ramanathan (2012) Tables 5.14 to 5.17
as adopted by Mangalathu (2017) Table 6.13, with the shear key exception above;
the exception among the bearings is `CRCB.Bearing.SingleSpan.CPR22` (row 14),
whose capacities are those of Cavalcante et al. (2022, Table 8). The column
drift capacities are those of Mangalathu et al. (2018, Table 2), and the
approach capacities are the Minor (LS2) and Moderate (LS3) settlement
capacities of Shao et al. (2022, Table 2), converted from cm to inches.

## Damage-state descriptions

The LimitStates descriptions reuse the damage-state wording of the Sa(1.0 s)
datasets, which `../../California RC Bridges 2025/metadata_text.json` holds in
its `limit_states` block; the generator reads it from there. The `limit_states`
block of `../metadata_text.json` changes that wording by kind or scope. Its
`thresholds`, keyed by `<kind>` and, for one model, by
`<kind>.<scope>.<source>`, give the column drift words and replace the notes
of the column curvature ductility and unseating states with "for {era}
bridges" ("for single-span bridges" for `CRCB.Unseating.SingleSpan.CPR22`);
both foundation rotation states use the short note (the basis of the values is
in the Comments). Its `bases` choose the neighbor's approach sentences by
scope token ("Approach Step" or "Approach Slope"), and its `labels` name the
Shao et al. (2022) state each approach state corresponds to.
Each description quotes the stored median capacity of its state, in the unit of
the model. The column drift descriptions add the published percentage, and the
two row-6 models describe LS1 and LS2 as not defined by the source. The sources
of the wording are listed in section "Damage-state descriptions" of
`../../California RC Bridges 2025/source/README.md`.

## Number formatting

Every median is written to `fragility.csv` as the file prints it (`0.8`, `1`,
`6`, `12.7`). The file prints 1 and 6 where Mangalathu (2017) Table 6.13
prints 1.0 and 6.0, and the stored values follow the file. The column drift
ratios are the published percentages divided by 100, written without trailing
zeros (0.7 becomes 0.007, 10 becomes 0.1). The dispersion is 0.35 in every row
and is written as printed.
