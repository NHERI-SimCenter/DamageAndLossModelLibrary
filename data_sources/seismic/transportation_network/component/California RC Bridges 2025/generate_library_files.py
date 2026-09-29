"""Generate the California RC Bridges 2025 fragility datasets.

Reads the Sa(1.0 s)-based fragility table of the Chen et al. (2025) database
(DesignSafe PRJ-5910, version 1, ``source/Sa_1_Fragility Database.csv``) and
writes two datasets under ``src/dlml/data/seismic/transportation_network``:

- ``portfolio/California RC Bridges 2025``: the bridge system curves;
- ``component/California RC Bridges 2025``: the bridge component curves.

It also writes ``id_crosswalk.csv`` beside this script, which maps every model
ID to its row in the source file and lists the rows that were omitted.

The damage-state descriptions quote median capacities that are confirmed
against ``source/EDP_Fragility Database.csv`` of the same project; that file
changes no fragility parameter. Every change to the published labels and values
is recorded in ``source/README.md``.

The metadata text (phrase tables, source-study paragraphs, correction
sentences, and damage-state descriptions) is read from ``metadata_text.json``,
the dataset descriptions from ``general_information.json``, and the citations
from ``references.json``, all beside this script.

Run from this folder with the repository environment::

    uv run python generate_library_files.py

Row numbers below are 0-based data rows of the source file; the spreadsheet
row is the data row plus 2. The script checks the source file as it goes, and
a failed check stops the run with a message that names the offending rows
where they are known.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
SOURCE_FILE = HERE / 'source' / 'Sa_1_Fragility Database.csv'
EDP_FILE = HERE / 'source' / 'EDP_Fragility Database.csv'
DATA_ROOT = (
    REPO_ROOT / 'src' / 'dlml' / 'data' / 'seismic' / 'transportation_network'
)
DATASET_NAME = 'California RC Bridges 2025'
OUTPUT = {
    'portfolio': DATA_ROOT / 'portfolio' / DATASET_NAME,
    'component': DATA_ROOT / 'component' / DATASET_NAME,
}
CROSSWALK_FILE = HERE / 'id_crosswalk.csv'
SOURCE_VERSION = 'PRJ-5910 v1, 2025-04-29'

TEXT = json.loads((HERE / 'metadata_text.json').read_text(encoding='utf-8'))

STATES = ['Slight', 'Moderate', 'Extensive', 'Complete']
PARAMS = [f'{state}_{par}' for state in STATES for par in ('λ', 'ζ')]
COLUMNS = ['Bridge group', 'Fragility notation', *PARAMS, 'Notes', 'Reference']
EXPECTED_ROWS = 1018
# Decimals kept for the recomputed values of data rows 0 to 7; the source
# values have at most two decimals and are written as published.
RECOMPUTED_DECIMALS = 4

PRIMARY = {'Column', 'Unseating', 'System'}

# Group-code levels in ID order; each table gives the Description phrase, and
# the ComponentGroups label is the phrase without its leading article.
GROUP_LEVELS = [
    ('era', TEXT['era_phrases']),
    ('spans', TEXT['span_phrases']),
    ('bents', TEXT['bent_phrases']),
    ('abutment', TEXT['abutment_phrases']),
]
COMPONENT_PHRASES = TEXT['component_phrases']

# ---------------------------------------------------------------------------
# Source studies
# ---------------------------------------------------------------------------

# normalized Reference string -> source token
SOURCE_TOKENS = {
    'Mangalathu (2017)': 'M17',
    'Mangalathu, Soleimani, et al. (2017)': 'MSJ17',
    'Mangalathu et al. (2018)': 'MCP18',
    'Soleimani (2017)': 'S17',
    'Soleimani et al. (2017)': 'SMD17',
    'Jeon, Mangalathu and Lee (2019)': 'JML19',
    'Shao et al. (2022)': 'SXR22',
    'Cavalcante et al. (2022)': 'CPR22',
}

REFERENCE_KEYS = {
    'M17': 'mangalathu2017',
    'MSJ17': 'mangalathu2017b',
    'MCP18': 'mangalathu2018',
    'S17': 'soleimani2017',
    'SMD17': 'soleimani2017b',
    'JML19': 'jeon2019',
    'SXR22': 'shao2022',
    'CPR22': 'cavalcante2022',
}

# JML19 rows whose parameters equal another configuration's at two decimals;
# checked against Jeon, Mangalathu, and Lee (2019) Tables 4 and 5.
JML19_TWIN_ROWS = {332, 333, 883, 884}

# ---------------------------------------------------------------------------
# Notes -> attribute tokens
# ---------------------------------------------------------------------------

SLOTS = [
    'deck',
    'bent',
    'span',
    'frames',
    'disc',
    'shape',
    'direction',
    'mod',
    'abutfnd',
    'approach',
]

_MOD = 'modification factors computed using Era {} bridge fragilities'

# (slot, pattern, token); applied in this order, each match removed from the
# string before the next pattern is tried.
NOTE_PATTERNS = [
    ('deck', r'Box-girder', 'BoxGirder'),
    ('deck', r'Slab deck', 'Slab'),
    ('deck', r'T-beam', 'TBeam'),
    ('frames', r'multiple frames', 'MF'),
    ('disc', r'discontinuous spans', 'Disc'),
    ('disc', r'only for discontinuous bridges', 'Disc'),
    ('shape', r'one-way flared column along the transverse direction', 'Flr1T'),
    ('shape', r'one-way flared column', 'Flr1'),
    ('shape', r'two-way flared column', 'Flr2'),
    ('shape', r'circular column', 'Circ'),
    ('shape', r'oblong column', 'Obl'),
    ('shape', r'rectangular column', 'Rect'),
    ('direction', r'longitudinal direction', 'Lng'),
    ('direction', r'transverse direction', 'Trn'),
    ('mod', re.escape(_MOD.format('2 and Era 3')), 'Mod23'),
    ('mod', re.escape(_MOD.format('1')), 'Mod1'),
    ('mod', re.escape(_MOD.format('2')), 'Mod2'),
    ('mod', re.escape(_MOD.format('3')), 'Mod3'),
    ('abutfnd', r'Abutment on piles', 'Pile'),
    ('abutfnd', r'Cantilever abutment on spread footing', 'Cant'),
    ('abutfnd', r'Regular-height abutment on spread footing', 'Reg'),
    ('approach', r'with approach slab damage through step offset', 'Step'),
    (
        'approach',
        r'with approach slab \(length (10|30|45) ft\) damage through slope offset',
        'Slope{}',
    ),
    (
        'approach',
        r'with approach slab damage through slope offset, with an approach slab '
        r'length of (10|30|45) ft',
        'Slope{}',
    ),
]

# ---------------------------------------------------------------------------
# Damage-state thresholds
# ---------------------------------------------------------------------------

# The median capacities quoted in the damage-state descriptions (LimitStates),
# one value per damage state, keyed by (component, threshold set, variant);
# None where a description quotes no number. They are text only and change no
# fragility parameter; the fragility curves come from the Sa(1.0 s) file alone.
# Every value comes from a row of source/EDP_Fragility Database.csv of the same
# DesignSafe project, and check_thresholds() confirms it against that file. The
# row for the Cavalcante et al. (2022) bearing carries the values of that
# study's Table 8; the Approach rows carry the Shao et al. (2022) Table 2 LS2 and
# LS3 values converted from cm to inches. The System descriptions quote no
# number, and the moderate Seal description quotes none because the published
# curve does not match the 5 in. of the file (see the seal Comments sentence).
# The unit and the wording of each set are in metadata_text.json. Values are
# written as the EDP file prints them, except the Key slight 1.0 and the FndRot
# moderate 6.0, which follow Mangalathu (2017) Table 6.13 (the EDP file prints 1
# and 6).
UNSEATING_M17_E2 = [1, 4.5, 10, 15]  # joint opening, in.
SECONDARY = {  # slight, moderate; generic capacity set of Mangalathu (2017)
    'AbAct': [1.5, 4],
    'AbPass': [3, 10],
    'AbTran': [1, 4],
    'Bearing': [1, 4],
    'DeckMax': [4, 12],
    'FndTran': [1, 4],
    'FndRot': [1.5, 6.0],  # rad, as published
    'Key': [1.0, 5],
    'Seal': [2, None],
}
CAPACITIES = {
    # curvature ductility
    ('Column', '', 'E1'): [0.8, 2, 5, 8],
    ('Column', '', 'E2'): [1, 5, 8, 11],
    ('Column', '', 'E3'): [1, 5, 11, 17],
    # percent, as published
    ('Column', 'drift', 'E1'): [0.7, 1.5, 2.5, 5],
    ('Column', 'drift', 'E2'): [1, 2.5, 5, 7.5],
    ('Column', 'drift', 'E3'): [1, 2.5, 7.5, 10],
    ('Unseating', '', 'E1'): [0.5, 1, 2, 3],
    ('Unseating', '', 'E2'): UNSEATING_M17_E2,
    ('Unseating', 'E3', 'E2'): UNSEATING_M17_E2,
    ('Unseating', 'two-state', 'MCP18 E1'): [None, None, 2, 3],
    ('Unseating', 'two-state', 'MCP18 E2'): [None, None, 10, 15],
    ('Unseating', 'two-state', 'MCP18 E3'): [None, None, 14, 21],
    ('Unseating', 'two-state', 'JML19 E2'): [None, None, 10, 15],
    ('Unseating', 'two-state', 'CPR22 E123'): [None, None, 6, 9],
    ('Bearing', 'CPR22', ''): [1, 3],  # Cavalcante et al. (2022) Table 8
    # Shao et al. (2022) Table 2 LS2 and LS3 values, converted from cm to inches
    ('Approach', '', 'Step'): [1.4, 2.5],
    ('Approach', '', 'Slope10'): [2.1, 4.2],
    ('Approach', '', 'Slope30'): [6.4, 12.7],
    ('Approach', '', 'Slope45'): [9.5, 19.1],
    ('System', '', ''): [None] * 4,
    **{(component, '', ''): values for component, values in SECONDARY.items()},
}

SHAPE_FIXES = {298: ('circular', 'oblong'), 299: ('oblong', 'circular'),
               741: ('circular', 'rectangular')}  # fmt: skip
SWAPS = {26: ('AbAct', 'AbPass'), 27: ('AbAct', 'AbPass'),
         28: ('AbPass', 'AbAct'), 29: ('AbPass', 'AbAct')}  # fmt: skip
OTHER_DIRECTION = {'AbAct': 'AbPass', 'AbPass': 'AbAct'}
DIRECTION_WORDS = {'AbAct': 'active', 'AbPass': 'passive'}


def fail(message: str, rows=()) -> None:
    """Stop the run with a message and the offending rows, if given."""
    rows = list(rows)
    suffix = f' (data rows {rows})' if rows else ''
    raise SystemExit(f'ERROR: {message}{suffix}')


def check(condition: bool, message: str, rows=()) -> None:  # noqa: FBT001
    """Stop the run unless ``condition`` holds."""
    if not condition:
        fail(message, rows)


def squash(text: str) -> str:
    """Strip and collapse whitespace."""
    return ' '.join(str(text).split())


def normalize_notes(text: str) -> str:
    """Normalize a Notes string for token matching."""
    text = squash(text).rstrip('.')
    return text.replace('by using', 'using')


def fmt(value: float) -> str:
    """Format a parameter without float noise, padding, or trailing zeros.

    Source values have at most two decimals and no trailing zeros (both checked
    in read_source), so they are written exactly as published; the recomputed
    values of data rows 0 to 7 keep RECOMPUTED_DECIMALS decimals.
    """
    if pd.isna(value):
        return ''
    return f'{value:.{RECOMPUTED_DECIMALS}f}'.rstrip('0').rstrip('.')


def join_phrases(items: list) -> str:
    """Join phrases as 'a', 'a and b', or 'a, b, and c'."""
    if len(items) <= 1:
        return ''.join(items)
    if len(items) == 2:  # noqa: PLR2004
        return f'{items[0]} and {items[1]}'
    return ', '.join(items[:-1]) + f', and {items[-1]}'


def sentence(text: str) -> str:
    """End ``text`` with a period unless it already ends with one."""
    return text if text.endswith('.') else f'{text}.'


def direction_word(notation: str) -> str:
    """'active' or 'passive' for AbAct or AbPass."""
    check(notation in DIRECTION_WORDS, f'no direction word for {notation}')
    return DIRECTION_WORDS[notation]


def era_tag(era: str) -> str:
    """'Era 1' for E1, and so on; 'any era' for E123."""
    return 'any era' if era == 'E123' else f'Era {era.removeprefix("E")}'


def fill(template: str, **values: str) -> str:
    """Fill the named placeholders, each of which must occur exactly once."""
    for key in values:
        check(
            template.count(f'{{{key}}}') == 1,
            f'placeholder {{{key}}} must occur exactly once',
        )
    result = template.format(**values)
    check('{' not in result and '}' not in result, 'unresolved placeholder')
    return result


def without_article(phrase: str) -> str:
    """A group phrase without its leading article, for ComponentGroups."""
    return re.sub(r'^(a|an) ', '', phrase)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def read_source() -> pd.DataFrame:
    """Read the source table and normalize the text columns."""
    raw = pd.read_csv(
        SOURCE_FILE, encoding='utf-8-sig', dtype=str, keep_default_na=False
    )
    check(
        len(raw) == EXPECTED_ROWS,
        f'expected {EXPECTED_ROWS} rows, found {len(raw)}',
    )
    check(
        list(raw.columns) == COLUMNS,
        f'unexpected columns {list(raw.columns)}',
    )
    table = pd.DataFrame(index=raw.index)
    table['group'] = raw['Bridge group'].map(squash)
    table['notation_raw'] = raw['Fragility notation']
    table['notation'] = raw['Fragility notation'].map(squash)
    table['notes_raw'] = raw['Notes']
    table['reference_raw'] = raw['Reference']
    table['notes'] = raw['Notes'].map(squash)
    table['reference'] = raw['Reference'].map(squash)
    for par in PARAMS:
        cells = raw[par].str.strip()
        table[f'published_{par}'] = cells
        table[f'na_{par}'] = cells == 'N/A'
        numeric = ~cells.isin(['', 'N/A'])
        malformed = numeric & ~cells.str.fullmatch(r'\d+(\.\d{0,1}[1-9])?')
        check(
            not malformed.any(),
            f'{par} cells with more than two decimals or a trailing zero',
            raw.index[malformed],
        )
        numbers = pd.to_numeric(cells.where(numeric), errors='coerce')
        bad = cells.ne('') & cells.ne('N/A') & numbers.isna()
        check(not bad.any(), f'non-numeric {par} cells', raw.index[bad])
        table[par] = numbers.astype(float)
    table['omit'] = ''
    table['filled'] = False
    table['corrections'] = [[] for _ in range(len(table))]
    table['comments_extra'] = [[] for _ in range(len(table))]

    unknown = sorted(set(table['reference']) - set(SOURCE_TOKENS))
    check(
        not unknown,
        f'unknown Reference strings {unknown}',
        table.index[table['reference'].isin(unknown)],
    )
    table['source'] = table['reference'].map(SOURCE_TOKENS)
    parts = table['group'].str.split('-')
    check(
        bool(parts.map(len).eq(4).all()),
        'malformed Bridge group codes',
        table.index[parts.map(len).ne(4)],
    )
    for i, (col, phrases) in enumerate(GROUP_LEVELS):
        table[col] = parts.str[i]
        bad = ~table[col].isin(list(phrases))
        check(not bad.any(), f'unknown {col} code', table.index[bad])
    bad = ~table['notation'].isin(list(COMPONENT_PHRASES))
    check(not bad.any(), 'unknown Fragility notation', table.index[bad])
    return table


def same_parameters(table: pd.DataFrame, first_row: int, second_row: int) -> bool:
    """Whether two data rows hold the same eight parameters."""
    first_values = table.loc[first_row, PARAMS].to_numpy(dtype=float)
    second_values = table.loc[second_row, PARAMS].to_numpy(dtype=float)
    return bool(np.array_equal(first_values, second_values, equal_nan=True))


def drop_duplicates(table: pd.DataFrame) -> None:
    """Omit the six Soleimani (2017) System rows that repeat Mangalathu (2017).

    In three single-column, multi-span seat-abutment groups the source lists
    six Mangalathu (2017) System curves a second time, attributed to Soleimani
    (2017). The evidence is that all eight parameters of each repeated row equal
    those of its Mangalathu (2017) partner, which is checked here; the partner
    is kept. The pairs are listed in source/README.md.
    """
    duplicate_partners = {742: 740, 744: 741, 835: 833, 837: 834, 957: 955, 959: 956}
    for duplicate_row, partner_row in duplicate_partners.items():
        check(
            table.loc[duplicate_row, 'reference'] == 'Soleimani (2017)',
            'duplicate row is not a Soleimani (2017) row',
            [duplicate_row],
        )
        check(
            table.loc[partner_row, 'reference'] == 'Mangalathu (2017)',
            'duplicate partner is not a Mangalathu (2017) row',
            [partner_row],
        )
        check(
            same_parameters(table, duplicate_row, partner_row),
            'duplicate row differs from its Mangalathu (2017) partner',
            [duplicate_row, partner_row],
        )
        table.loc[duplicate_row, 'omit'] = 'duplicate'


def drop_all_na(table: pd.DataFrame) -> pd.DataFrame:
    """Omit the 95 rows without parameters; return the omitted rows.

    These component rows give N/A (or nothing) for all eight parameters, which
    the database uses to mark a negligible probability of damage from ground
    shaking, so there is no curve to store. The run stops if any System row is
    among them. The dataset Description summarizes them by component, and
    source/README.md lists them by component and group.
    """
    empty = table[PARAMS].isna().all(axis=1) & table['omit'].eq('')
    rows = list(table.index[empty])
    check(len(rows) == 95, f'expected 95 all-N/A rows, found {len(rows)}', rows)  # noqa: PLR2004
    systems = [row for row in rows if table.loc[row, 'notation'] == 'System']
    check(not systems, 'all-N/A System rows', systems)
    table.loc[rows, 'omit'] = 'all_na'
    return table.loc[rows, ['group', 'notation']]


def add_correction(table: pd.DataFrame, row: int, kind: str, text: str) -> None:
    """Record a correction in the crosswalk column and the model comments."""
    table.at[row, 'corrections'].append(kind)
    table.at[row, 'comments_extra'].append(text)


def correct_labels(table: pd.DataFrame) -> None:
    """Correct three column-shape labels and four swapped abutment directions.

    The Notes of data rows 298 and 741 name the wrong column shape: their values
    match the oblong and the rectangular class of the Mangalathu (2017) thesis.
    Row 299 is labeled oblong, but Mangalathu et al. (2018) model circular
    columns only. Rows 26 to 29 carry the Cavalcante et al. (2022, Table 9)
    values of the other abutment direction, so AbAct and AbPass are exchanged.
    The values are not changed; see source/README.md for the full record.
    """
    for row, (old, new) in SHAPE_FIXES.items():
        notes = table.loc[row, 'notes']
        check(
            f'{old} column' in notes,
            f'row Notes {notes!r} does not contain "{old} column"',
            [row],
        )
        table.loc[row, 'notes'] = notes.replace(f'{old} column', f'{new} column')
        add_correction(table, row, 'label', TEXT['corrections']['shape'][str(row)])

    for row, (old, new) in SWAPS.items():
        check(
            table.loc[row, 'notation'] == old
            and table.loc[row, 'source'] == 'CPR22'
            and table.loc[row, 'group'] == 'E123-S1-C0-S',
            f'expected a Cavalcante et al. (2022) {old} row in E123-S1-C0-S',
            [row],
        )
        table.loc[row, 'notation'] = new
        add_correction(
            table,
            row,
            'swap',
            TEXT['corrections']['swap'].format(
                old=old, new=new, direction=direction_word(new)
            ),
        )


def recompute_single_span_diaphragm(table: pd.DataFrame) -> None:
    """Recompute data rows 0 to 7, which inherit the swap of rows 26 to 29.

    Chen et al. (2025) found no single-span diaphragm-abutment curves and built
    the AbAct and AbPass curves of E123-S1-C0-D (rows 0 to 7) from the seat
    curves of E123-S1-C0-S with modification factors: the ratios of the
    diaphragm-abutment to the seat-abutment fragility parameters of the
    two-span, single-column groups of the named era, applied to the medians and
    the dispersions. The database does not list the factors. They were applied
    by the labels of the seat rows, so the AbAct diaphragm curve was built on
    the seat curve labeled AbAct, which holds the passive-direction values
    (rows 26 and 27), and the AbPass curve on the active-direction values: the
    swap that correct_labels() undoes propagates into rows 0 to 7.

    Each published value is F * seat_as_labeled, where F is the factor for the
    row's direction and parameter and seat_as_labeled the value of the seat
    row with the same deck and the same label in the source. Keeping F and
    using the seat row that holds this direction after the relabeling gives

        recomputed = published * seat_correct / seat_as_labeled,

    so F need not be known. Results keep RECOMPUTED_DECIMALS decimals. The
    eight published and recomputed value sets are in source/README.md.
    """
    templates = TEXT['corrections']
    seat = table[
        (table['group'] == 'E123-S1-C0-S')
        & table['notation_raw'].isin(list(OTHER_DIRECTION))
    ]
    seat_deck = seat['tokens'].map(lambda tokens: tokens['deck'])
    for data_row in range(8):
        model = table.loc[data_row]
        check(
            model['group'] == 'E123-S1-C0-D'
            and model['source'] == 'CPR22'
            and model['notation'] in OTHER_DIRECTION,
            'expected a Cavalcante et al. (2022) AbAct or AbPass row in '
            'E123-S1-C0-D',
            [data_row],
        )
        other = OTHER_DIRECTION[model['notation']]
        same_deck = seat_deck == model['tokens']['deck']
        as_labeled = seat.index[
            same_deck & (seat['notation_raw'] == model['notation'])
        ]
        corrected = seat.index[same_deck & (seat['notation_raw'] == other)]
        check(
            len(as_labeled) == 1 and len(corrected) == 1,
            'seat-abutment partner rows not unique',
            [data_row, *as_labeled, *corrected],
        )
        labeled_row, corrected_row = as_labeled[0], corrected[0]
        check(
            corrected_row in SWAPS,
            'corrected seat-abutment row was not relabeled',
            [data_row, corrected_row],
        )
        changes, unchanged = [], []
        for par in PARAMS:
            published = model[par]
            if pd.isna(published):
                continue
            ratio = table.loc[corrected_row, par] / table.loc[labeled_row, par]
            recomputed = round(published * ratio, RECOMPUTED_DECIMALS)
            table.loc[data_row, par] = recomputed
            state, kind = par.split('_')
            name = f'{state.lower()} {"median" if kind == "λ" else "dispersion"}'
            if recomputed == published:
                unchanged.append(name)
                continue
            unit = ' g' if kind == 'λ' else ''
            changes.append(
                templates['recomputed_change'].format(
                    name=name,
                    old=f'{model[f"published_{par}"]}{unit}',
                    new=f'{fmt(recomputed)}{unit}',
                    ratio=f'{ratio:.4f}',
                )
            )
        if unchanged:
            changes.append(
                templates['recomputed_unchanged'].format(
                    names=join_phrases(unchanged)
                )
            )
        add_correction(
            table,
            data_row,
            'recomputed',
            templates['recomputed'].format(
                notation=model['notation'],
                labeled_row=labeled_row,
                correct_row=corrected_row,
                labeled_direction=direction_word(other),
                direction=direction_word(model['notation']),
                changes='; '.join(changes),
            ),
        )


def check_recomputed_slight_medians(table: pd.DataFrame) -> None:
    """Check rows 1 and 5 against factors computed from the source rows.

    The Era 2 and Era 3 slight-median factors F are the mean of the
    diaphragm-to-seat ratios of the circular- and oblong-column curves of
    E2-S2-C1-D (diaphragm) and E2-S2-C1-S (seat): data rows 114 and 115 over
    536 and 537 for AbAct, and 116 and 117 over 538 and 539 for AbPass. The
    Era 3 groups publish the same curves. F times the seat value as labeled
    must give the published slight median, and F times the seat value that
    holds the direction of the row must give the recomputed one, each within
    0.01 g. The derivation is in source/README.md.
    """
    tolerance = 0.01  # g, the published values have two decimals
    cases = {  # row: notation, (labeled, correct) seat row, diaphragm and seat rows
        1: ('AbAct', (26, 28), (114, 115), (536, 537)),
        5: ('AbPass', (28, 26), (116, 117), (538, 539)),
    }

    def slight(row: int) -> float:
        return float(table.loc[row, 'published_Slight_λ'])

    for data_row, case in cases.items():
        notation, seat_rows, diaphragm_rows, factor_seat_rows = case
        labeled_row, corrected_row = seat_rows
        check(
            table.loc[labeled_row, 'notation_raw'] == notation
            and all(
                table.loc[row, 'group'] == 'E2-S2-C1-D' for row in diaphragm_rows
            )
            and all(
                table.loc[row, 'group'] == 'E2-S2-C1-S' for row in factor_seat_rows
            )
            and all(
                table.loc[row, 'notation'] == notation
                for row in (*diaphragm_rows, *factor_seat_rows)
            ),
            f'unexpected rows for the {notation} slight-median factor',
            [data_row, labeled_row, *diaphragm_rows, *factor_seat_rows],
        )
        factor = np.mean(
            [
                slight(diaphragm) / slight(seat)
                for diaphragm, seat in zip(
                    diaphragm_rows, factor_seat_rows, strict=True
                )
            ]
        )
        check(
            abs(factor * slight(labeled_row) - slight(data_row)) < tolerance,
            f'factor {factor:.4f} times seat row {labeled_row} does not give the '
            f'published slight median {slight(data_row)}',
            [data_row],
        )
        expected = factor * slight(corrected_row)
        got = table.loc[data_row, 'Slight_λ']
        check(
            abs(got - expected) < tolerance,
            f'recomputed slight median {got} differs from {expected:.4f}',
            [data_row],
        )


def fix_sxr22_dispersion(table: pd.DataFrame) -> None:
    """Store 0.99 for the printed 1.99 moderate dispersion of ten SXR22 rows.

    The ten rows repeat one Shao et al. (2022) configuration (abutment on piles,
    10 ft approach slab, slope offset) in different groups, and the source and
    Shao et al. (2022, Table 6) both print 1.99. Shao et al. (2022, Sec. 4.4)
    define the dispersion of their integrated model as the square root of the
    sum of squares of the dispersions of three methods, and their Tables 3 to 5
    give 0.41, 0.62, and 0.65 for this configuration, which combine to 0.987.
    The medians are unchanged. See source/README.md for the full record.
    """
    rows = [17, 41, 63, 103, 247, 286, 466, 522, 739, 795]
    for row in rows:
        notes = table.loc[row, 'notes']
        check(
            table.loc[row, 'Moderate_ζ'] == 1.99  # noqa: PLR2004
            and table.loc[row, 'source'] == 'SXR22'
            and all(word in notes for word in ('piles', '10 ft', 'slope')),
            'expected an SXR22 piles, 10 ft slope-offset row with Moderate_ζ 1.99',
            [row],
        )
        table.loc[row, 'Moderate_ζ'] = 0.99
        add_correction(table, row, 'dispersion', TEXT['corrections']['dispersion'])


def fill_lower_states(table: pd.DataFrame) -> None:
    """Fill the slight and moderate states of nine two-state Unseating rows.

    The source defines unseating capacity for extensive and complete only in
    these rows and leaves the lower states empty. pelicun (checked in version
    3.10) cannot evaluate such a model: it ties the capacities of all limit
    states of a component to the random variable of the first limit state, so
    an empty LS1 below a populated LS3 raises an error in damage_model.py, and
    the DLML documentation plotter fails on the same layout. Setting the slight
    and moderate parameters equal to the extensive ones keeps the extensive and
    complete curves unchanged at DS3 and DS4; the lower states then cannot
    occur. See source/README.md.
    """
    rows = [24, 25, 249, 316, 400, 851, 852, 853, 854]
    kept = table['omit'].eq('')
    lower_empty = (
        table[['Slight_λ', 'Slight_ζ', 'Moderate_λ', 'Moderate_ζ']]
        .isna()
        .all(axis=1)
    )
    extensive = table[['Extensive_λ', 'Extensive_ζ']].notna().all(axis=1)
    found = list(table.index[kept & lower_empty & extensive])
    check(found == rows, f'rows with only upper states are {found}', found)
    bad = [row for row in rows if table.loc[row, 'notation'] != 'Unseating']
    check(not bad, 'rows with only upper states that are not Unseating', bad)
    for row in rows:
        for state in ('Slight', 'Moderate'):
            table.loc[row, f'{state}_λ'] = table.loc[row, 'Extensive_λ']
            table.loc[row, f'{state}_ζ'] = table.loc[row, 'Extensive_ζ']
        table.at[row, 'corrections'].append('filled_lower_states')
        table.loc[row, 'filled'] = True


def parse_notes(row: int, notes: str, source: str) -> dict:
    """Map a Notes string to attribute tokens; every fragment must match."""
    rest = normalize_notes(notes)
    tokens: dict = {}
    for slot, pattern, token in NOTE_PATTERNS:
        match = re.search(pattern, rest)
        if match is None:
            continue
        check(slot not in tokens, f'two {slot} tokens in Notes {notes!r}', [row])
        tokens[slot] = token.format(*match.groups())
        rest = rest[: match.start()] + ' | ' + rest[match.end() :]
        check(
            re.search(pattern, rest) is None,
            f'repeated fragment {pattern!r} in Notes {notes!r}',
            [row],
        )
    check(
        re.fullmatch(r'[\s,|]*', rest) is not None,
        f'unmatched Notes fragment {rest!r} in {notes!r}',
        [row],
    )
    if 'direction' in tokens:
        check(source == 'CPR22', 'direction token outside Cavalcante rows', [row])
    return tokens


def assign_tokens(table: pd.DataFrame) -> None:
    """Parse Notes for every row (after the label corrections)."""
    table['tokens'] = [
        parse_notes(row, table.loc[row, 'notes'], table.loc[row, 'source'])
        for row in table.index
    ]


def assign_bents_and_spans(table: pd.DataFrame) -> None:
    """Bent-type and span-count tokens that the Notes do not carry.

    Mangalathu (2017) blocks are taken over all source rows in file order,
    including the rows omitted later, so positions match the thesis classes.
    """
    m17_rows = table[(table['source'] == 'M17') & table['bents'].eq('C2P')]
    for (group, notation), block in m17_rows.groupby(
        ['group', 'notation'], sort=False
    ):
        rows = list(block.index)
        short = notation == 'Column' and group in ('E2-S2-C2P-D', 'E3-S2-C2P-D')
        check(
            len(rows) == (3 if short else 4),
            f'Mangalathu (2017) block {group} {notation} has {len(rows)} rows',
            rows,
        )
        check(
            rows == list(range(rows[0], rows[0] + len(rows))),
            f'Mangalathu (2017) block {group} {notation} is not contiguous',
            rows,
        )
        shapes = [table.loc[data_row, 'tokens'].get('shape') for data_row in rows]
        ok = (
            shapes[0] == 'Circ'
            and shapes[2] == 'Circ'
            and shapes[1] in ('Obl', 'Rect')
            and (len(shapes) == 3 or shapes[3] in ('Obl', 'Rect'))  # noqa: PLR2004
        )
        check(ok, f'unexpected shape sequence {shapes} in {group} {notation}', rows)
        for position, data_row in enumerate(rows):
            bent = 'TCB' if position < 2 else 'MCB'  # noqa: PLR2004
            table.loc[data_row, 'tokens']['bent'] = bent

    for source in ('MSJ17', 'JML19'):
        for data_row in table.index[
            (table['source'] == source) & table['bents'].eq('C2P')
        ]:
            table.loc[data_row, 'tokens']['bent'] = 'TCB'

    msj17_rows = list(table.index[table['source'] == 'MSJ17'])
    check(msj17_rows == [531, 800, 801], f'MSJ17 rows are {msj17_rows}', msj17_rows)
    check(table.loc[531, 'group'] == 'E1-S2-C2P-S', 'row 531 group', [531])
    for data_row, token in ((800, 'Span3'), (801, 'Span4')):
        check(
            table.loc[data_row, 'group'] == 'E1-S3P-C2P-S',
            'MSJ17 span-class group',
            [data_row],
        )
        table.loc[data_row, 'tokens']['span'] = token
    for source, token in (('MCP18', 'Span5'), ('JML19', 'Span3')):
        rows = table.index[table['source'] == source]
        bad = [
            data_row for data_row in rows if table.loc[data_row, 'spans'] != 'S3P'
        ]
        check(not bad, f'{source} rows outside S3P groups', bad)
        for data_row in rows:
            table.loc[data_row, 'tokens']['span'] = token


def thesis_class(model: pd.Series) -> str:
    """The Mangalathu (2017) class name of a row, e.g. T-E2-S34-C-S."""
    if model['source'] != 'M17':
        return ''
    bent = {'C1': 'S'}.get(model['bents']) or {'TCB': 'T', 'MCB': 'M'}[
        model['tokens']['bent']
    ]
    spans = {'S2': 'S22', 'S3P': 'S34'}[model['spans']]
    shape = {'Circ': 'C', 'Obl': 'O', 'Rect': 'R'}[model['tokens']['shape']]
    return f'{bent}-{model["era"]}-{spans}-{shape}-{model["abutment"]}'


def build_ids(table: pd.DataFrame) -> None:
    """Build the model IDs."""
    ids = []
    for row in table.index:
        model = table.loc[row]
        attrs = [model['tokens'][slot] for slot in SLOTS if slot in model['tokens']]
        groups = [model[col] for col, _ in GROUP_LEVELS]
        model_id = '.'.join(
            ['CRCB', *groups, model['notation'], *attrs, model['source']]
        )
        check(
            re.fullmatch(r'[A-Za-z0-9.]+', model_id) is not None,
            f'bad ID {model_id}',
            [row],
        )
        ids.append(model_id)
    table['ID'] = ids
    table['thesis_class'] = [thesis_class(table.loc[row]) for row in table.index]
    kept = table[table['omit'].eq('')]
    duplicated_ids = kept['ID'][kept['ID'].duplicated(keep=False)]
    check(
        duplicated_ids.empty,
        f'duplicate IDs {sorted(set(duplicated_ids))}',
        duplicated_ids.index,
    )
    check(len(kept) == 917, f'expected 917 IDs, found {len(kept)}')  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Metadata text
# ---------------------------------------------------------------------------


def description(model: pd.Series) -> str:
    """The model description."""
    templates = TEXT['description']
    parts = [phrases[model[col]] for col, phrases in GROUP_LEVELS]
    era, parts = parts[0], parts[1:]
    if model['era'] == 'E123':
        parts = parts[1:]
    attrs = [
        TEXT['attribute_phrases'][model['tokens'][slot]]
        for slot in SLOTS
        if slot in model['tokens']
    ]
    return templates['model'].format(
        component=COMPONENT_PHRASES[model['notation']],
        group=templates['group'].format(era=era, parts=join_phrases(parts)),
        attributes=templates['attributes'].format(attributes=', '.join(attrs))
        if attrs
        else '',
        citation=TEXT['citations'][model['source']],
    )


def provenance(data_row: int, model: pd.Series) -> str:
    """Comments paragraph 1."""
    templates = TEXT['comments']
    head = templates['provenance'].format(
        row=data_row,
        sheet_row=data_row + 2,
        group=model['group'],
        notation=squash(model['notation_raw']),
        notes=squash(model['notes_raw']),
    )
    tail = (
        templates['system']
        if model['notation'] == 'System'
        else templates['component']
    )
    return f'{head} {tail}'


def source_study(model: pd.Series) -> str:
    """Comments paragraph 2."""
    text = TEXT['source_study'][model['source']].format(**TEXT['shared_sentences'])
    if model['source'] == 'CPR22' and model['abutment'] == 'D':
        text = f'{text} {TEXT["cpr22_diaphragm"]}'
    return text


def state_notes(model: pd.Series) -> str:
    """Comments paragraph 3."""
    templates = TEXT['comments']
    if model['filled']:
        return templates['filled_states']
    populated = [not pd.isna(model[f'{state}_λ']) for state in STATES]
    na = [bool(model[f'na_{state}_λ']) for state in STATES]
    if model['notation'] in PRIMARY:
        check(
            not any(na[i] and any(populated[:i]) for i in range(1, 4)),
            'primary component with an N/A state above a populated one',
            [model.name],
        )
        if model['notation'] == 'System' and model['group'] == 'E123-S1-C0-D':
            check(
                populated == [True, True, False, False],
                'E123-S1-C0-D System row with unexpected states',
                [model.name],
            )
            return templates['e123_system']
        return ''
    if populated[0] and na[1]:
        return templates['na_moderate']
    return ''


def inherited_notes(model: pd.Series, twins: dict) -> list:
    """Comments paragraph 4: the corrections and inherited values."""
    templates = TEXT['comments']
    sentences = list(model['comments_extra'])
    if model.name in twins:
        key = 'twin_jml19' if model.name in JML19_TWIN_ROWS else 'twin'
        sentences.append(
            templates[key].format(partners=join_phrases(twins[model.name]))
        )
    if (
        model['notation'] == 'Unseating'
        and model['source'] == 'M17'
        and model['era'] == 'E3'
    ):
        sentences.append(templates['e3_unseating'])
    if model['notation'] == 'Seal':
        sentences.append(templates['seal'])
    if model['notation'] == 'Key':
        sentences.append(templates['key'])
    return sentences


def find_twins(table: pd.DataFrame) -> dict:
    """Kept rows with identical parameters within group, notation, source."""
    kept = table[table['omit'].eq('')]
    twins: dict = {}
    keys = ['group', 'notation', 'source', *PARAMS]
    for _, block in kept.groupby(keys, dropna=False, sort=False):
        for row in block.index:
            if len(block) > 1:
                twins[row] = sorted(block.loc[block.index != row, 'ID'])
    jml19 = {row for row in twins if table.loc[row, 'source'] == 'JML19'}
    check(
        jml19 == JML19_TWIN_ROWS,
        f'JML19 rows with identical parameters are {sorted(jml19)}',
        sorted(jml19 ^ JML19_TWIN_ROWS),
    )
    return twins


def check_e3_unseating(table: pd.DataFrame) -> None:
    """E3 Unseating curves of Mangalathu (2017) equal the E2 curves."""
    rows = table.index[
        (table['notation'] == 'Unseating')
        & (table['source'] == 'M17')
        & (table['era'] == 'E3')
    ]
    for row in rows:
        partner_id = 'CRCB.E2.' + table.loc[row, 'ID'].split('.', 2)[2]
        partner = table.index[table['ID'] == partner_id]
        check(
            len(partner) == 1 and same_parameters(table, row, partner[0]),
            'E3 Unseating curve differs from its E2 counterpart',
            [row, *partner],
        )


def threshold_set(model: pd.Series) -> tuple:
    """The (component, threshold set, variant) key of CAPACITIES for a row."""
    component, source, era = model['notation'], model['source'], model['era']
    if component == 'Column':
        return (component, 'drift' if source == 'MCP18' else '', era)
    if component == 'Unseating':
        if model['filled']:
            return (component, 'two-state', f'{source} {era}')
        check(source == 'M17', 'Unseating row without a capacity set', [model.name])
        return (component, 'E3', 'E2') if era == 'E3' else (component, '', era)
    if component == 'Approach':
        return (component, '', model['tokens']['approach'])
    if component == 'Bearing' and source == 'CPR22':
        return (component, 'CPR22', '')
    return (component, '', '')


def edp_row_key(key: tuple) -> tuple:
    """The EDP-file row behind a CAPACITIES key, or () where there is none.

    Rows of source/EDP_Fragility Database.csv are identified by (Fragility
    notation, EDP (unit), Notes), whitespace collapsed. The mapping is:

    - Column curvature ductility (M17, S17, MSJ17, JML19), variant E<n>: the
      Curvature ductility row whose Notes read "For Era <n> bridges".
    - Column drift (MCP18), variant E<n>: the Column drift (%) row whose Notes
      read "For E<n>-S3P-C1-D bridge group".
    - Unseating of Mangalathu (2017), variant E<n>: the Displacement (in.) row
      "For Era <n> bridges"; the E3 set quotes the Era 2 values, so its variant
      is E2 and it maps to the Era 2 row.
    - Two-state Unseating of MCP18 and JML19, variant "<source> E<n>": the same
      era rows, compared at extensive and complete only.
    - Two-state Unseating of CPR22: the row whose Notes list the bridge groups
      E1-S2-C2P-S, E1-S3P-C2P-S, E123-S1-C0-D, and E123-S1-C0-S.
    - Bearing of CPR22: the row "For E123-S1-C0-S bridge group".
    - Approach: the Settlement (in.) row "Approach with step offset" or
      "<length> ft approach with slope offset".
    - The generic set of the other components: the row of that component with
      empty Notes, EDP Rotation (rad) for FndRot and Displacement (in.) else.
    - System: no row.
    """
    component, threshold_name, variant = key
    displacement = 'Displacement (in.)'
    if component == 'System':
        return ()
    if component == 'Column':
        era = variant.removeprefix('E')
        if threshold_name == 'drift':
            return (
                component,
                'Column drift (%)',
                f'For E{era}-S3P-C1-D bridge group',
            )
        return (component, 'Curvature ductility', f'For Era {era} bridges')
    if component == 'Unseating':
        era = variant.split()[-1]
        if variant.startswith('CPR22'):
            groups = 'E1-S2-C2P-S, E1-S3P-C2P-S, E123-S1-C0-D, E123-S1-C0-S'
            return (component, displacement, f'Bridge groups: {groups}')
        return (component, displacement, f'For Era {era.removeprefix("E")} bridges')
    if component == 'Approach':
        if variant == 'Step':
            notes = 'Approach with step offset'
        else:
            notes = f'{variant.removeprefix("Slope")} ft approach with slope offset'
        return (component, 'Settlement (in.)', notes)
    if component == 'Bearing' and threshold_name == 'CPR22':
        return (component, displacement, 'For E123-S1-C0-S bridge group')
    edp = 'Rotation (rad)' if component == 'FndRot' else displacement
    return (component, edp, '')


def check_thresholds() -> None:
    """Every quoted capacity equals its value in the EDP file.

    Each CAPACITIES entry is matched to its EDP-file row by edp_row_key(), and
    every value that a description quotes (not None) must equal the file's
    median for that damage state. Values the descriptions do not quote are not
    compared.
    """
    raw = pd.read_csv(
        EDP_FILE, encoding='utf-8-sig', dtype=str, keep_default_na=False
    )
    edp_rows: dict = {}
    for _, edp_row in raw.iterrows():
        key = tuple(
            squash(edp_row[column])
            for column in ('Fragility notation', 'EDP (unit)', 'Notes')
        )
        check(key not in edp_rows, f'EDP file lists {key} twice')
        edp_rows[key] = [
            float(edp_row[f'{state}_median']) if edp_row[f'{state}_median'] else None
            for state in STATES
        ]
    for key, values in CAPACITIES.items():
        edp_key = edp_row_key(key)
        if not edp_key:
            check(
                all(value is None for value in values),
                f'thresholds {key} have no EDP-file row',
            )
            continue
        check(edp_key in edp_rows, f'no EDP-file row {edp_key} for thresholds {key}')
        published = edp_rows[edp_key]
        populated = [
            i for i, edp_value in enumerate(published) if edp_value is not None
        ]
        check(
            len(values) == populated[-1] + 1,
            f'thresholds {key} list {len(values)} values; EDP-file row {edp_key} '
            f'defines {populated[-1] + 1} states',
        )
        for state, value, edp_value in zip(STATES, values, published, strict=False):
            if value is None:
                continue
            check(
                edp_value is not None and float(value) == edp_value,
                f'{state} threshold {value} of {key} differs from the EDP file '
                f'value {edp_value} of {edp_key}',
            )


def limit_state_descriptions(model: pd.Series) -> dict:
    """LS<n> descriptions for one model; keys are the populated states."""
    templates = TEXT['limit_states']
    component, set_name, variant = key = threshold_set(model)
    values = CAPACITIES[key]
    words = {
        **templates['thresholds'].get(component, {}),
        **templates['thresholds'].get(f'{component} {set_name}', {}),
    }
    if component == 'Approach':
        shape = 'Step' if variant == 'Step' else 'Slope'
        length = variant.removeprefix('Slope')
        bases = [
            base.format(length=length)
            for base in templates['bases'][f'{component} {shape}']
        ]
    else:
        bases = templates['bases'][component]
    result = {}
    for i, state in enumerate(STATES):
        if pd.isna(model[f'{state}_λ']):
            continue
        check(
            i < len(bases), f'no description for {component} LS{i + 1}', [model.name]
        )
        if model['filled'] and i < 2:  # noqa: PLR2004
            state_text = templates['undefined'].format(state=state)
        elif values[i] is None:
            state_text = bases[i]
        else:
            note = words['note']
            if isinstance(note, list):
                check(
                    i < len(note), f'no note for {component} LS{i + 1}', [model.name]
                )
                note = note[i]
            state_text = templates['threshold'].format(
                base=bases[i],
                edp=words['edp'],
                value=values[i],
                unit=words['unit'],
                note=note.format(era=era_tag(model['era'])),
            )
        result[f'LS{i + 1}'] = {f'DS{i + 1}': {'Description': sentence(state_text)}}
    return result


def omitted_sentence(table: pd.DataFrame, omitted: pd.DataFrame) -> str:
    """Summarize the all-N/A rows by component for the dataset description."""
    templates = TEXT['omitted']
    totals = table.groupby(['notation', 'group']).size()
    parts = []
    for notation, rows in omitted.groupby('notation'):
        counts = rows['group'].value_counts().sort_index()
        occurring = totals[notation]
        if len(counts) == len(occurring):
            key = 'all_groups_one' if len(occurring) == 1 else 'all_groups'
            where = templates[key].format(n=len(occurring))
        else:
            where = templates['some_groups'].format(
                groups=join_phrases(list(counts.index))
            )
        exceptions = [
            templates['exception'].format(
                kept=occurring[group] - count, total=occurring[group], group=group
            )
            for group, count in counts.items()
            if count < occurring[group]
        ]
        parts.append(
            templates['component'].format(
                count=len(rows),
                component=COMPONENT_PHRASES[notation].lower(),
                notation=notation,
                where=where,
                exceptions=templates['exceptions'].format(
                    items=join_phrases(exceptions)
                )
                if exceptions
                else '',
            )
        )
    return '; '.join(parts)


def component_groups(ids: list, depth: int) -> dict:
    """Nested ComponentGroups; ``depth`` 3 stops at span, 5 at abutment."""
    present = {tuple(model_id.split('.')[1:depth]) for model_id in ids}

    def label(prefix: tuple) -> str:
        code = 'CRCB.' + '.'.join(prefix)
        phrases = GROUP_LEVELS[len(prefix) - 1][1]
        return f'{code} - {without_article(phrases[prefix[-1]])}'

    def build(prefix: tuple):
        level = len(prefix)
        children = [
            (*prefix, code)
            for code in GROUP_LEVELS[level][1]
            if any(path[: level + 1] == (*prefix, code) for path in present)
        ]
        if level + 1 == depth - 1:
            return [label(child) for child in children]
        return {label(child): build(child) for child in children}

    return {TEXT['group_root']: build(())}


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

CSV_HEADER = [
    'ID',
    'Incomplete',
    'Demand-Type',
    'Demand-Unit',
    'Demand-Offset',
    'Demand-Directional',
] + [
    f'LS{i}-{field}'
    for i in range(1, 5)
    for field in ('Family', 'Theta_0', 'Theta_1', 'DamageStateWeights')
]


def write_csv(path: Path, rows: pd.DataFrame) -> None:
    """Write the fragility table."""
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(CSV_HEADER)
        for _, model in rows.sort_values('ID').iterrows():
            line = [model['ID'], 0, 'Spectral Acceleration|1.0', 'g', 0, 0]
            seen_empty = False
            for state in STATES:
                median, beta = model[f'{state}_λ'], model[f'{state}_ζ']
                if pd.isna(median):
                    check(pd.isna(beta), 'dispersion without a median', [model.name])
                    seen_empty = True
                    line += ['', '', '', '']
                    continue
                check(
                    not seen_empty,
                    'populated state above an empty one',
                    [model.name],
                )
                check(not pd.isna(beta) and beta > 0, 'bad dispersion', [model.name])
                line += ['lognormal', fmt(median), fmt(beta), '']
            writer.writerow(line)


def write_json(path: Path, general: dict, references: dict, models: dict) -> None:
    """Write the fragility metadata."""
    metadata = {'_GeneralInformation': general, 'References': references}
    for model_id in sorted(models):
        metadata[model_id] = models[model_id]
    with path.open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write('\n')


def write_crosswalk(table: pd.DataFrame) -> None:
    """Write id_crosswalk.csv: one row per model, then the omitted rows."""
    header = [
        'ID',
        'dataset',
        'source_file',
        'source_version',
        'data_row',
        'spreadsheet_row',
        'bridge_group',
        'fragility_notation',
        'notes_raw',
        'reference_raw',
        'source_token',
        'thesis_class',
        'correction',
    ]
    kept = table[table['omit'].eq('')].sort_values('ID')
    omitted = table[table['omit'].ne('')].sort_index()
    with CROSSWALK_FILE.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(header)
        for frame, is_kept in ((kept, True), (omitted, False)):
            for data_row, model in frame.iterrows():
                writer.writerow(
                    [
                        model['ID'] if is_kept else '',
                        model['dataset'] if is_kept else 'omitted',
                        SOURCE_FILE.name,
                        SOURCE_VERSION,
                        data_row,
                        data_row + 2,
                        model['group'],
                        model['notation_raw'],
                        model['notes_raw'],
                        model['reference_raw'],
                        model['source'],
                        model['thesis_class'],
                        ';'.join(model['corrections']) if is_kept else model['omit'],
                    ]
                )


def main() -> None:
    """Run the pipeline and write both datasets and the crosswalk."""
    table = read_source()
    drop_duplicates(table)
    omitted = drop_all_na(table)
    correct_labels(table)
    assign_tokens(table)
    recompute_single_span_diaphragm(table)

    check_recomputed_slight_medians(table)
    fix_sxr22_dispersion(table)
    fill_lower_states(table)
    assign_bents_and_spans(table)
    build_ids(table)
    check_e3_unseating(table)
    check_thresholds()

    table['dataset'] = np.where(
        table['notation'] == 'System', 'portfolio', 'component'
    )
    kept = table[table['omit'].eq('')]
    counts = kept['dataset'].value_counts().to_dict()
    check(
        counts == {'component': 801, 'portfolio': 116},
        f'unexpected dataset sizes {counts}',
    )

    twins = find_twins(table)
    references = json.loads((HERE / 'references.json').read_text(encoding='utf-8'))
    general = json.loads(
        (HERE / 'general_information.json').read_text(encoding='utf-8')
    )
    s17_ids = sorted(kept.loc[kept['source'] == 'S17', 'ID'])
    all_system = all('.System.' in model_id for model_id in s17_ids)
    check(
        len(s17_ids) == 6 and all_system,  # noqa: PLR2004
        f'expected six Soleimani (2017) System models, found {s17_ids}',
    )
    general['component']['Description'] = fill(
        general['component']['Description'],
        omitted=omitted_sentence(table, omitted),
    )

    for dataset, depth in (('portfolio', 3), ('component', 5)):
        rows = kept[kept['dataset'] == dataset]
        models = {}
        for data_row, model in rows.iterrows():
            paragraphs = [
                provenance(data_row, model),
                source_study(model),
                state_notes(model),
                ' '.join(inherited_notes(model, twins)),
            ]
            models[model['ID']] = {
                'Description': description(model),
                'Comments': '\n\n'.join(
                    paragraph for paragraph in paragraphs if paragraph
                ),
                'SuggestedComponentBlockSize': '1 EA',
                'RoundUpToIntegerQuantity': 'True',
                'Reference': [REFERENCE_KEYS[model['source']], 'chen2025'],
                'LimitStates': limit_state_descriptions(model),
            }
            for key in models[model['ID']]['Reference']:
                check(key in references, f'missing reference {key}', [data_row])
        general_info = dict(general[dataset])
        general_info['ComponentGroups'] = component_groups(sorted(models), depth)
        out_dir = OUTPUT[dataset]
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out_dir / 'fragility.csv', rows)
        write_json(out_dir / 'fragility.json', general_info, references, models)
        print(f'{dataset}: {len(models)} models -> {out_dir}')  # noqa: T201

    write_crosswalk(table)
    print(  # noqa: T201
        f'crosswalk: {len(kept)} models, {int(table["omit"].ne("").sum())} omitted '
        f'rows -> {CROSSWALK_FILE}'
    )
    print(f'identical-parameter models: {len(twins)}')  # noqa: T201


if __name__ == '__main__':
    main()
