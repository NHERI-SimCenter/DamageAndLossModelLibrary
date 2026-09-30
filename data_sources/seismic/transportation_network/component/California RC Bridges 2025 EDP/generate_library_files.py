"""Generate the California RC Bridges 2025 EDP fragility dataset.

Reads ``models.csv``, the model table derived from the EDP capacity table of
the Chen et al. (2025) database (DesignSafe PRJ-5910, version 1,
``source/EDP_Fragility Database.csv``), and writes the dataset
``component/California RC Bridges 2025 EDP`` under
``src/dlml/data/seismic/transportation_network``. Each row of ``models.csv``
is one model: a lognormal fragility function of the response of one bridge
component (a peak value, or the permanent settlement for approaches), with
the median capacity and dispersion of the source. ``models.csv`` holds the
published rows with a ``Model ID`` column added; one published row appears
twice, once per model it yields.

It also writes ``id_crosswalk.csv`` beside this script, which maps every model
ID to its row in ``models.csv`` and in the published file and names the change
made to its values.

A model ID has the form ``CRCB.<component>[.<qualifier>].<scope>.<source>``,
for example ``CRCB.Column.CD.E1.M17``. The demand type follows from the
component and qualifier, the ``Reference`` from the source token, and the
metadata text from the ID segments: ``metadata_text.json`` beside this script
holds its entries keyed by fragments of the ID, and ``source/README.md``
describes how each key resolves. The dataset description
is read from ``general_information.json``. The neighbor folder ``California
RC Bridges 2025``, which generates the Sa(1.0 s)-based datasets, supplies the
citations (``references.json``) and the damage-state wording (the
``limit_states`` block of its ``metadata_text.json``). The demand types come
from ``src/dlml/vocabulary.py``. Every change to the published values is
recorded in ``source/README.md``.

Run from this folder with the repository environment::

    uv run python generate_library_files.py

Row numbers are 0-based data rows; the spreadsheet row is the data row plus 2.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import NamedTuple

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
SOURCE_FILE = HERE / 'source' / 'EDP_Fragility Database.csv'
MODELS_FILE = HERE / 'models.csv'
NEIGHBOR_FOLDER = HERE.parent / 'California RC Bridges 2025'
VOCABULARY_FILE = REPO_ROOT / 'src' / 'dlml' / 'vocabulary.py'
NETWORK_DATA = (
    REPO_ROOT / 'src' / 'dlml' / 'data' / 'seismic' / 'transportation_network'
)
OUTPUT = NETWORK_DATA / 'component' / 'California RC Bridges 2025 EDP'
CROSSWALK_FILE = HERE / 'id_crosswalk.csv'

STATES = ['Slight', 'Moderate', 'Extensive', 'Complete']
MEDIANS = [f'{state}_median' for state in STATES]
# The columns a models.csv row shares with the published row it comes from.
VALUE_COLUMNS = ['Fragility notation', 'EDP (unit)', *MEDIANS, 'Dispersion']

# Source token (the last ID segment) -> the key of the study in the
# neighbor's references.json. Every model also cites the database.
SOURCE_KEYS = {
    'M17': 'mangalathu2017',
    'MCP18': 'mangalathu2018',
    'MSJ17': 'mangalathu2017b',
    'CPR22': 'cavalcante2022',
    'SXR22': 'shao2022',
}
DATABASE_KEY = 'chen2025'

# Source token -> the study as the References of models.csv name it, before
# an "except for" clause. M17 also stands for the rows credited to all six
# multi-span studies of Chen et al. (2025, Table 3), whose capacities come
# from Mangalathu (2017) Table 6.13.
SOURCE_STUDIES = {
    'M17': 'All six listed in Table 3 of the paper',
    'MCP18': 'Mangalathu et al. (2018)',
    'MSJ17': 'Mangalathu, Soleimani, et al. (2017)',
    'CPR22': 'Cavalcante et al. (2022)',
    'SXR22': 'Shao et al. (2022)',
}

# Model kind with a qualifier -> the `EDP (unit)` of its rows. Only Column
# models carry a qualifier, and every Column model carries one.
QUALIFIED_EDPS = {
    'Column.CD': 'Curvature ductility',
    'Column.DR': 'Column drift (%)',
}

# Demand type per model kind, the component and qualifier of the model ID.
DEMAND_TYPES = {
    'Column.CD': 'Peak Column Curvature Ductility',
    'Column.DR': 'Peak Column Drift Ratio',
    'Unseating': 'Peak Joint Opening',
    'AbAct': 'Peak Abutment Active Displacement',
    'AbPass': 'Peak Abutment Passive Displacement',
    'AbTran': 'Peak Abutment Transverse Displacement',
    'Bearing': 'Peak Bearing Deformation',
    'DeckMax': 'Peak Deck Displacement',
    'FndRot': 'Peak Foundation Rotation',
    'FndTran': 'Peak Foundation Translation',
    'Key': 'Peak Shear Key Deformation',
    'Seal': 'Peak Joint Opening',
    'Approach': 'Permanent Approach Settlement',
}

# Unit in parentheses at the end of `EDP (unit)` -> Demand-Unit ('' when the
# EDP names no unit). Percentages are stored as ratios.
DEMAND_UNITS = {'in.': 'inch', 'rad': 'rad', '%': 'unitless', '': 'unitless'}

# Demand-Unit -> the UnitType its demand type must have in the vocabulary.
UNIT_TYPES = {'unitless': 'unitless', 'inch': 'displacement', 'rad': 'rotation'}

# A bridge group code in Notes, such as E1-S3P-C1-D.
GROUP_CODE = r'E\d+(?:-[A-Z0-9]+){3}'

# A placeholder in a text-file template, such as {era}.
PLACEHOLDER = r'\{(\w*)\}'

# Populated states of a row that defines only the extensive and complete
# states; its slight and moderate medians are set equal to extensive.
FILLED_LAYOUT = [False, False, True, True]

PERCENT_CHANGE = 'column drift converted from percent to ratio'
FILLED_CHANGE = 'slight and moderate set equal to extensive'


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


def sentence(text: str) -> str:
    """End ``text`` with a period unless it already ends with one."""
    return text if text.endswith('.') else f'{text}.'


def join_and(parts: list[str]) -> str:
    """Join with 'and', with commas between three or more parts."""
    if len(parts) < 3:
        return ' and '.join(parts)
    return ', '.join(parts[:-1]) + f', and {parts[-1]}'


def fill(template: str, values: dict, where: str, *, exact: bool = False) -> str:
    """Fill the placeholders of a text-file template from ``values``.

    Each placeholder may occur once and must have a value; with ``exact``,
    every value must also be used. Otherwise the run stops with a message that
    starts with ``where``, which names the model and the text key.
    """
    names = re.findall(PLACEHOLDER, template)
    check(len(names) == len(set(names)), f'{where}: a placeholder occurs twice')
    missing = [f'{{{name}}}' for name in names if name not in values]
    check(not missing, f'{where}: no value for {", ".join(missing)}')
    unused = sorted(set(values) - set(names))
    check(not exact or not unused, f'{where}: the template does not use {unused}')
    result = re.sub(PLACEHOLDER, lambda match: values[match[1]], template)
    check('{' not in result and '}' not in result, f'{where}: unresolved brace')
    return result


# ---------------------------------------------------------------------------
# Inputs and checks
# ---------------------------------------------------------------------------


class Ident(NamedTuple):
    """The segments of a model ID, its models.csv row, and both for messages.

    ``kind`` is the component with its qualifier: 'Column.CD', 'Seal'.
    """

    model_id: str
    row: int
    component: str
    kind: str
    scope: str
    source: str
    where: str


def parse_id(model_id: str, row: int) -> Ident:
    """Split ``CRCB.<component>[.<qualifier>].<scope>.<source>`` into segments."""
    parts = model_id.split('.')
    where = f'models.csv row {row}, {model_id}'
    check(
        parts[0] == 'CRCB' and len(parts) in {4, 5},
        f'{where}: not CRCB.<component>[.<qualifier>].<scope>.<source>',
    )
    kind = '.'.join(parts[1:-2])
    return Ident(model_id, row, parts[1], kind, parts[-2], parts[-1], where)


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file with every cell as a string, empty cells as ''."""
    return pd.read_csv(path, encoding='utf-8-sig', dtype=str, keep_default_na=False)


def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding='utf-8'))


def read_edp_types() -> dict:
    """Load EDP_TYPES from src/dlml/vocabulary.py without importing dlml."""
    spec = importlib.util.spec_from_file_location('vocabulary', VOCABULARY_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EDP_TYPES


def items(text: str) -> set[str]:
    """Return the items of a Notes or References string, split at , ; and :."""
    return {item.strip() for item in re.split(r'[,;:]', text)} - {''}


def check_models_file(
    models: pd.DataFrame, idents: list[Ident], published: pd.DataFrame
) -> list[int]:
    """Stop unless models.csv is the published table with one row given twice.

    The model IDs are unique. Each row's notation is the component of its ID;
    a Column ID carries the qualifier of its EDP (QUALIFIED_EDPS), and no
    other ID carries one; the component and qualifier name a demand type; and
    the source token names the study its References credit, read without an
    'except for' clause (SOURCE_STUDIES). Every row has the value columns of a
    published row; without the repeated row, the rows are the published rows
    in order. For every published row, the Notes and References items of its
    copies together are the items of the published strings, and each copy of
    the repeated row names only some of them. Returns, per models.csv row, the
    published row it copies.
    """
    check(
        list(models.columns) == ['Model ID', *published.columns],
        f'models.csv columns {list(models.columns)}',
    )
    duplicates = models['Model ID'][models['Model ID'].duplicated()]
    check(duplicates.empty, f'repeated model IDs {sorted(duplicates)}')
    qualified = {kind.split('.')[0] for kind in QUALIFIED_EDPS}
    for ident, (_, model) in zip(idents, models.iterrows(), strict=True):
        where = ident.where
        check(
            model['Fragility notation'] == ident.component,
            f'{where}: the notation {model["Fragility notation"]} is not the component',
        )
        edp = squash(model['EDP (unit)'])
        if ident.kind != ident.component or ident.component in qualified:
            check(
                QUALIFIED_EDPS.get(ident.kind) == edp,
                f'{where}: the qualifier does not fit the EDP "{edp}"',
            )
        check(
            ident.kind in DEMAND_TYPES, f'{where}: no demand type for {ident.kind}'
        )
        credited = squash(model['References']).split(', except for ')[0]
        check(
            SOURCE_STUDIES.get(ident.source) == credited,
            f'{where}: the source token does not match the References "{credited}"',
        )
    index = {
        tuple(values): row
        for row, values in enumerate(published[VALUE_COLUMNS].to_numpy())
    }
    rows = [index.get(tuple(values)) for values in models[VALUE_COLUMNS].to_numpy()]
    check(
        len(rows) == len(published) + 1
        and list(dict.fromkeys(rows)) == list(range(len(published))),
        'the value columns of models.csv are not the published rows with one repeat',
    )
    for row, record in published.iterrows():
        copies = models[[match == row for match in rows]]
        for column in ('Notes', 'References'):
            expected = items(record[column])
            named = [items(text) for text in copies[column]]
            check(
                set().union(*named) == expected
                and (len(named) == 1 or all(n < expected for n in named)),
                f'{column} of models.csv do not match the published row',
                [row],
            )
    return rows


def check_states(model: pd.Series, ident: Ident) -> None:
    """Stop if a populated state lies above an empty one, except in a filled row.

    A filled row defines only the extensive and complete states; it must be
    an unseating model.
    """
    if is_filled(model):
        check(
            ident.kind == 'Unseating',
            f'{ident.where}: leaves slight and moderate empty, not unseating',
        )
    else:
        populated = [bool(value) for value in published_medians(model)]
        check(
            populated == sorted(populated, reverse=True),
            f'{ident.where}: a populated state lies above an empty one',
        )


def check_unit(model: pd.Series, ident: Ident, edp_types: dict) -> None:
    """Stop unless the demand unit fits the unit type of the demand type."""
    demand = DEMAND_TYPES[ident.kind]
    check(
        edp_types[demand]['UnitType'] == UNIT_TYPES[demand_unit(model)],
        f'{ident.where}: unit {demand_unit(model)} does not fit {demand}',
    )


def check_text_used(text: dict, used: dict) -> None:
    """Stop if a scopes, capacity or use key was never resolved (a typo guard)."""
    use = text['use']
    for name, table in [
        ('scopes', text['scopes']),
        ('capacity', text['capacity']),
        ('use.sources', use['sources']),
        ('use.components', use['components']),
    ]:
        unused = sorted(set(table) - used[name])
        check(
            not unused,
            f'metadata_text.json {name} keys that no model uses: {unused}',
        )


# ---------------------------------------------------------------------------
# Model values
# ---------------------------------------------------------------------------


def unit_label(model: pd.Series) -> str:
    """Return the unit in parentheses of `EDP (unit)`: 'in.', '%', or ''."""
    match = re.search(r'\(([^()]*)\)$', squash(model['EDP (unit)']))
    return match.group(1) if match else ''


def demand_unit(model: pd.Series) -> str:
    """Return the Demand-Unit of the model."""
    return DEMAND_UNITS[unit_label(model)]


def is_percent(model: pd.Series) -> bool:
    """Return whether the row gives its medians in percent."""
    return unit_label(model) == '%'


def published_medians(model: pd.Series) -> list[str]:
    """Return the median of each state as printed, '' where empty."""
    return [model[column] for column in MEDIANS]


def is_filled(model: pd.Series) -> bool:
    """Return whether the row defines only the extensive and complete states."""
    return [bool(value) for value in published_medians(model)] == FILLED_LAYOUT


def ratio_from_percent(text: str) -> str:
    """Convert a percentage to a ratio without trailing zeros: '0.7' -> '0.007'."""
    return format((Decimal(text) / 100).normalize(), 'f')


def stored_medians(model: pd.Series) -> list[str]:
    """Return the stored median of each state, '' where undefined.

    Values are the strings printed in the source, except percentages, which
    are converted to ratios, and the slight and moderate medians of a filled
    row, which are set equal to extensive.
    """
    medians = published_medians(model)
    if is_filled(model):
        medians[0] = medians[1] = medians[2]
    if is_percent(model):
        medians = [ratio_from_percent(value) if value else '' for value in medians]
    return medians


def crosswalk_change(model: pd.Series) -> str:
    """Return the `change` column of the crosswalk."""
    changes = []
    if is_percent(model):
        changes.append(PERCENT_CHANGE)
    if is_filled(model):
        changes.append(FILLED_CHANGE)
    return '; '.join(changes)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def conversions(model: pd.Series) -> str:
    """Return the percent-to-ratio conversions the Column.DR capacity quotes.

    For example '0.7% as 0.007, 1.5% as 0.015, 2.5% as 0.025, and 5% as
    0.05'. The drift rows define all four states.
    """
    pairs = zip(published_medians(model), stored_medians(model), strict=True)
    return join_and([f'{percent}% as {ratio}' for percent, ratio in pairs if ratio])


def placeholders(model: pd.Series, ident: Ident, text: dict, used: dict) -> dict:
    """Return the values the model's text may use, derived from its ID and Notes.

    ``era`` ('Era 1') comes from an era scope; ``length`` ('10') from a slab
    scope, for the neighbor's slab sentences; ``group`` ('group E1-S3P-C1-D',
    or 'groups ... and ...') from the bridge group codes in the Notes;
    ``conversions`` from a percent row; and ``scope`` is the scope phrase, the
    ``scopes`` entry ``<scope>.<source>`` if there is one, else ``<scope>``.
    A value that does not apply is absent, and a template that uses it stops
    the run.
    """
    values = {}
    if era := re.fullmatch(r'E(\d)', ident.scope):
        values['era'] = f'Era {era[1]}'
    if length := re.fullmatch(r'Slope(\d+)', ident.scope):
        values['length'] = length[1]
    if groups := re.findall(GROUP_CODE, model['Notes']):
        noun = 'group' if len(groups) == 1 else 'groups'
        values['group'] = f'{noun} {join_and(groups)}'
    if is_percent(model):
        values['conversions'] = conversions(model)
    keys = [f'{ident.scope}.{ident.source}', ident.scope]
    found = [key for key in keys if key in text['scopes']]
    check(bool(found), f'{ident.where}: no scopes entry {keys[0]} or {keys[1]}')
    used['scopes'].add(found[0])
    where = f'{ident.where}, scopes {found[0]}'
    values['scope'] = fill(text['scopes'][found[0]], values, where)
    return values


def provenance(data_row: int, split: bool, record: pd.Series, text: dict) -> str:  # noqa: FBT001
    """Return Comments paragraph 1: the published row the model comes from.

    A model whose published row gives two models (``split``) adds the
    ``row_split`` sentence.
    """
    comments = text['comments']
    notes = squash(record['Notes'])
    source_note = ''
    if notes:
        source_note = fill(comments['source_note'], {'notes': notes}, 'source_note')
    paragraph = fill(
        comments['provenance'],
        {
            'row': str(data_row),
            'sheet_row': str(data_row + 2),
            'notation': squash(record['Fragility notation']),
            'edp': squash(record['EDP (unit)']),
            'source_note': source_note,
            'references': squash(record['References']),
        },
        'comments provenance',
        exact=True,
    )
    if split:
        paragraph += ' ' + comments['row_split']
    return paragraph


def comments_text(
    model: pd.Series,
    ident: Ident,
    record: pd.Series,
    values: dict,
    text: dict,
    edp_types: dict,
    used: dict,
) -> str:
    """Return the Comments: provenance, use, capacity source, filled states.

    The capacity paragraph joins the ``capacity`` entries ``<kind>``,
    ``<kind>.<source>`` and ``<kind>.<scope>.<source>`` that exist, in this
    order, and the entry ``MultiSpan`` for a MultiSpan model. The run stops if
    none of the first three exists.
    """
    use = text['use']
    words = dict(use['default'])
    for table, key in (('sources', ident.source), ('components', ident.component)):
        if key in use[table]:
            used[f'use.{table}'].add(key)
            words.update(use[table][key])
    acronym = edp_types[DEMAND_TYPES[ident.kind]]['Acronym']
    use_paragraph = fill(
        use['template'],
        {'acronym': acronym, **words},
        f'{ident.where}, use',
        exact=True,
    )
    keys = [ident.kind, f'{ident.kind}.{ident.source}']
    keys.append(f'{ident.kind}.{ident.scope}.{ident.source}')
    names = [key for key in keys if key in text['capacity']]
    check(bool(names), f'{ident.where}: no capacity entry among {keys}')
    if ident.scope == 'MultiSpan' and 'MultiSpan' in text['capacity']:
        names.append('MultiSpan')
    used['capacity'].update(names)
    capacity = ' '.join(
        fill(text['capacity'][name], values, f'{ident.where}, capacity {name}')
        for name in names
    )
    paragraphs = [
        provenance(model['data_row'], model['split'], record, text),
        use_paragraph,
        capacity,
    ]
    if is_filled(model):
        paragraphs.append(text['comments']['filled_states'])
    return '\n\n'.join(paragraphs)


def limit_states(
    model: pd.Series, ident: Ident, values: dict, text: dict, neighbor: dict
) -> dict:
    """Return the LS<n> descriptions of the states with a stored median.

    Each description is the neighbor's ``threshold`` template filled with the
    neighbor's base sentences and threshold words. The base sentences are
    those of the component, or of the scope where ``limit_states.bases`` of
    this folder's text names one; the threshold words are updated from its
    ``limit_states.thresholds``, by kind and then by ``<kind>.<scope>.<source>``,
    and the approach bases get its labels. A filled state gets the neighbor's
    ``undefined`` template.
    """
    own, where = text['limit_states'], ident.where
    name = own['bases'].get(ident.scope, ident.component)
    bases = [
        fill(base, values, f'{where}, neighbor bases {name}')
        for base in neighbor['bases'][name]
    ]
    labels = own['labels'].get(ident.component)
    if labels:
        for base in bases:
            check(': ' in base, f'neighbor base without ": " to relabel: {base}')
        bases = [
            f'{label}: {base.partition(": ")[2]}'
            for label, base in zip(labels, bases, strict=True)
        ]
    words = dict(neighbor['thresholds'][ident.component])
    for key in (ident.kind, f'{ident.kind}.{ident.scope}.{ident.source}'):
        words.update(own['thresholds'].get(key, {}))
    published, stored = published_medians(model), stored_medians(model)
    result = {}
    for i, state in enumerate(STATES):
        if not stored[i]:
            continue
        if published[i]:
            note = words['note']
            if isinstance(note, list):
                note = note[i]
            note = fill(note, {**values, 'percent': published[i]}, f'{where}, note')
            description = fill(
                neighbor['threshold'],
                {
                    'base': bases[i],
                    'edp': words['edp'],
                    'value': stored[i],
                    'unit': words['unit'],
                    'note': note,
                },
                f'{where}, neighbor threshold',
                exact=True,
            )
        else:
            description = fill(
                neighbor['undefined'],
                {'state': state},
                f'{where}, neighbor undefined',
                exact=True,
            )
        result[f'LS{i + 1}'] = {f'DS{i + 1}': {'Description': sentence(description)}}
    return result


def model_metadata(
    model: pd.Series,
    ident: Ident,
    record: pd.Series,
    text: dict,
    neighbor: dict,
    edp_types: dict,
    used: dict,
) -> dict:
    """Return the fragility.json entry of one model; record the text keys used."""
    values = placeholders(model, ident, text, used)
    return {
        'Description': fill(
            text['description'],
            {
                'component': text['component_phrases'][ident.component],
                'demand': DEMAND_TYPES[ident.kind].lower(),
                'scope': values['scope'],
            },
            f'{ident.where}, description',
            exact=True,
        ),
        'Comments': comments_text(
            model, ident, record, values, text, edp_types, used
        ),
        'SuggestedComponentBlockSize': '1 EA',
        'RoundUpToIntegerQuantity': 'True',
        'Reference': [SOURCE_KEYS[ident.source], DATABASE_KEY],
        'LimitStates': limit_states(model, ident, values, text, neighbor),
    }


def general_information(general: dict, text: dict) -> dict:
    """Return _GeneralInformation with the era years and ComponentGroups."""
    general = dict(general)
    general['Description'] = fill(
        general['Description'],
        {'eras': text['eras']},
        'general_information.json Description',
        exact=True,
    )
    general['ComponentGroups'] = {
        text['group_root']: [
            f'CRCB.{name} - {label}'
            for name, label in text['component_groups'].items()
        ]
    }
    return general


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
    for field in ('Family', 'Theta_0', 'Theta_1')
]


def write_csv(path: Path, pairs: list) -> None:
    """Write the fragility table, one row per model sorted by ID."""
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(CSV_HEADER)
        for ident, model in sorted(pairs, key=lambda pair: pair[0].model_id):
            demand = DEMAND_TYPES[ident.kind]
            line = [ident.model_id, 0, demand, demand_unit(model), 0, 1]
            for median in stored_medians(model):
                if median:
                    line += ['lognormal', median, model['Dispersion']]
                else:
                    line += ['', '', '']
            writer.writerow(line)


def write_json(path: Path, general: dict, references: dict, metadata: dict) -> None:
    """Write the fragility metadata, models sorted by ID."""
    document = {'_GeneralInformation': general, 'References': references}
    for model_id in sorted(metadata):
        document[model_id] = metadata[model_id]
    with path.open('w', encoding='utf-8') as f:
        json.dump(document, f, indent=2, ensure_ascii=False)
        f.write('\n')


def write_crosswalk(path: Path, pairs: list, published: pd.DataFrame) -> None:
    """Write id_crosswalk.csv, one row per model sorted by ID.

    ``models_row`` is the row in models.csv; ``data_row``, ``sheet_row`` and
    the raw strings are those of the published file.
    """
    header = ['ID', 'models_row', 'data_row', 'sheet_row', 'notation']
    header += ['edp_unit', 'notes', 'references', 'change']
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(header)
        for ident, model in sorted(pairs, key=lambda pair: pair[0].model_id):
            record = published.loc[model['data_row']]
            writer.writerow(
                [
                    ident.model_id,
                    ident.row,
                    model['data_row'],
                    model['data_row'] + 2,
                    record['Fragility notation'],
                    record['EDP (unit)'],
                    record['Notes'],
                    record['References'],
                    crosswalk_change(model),
                ]
            )


def main() -> None:
    """Read and check the inputs, then write the dataset and the crosswalk."""
    models = read_csv(MODELS_FILE)
    published = read_csv(SOURCE_FILE)
    text = read_json(HERE / 'metadata_text.json')
    neighbor = read_json(NEIGHBOR_FOLDER / 'metadata_text.json')['limit_states']
    edp_types = read_edp_types()
    references = read_json(NEIGHBOR_FOLDER / 'references.json')

    idents = [
        parse_id(model_id, row) for row, model_id in enumerate(models['Model ID'])
    ]
    models['data_row'] = check_models_file(models, idents, published)
    models['split'] = models['data_row'].duplicated(keep=False)
    pairs = list(zip(idents, (model for _, model in models.iterrows()), strict=True))
    for ident, model in pairs:
        check_states(model, ident)
        check_unit(model, ident, edp_types)

    used = defaultdict(set)
    metadata = {
        ident.model_id: model_metadata(
            model,
            ident,
            published.loc[model['data_row']],
            text,
            neighbor,
            edp_types,
            used,
        )
        for ident, model in pairs
    }
    check_text_used(text, used)
    general = general_information(read_json(HERE / 'general_information.json'), text)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / 'fragility.csv', pairs)
    write_json(OUTPUT / 'fragility.json', general, references, metadata)
    write_crosswalk(CROSSWALK_FILE, pairs, published)
    print(f'{len(models)} models -> {OUTPUT}')  # noqa: T201
    print(f'crosswalk: {len(models)} rows -> {CROSSWALK_FILE}')  # noqa: T201


if __name__ == '__main__':
    main()
