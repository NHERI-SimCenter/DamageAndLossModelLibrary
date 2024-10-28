"""Generates Hazus Flood loss functions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pelicun import base

idx = base.idx


def remove_repeated_chars(s):
    """
    Remove repeated characters.

    Removes all repeated instances of a character in a string with a
    single instance of that character, unless it is [A-Za-z].

    Parameters
    ----------
    s : str
        The input string.

    Returns
    -------
    str
        The string with repeated instances of characters replaced
        by a single instance.

    """
    if not s:
        return ''

    result = [s[0]]  # Initialize result with the first character

    for char in s[1:]:
        if char.isalpha() or char != result[-1]:
            result.append(char)

    return ''.join(result)


def create_Hazus_Flood_repair_db(  # noqa: C901
    source_file_dir: str = (
        'flood/building/portfolio/Hazus v6.1/data_sources/input_files'
    ),
    target_data_file: str = 'flood/building/portfolio/Hazus v6.1/loss_repair.csv',
):
    """
    Create HAZUS Flood metadata.

    Create a database metadata file for the HAZUS Flood
    loss functions.

    Parameters
    ----------
    source_file_dir: string
        Path to the directory containing Hazus Flood loss function
        data files.
    target_data_file: string
        Path where the loss function data should be saved. A CSV file
        is expected.

    """
    source_data = {}
    for subassembly_type in ('structural', 'inventory', 'contents'):
        source_file = (
            f'{source_file_dir}/HazusFloodDamageFunctions_'
            f'Hazus61_{subassembly_type}.csv'
        )
        source_data[subassembly_type] = pd.read_csv(source_file)

    # We have a dedicated column for `subassembly`, so we don't need
    # special names for the function ID for each subassembly set.
    source_data['structural'] = source_data['structural'].rename(
        columns={'BldgDmgFnID': 'FnID'}
    )
    source_data['inventory'] = source_data['inventory'].rename(
        columns={'InvDmgFnId': 'FnID'}
    )
    source_data['contents'] = source_data['contents'].rename(
        columns={'ContDmgFnId': 'FnID'}
    )

    # Merge the three subassembly datasets
    df = pd.concat(source_data.values(), keys=source_data.keys(), axis=0)
    df.index.names = ['subassembly', 'index']

    # Columns defining the loss for each inundation height
    ft_cols = []
    for col in df.columns:
        if col.startswith('ft'):
            ft_cols.append(col)
    ft_values_list = []
    for x in ft_cols:
        if 'm' in x:
            ft_values_list.append(-float(x.replace('m', '').replace('ft', '')))
        else:
            ft_values_list.append(float(x.replace('ft', '')))
    ft_values = np.array(ft_values_list)

    unique_sources = df['Source'].unique()
    source_map = {}
    for source in unique_sources:
        source_value = (
            source.replace('(MOD.)', 'Modified')
            .replace(' - ', '_')
            .replace('.', '')
            .replace(' ', '-')
        )
        source_map[source] = source_value

    lf_data = pd.DataFrame(
        index=df.index.get_level_values('index'),
        columns=[
            'ID',
            'Incomplete',
            'Demand-Type',
            'Demand-Unit',
            'Demand-Offset',
            'Demand-Directional',
            'DV-Unit',
            'LossFunction-Theta_0',
        ],
    )
    # assign common values
    lf_data['Incomplete'] = 0
    lf_data['Demand-Type'] = 'Peak Inundation Height'
    lf_data['Demand-Unit'] = 'in'
    lf_data['Demand-Offset'] = 0
    lf_data['Demand-Directional'] = 1
    lf_data['DV-Unit'] = 'loss_ratio'

    for index, row in df.iterrows():
        # Extract row data
        data_type = index[0]
        row_index = index[1]
        occupancy = row.Occupancy.strip()
        lf_id = row.FnID
        source = source_map[row.Source]
        description = row.Description

        # loss function information
        ys = ', '.join([f'{x/100.00:.3f}' for x in row[ft_cols].to_list()])
        xs = ', '.join([str(x) for x in ft_values.tolist()])
        lf_str = f'{ys}|{xs}'

        lf_data.loc[row_index, 'LossFunction-Theta_0'] = lf_str

        # assign an ID
        lf_id = '.'.join([occupancy, source, data_type])

        other_data = (
            description.lower()
            .replace('contents', '')
            .replace('(equipment)', 'equipment')
            .replace('(inventory)', 'inventory')
            .replace('(equipment/inventory)', 'equipment/inventory')
            .replace('(inventory/equipment)', 'equipment/inventory')
            .replace(':', '')
            .replace('(', '')
            .replace(')', '')
            .split(',')
        )
        for other in other_data:
            other = other.strip()
            if not other:
                continue
            elements = [
                x.replace('-', '_').replace('w/', 'with') for x in other.split(' ')
            ]
            if elements:
                lf_id += '.' + '_'.join(elements)
        lf_id += '-Cost'

        lf_data.loc[row_index, 'ID'] = lf_id

    lf_data['ID'] = lf_data['ID'].apply(remove_repeated_chars)

    lf_data.to_csv(target_data_file, index=False)
