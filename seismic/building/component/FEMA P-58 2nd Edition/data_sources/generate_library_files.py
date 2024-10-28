"""Generates FEMA P-58 2nd edition damage and loss library files."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pelicun import base
from pelicun.uq import fit_distribution_to_percentiles
from scipy.stats import norm  # type: ignore

# pylint: disable=possibly-used-before-assignment
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=used-before-assignment


def parse_DS_Hierarchy(DSH):  # noqa: N802
    """
    Parse the FEMA P58 DS hierarchy into a set of arrays.

    Parameters
    ----------
    DSH: str
       Damage state hierarchy

    Returns
    -------
    list
      Damage state setup
    """
    if DSH[:3] == 'Seq':
        DSH = DSH[4:-1]

    DS_setup = []

    while len(DSH) > 0:
        if DSH[:2] == 'DS':
            DS_setup.append(DSH[:3])
            DSH = DSH[4:]
        elif DSH[:5] in {'MutEx', 'Simul'}:
            closing_pos = DSH.find(')')
            subDSH = DSH[: closing_pos + 1]
            DSH = DSH[closing_pos + 2 :]

            DS_setup.append([subDSH[:5]] + subDSH[6:-1].split(','))

    return DS_setup


def create_FEMA_P58_fragility_files(  # noqa: C901, N802
    source_file,
    meta_file,
    target_data_file,
    target_meta_file,
):
    """
    Create a fragility parameter database based on the FEMA P58 data.

    The method was developed to process v3.1.2 of the FragilityDatabase xls
    that is provided with FEMA P58 2nd edition.

    Parameters
    ----------
    source_file: string
        Path to the fragility database file.
    meta_file: string
        Path to the JSON file with metadata about the database.
    target_data_file: string
        Path where the fragility data file should be saved. A csv file is
        expected.
    target_meta_file: string
        Path where the fragility metadata should be saved. A json file is
        expected.

    Raises
    ------
    ValueError
        If there are problems with the mutually exclusive damage state
        definition of some component.
    """
    # parse the source file
    df = pd.read_excel(
        source_file,
        sheet_name='Summary',
        header=2,
        index_col=1,
        true_values=['YES', 'Yes', 'yes'],
        false_values=['NO', 'No', 'no'],
    )

    # parse the extra metadata file
    if Path(meta_file).is_file():
        with open(meta_file, encoding='utf-8') as f:
            frag_meta = json.load(f)
    else:
        frag_meta = {}

    # remove the empty rows and columns
    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    # filter the columns that we need for the fragility database
    cols_to_db = [
        'Demand Parameter (value):',
        'Demand Parameter (unit):',
        'Demand Location (use floor above? Yes/No)',
        'Directional?',
        'DS Hierarchy',
        'DS 1, Probability',
        'DS 1, Median Demand',
        'DS 1, Total Dispersion (Beta)',
        'DS 2, Probability',
        'DS 2, Median Demand',
        'DS 2, Total Dispersion (Beta)',
        'DS 3, Probability',
        'DS 3, Median Demand',
        'DS 3, Total Dispersion (Beta)',
        'DS 4, Probability',
        'DS 4, Median Demand',
        'DS 4, Total Dispersion (Beta)',
        'DS 5, Probability',
        'DS 5, Median Demand',
        'DS 5, Total Dispersion (Beta)',
    ]

    # filter the columns that we need for the metadata
    cols_to_meta = [
        'Component Name',
        'Component Description',
        'Construction Quality:',
        'Seismic Installation Conditions:',
        'Comments / Notes',
        'Author',
        'Fragility Unit of Measure',
        'Round to Integer Unit?',
        'DS 1, Description',
        'DS 1, Repair Description',
        'DS 2, Description',
        'DS 2, Repair Description',
        'DS 3, Description',
        'DS 3, Repair Description',
        'DS 4, Description',
        'DS 4, Repair Description',
        'DS 5, Description',
        'DS 5, Repair Description',
    ]

    # remove special characters to make it easier to work with column names
    str_map = {
        ord(' '): '_',
        ord(':'): None,
        ord('('): None,
        ord(')'): None,
        ord('?'): None,
        ord('/'): None,
        ord(','): None,
    }

    df_db_source = df.loc[:, cols_to_db]
    df_db_source.columns = [s.translate(str_map) for s in cols_to_db]
    df_db_source.sort_index(inplace=True)

    df_meta = df.loc[:, cols_to_meta]
    df_meta.columns = [s.translate(str_map) for s in cols_to_meta]
    # replace missing values with an empty string
    df_meta.fillna('', inplace=True)
    # the metadata shall be stored in strings
    df_meta = df_meta.astype(str)

    # initialize the output fragility table
    df_db = pd.DataFrame(
        columns=[
            'Index',
            'Incomplete',
            'Demand-Type',
            'Demand-Unit',
            'Demand-Offset',
            'Demand-Directional',
            'LS1-Family',
            'LS1-Theta_0',
            'LS1-Theta_1',
            'LS1-DamageStateWeights',
            'LS2-Family',
            'LS2-Theta_0',
            'LS2-Theta_1',
            'LS2-DamageStateWeights',
            'LS3-Family',
            'LS3-Theta_0',
            'LS3-Theta_1',
            'LS3-DamageStateWeights',
            'LS4-Family',
            'LS4-Theta_0',
            'LS4-Theta_1',
            'LS4-DamageStateWeights',
        ],
        index=df_db_source.index,
        dtype=float,
    )

    # initialize the dictionary that stores the fragility metadata
    meta_dict = {}

    # add the general information to the meta dict
    if '_GeneralInformation' in frag_meta.keys():
        frag_meta = frag_meta['_GeneralInformation']

        # remove the decision variable part from the general info
        frag_meta.pop('DecisionVariables', None)

        meta_dict.update({'_GeneralInformation': frag_meta})

    # conversion dictionary for demand types
    convert_demand_type = {
        'Story Drift Ratio': 'Peak Interstory Drift Ratio',
        'Link Rotation Angle': 'Peak Link Rotation Angle',
        'Effective Drift': 'Peak Effective Drift Ratio',
        'Link Beam Chord Rotation': 'Peak Link Beam Chord Rotation',
        'Peak Floor Acceleration': 'Peak Floor Acceleration',
        'Peak Floor Velocity': 'Peak Floor Velocity',
    }

    # conversion dictionary for demand unit names
    convert_demand_unit = {
        'Unit less': 'unitless',
        'Radians': 'rad',
        'g': 'g',
        'meter/sec': 'mps',
    }

    # for each component...
    # (this approach is not efficient, but easy to follow which was considered
    # more important than efficiency.)
    for cmp in df_db_source.itertuples():
        # create a dotted component index
        ID = cmp.Index.split('.')
        cmpID = f'{ID[0][0]}.{ID[0][1:3]}.{ID[0][3:5]}.{ID[1]}'

        # store the new index
        df_db.loc[cmp.Index, 'Index'] = cmpID

        # assume the component information is complete
        incomplete = False

        # store demand specifications
        df_db.loc[cmp.Index, 'Demand-Type'] = convert_demand_type[
            cmp.Demand_Parameter_value
        ]
        df_db.loc[cmp.Index, 'Demand-Unit'] = convert_demand_unit[
            cmp.Demand_Parameter_unit
        ]
        df_db.loc[cmp.Index, 'Demand-Offset'] = int(
            cmp.Demand_Location_use_floor_above_YesNo
        )
        df_db.loc[cmp.Index, 'Demand-Directional'] = int(cmp.Directional)

        # parse the damage state hierarchy
        DS_setup = parse_DS_Hierarchy(cmp.DS_Hierarchy)

        # get the raw metadata for the component
        cmp_meta = df_meta.loc[cmp.Index, :]

        # store the global (i.e., not DS-specific) metadata

        # start with a comp. description
        if not pd.isna(cmp_meta['Component_Description']):
            comments = cmp_meta['Component_Description']
        else:
            comments = ''

        # the additional fields are added to the description if they exist

        if cmp_meta['Construction_Quality'] != 'Not Specified':
            comments += f'\nConstruction Quality: {cmp_meta["Construction_Quality"]}'

        if cmp_meta['Seismic_Installation_Conditions'] not in [
            'Not Specified',
            'Not applicable',
            'Unknown',
            'Any',
        ]:
            comments += (
                f'\nSeismic Installation Conditions: '
                f'{cmp_meta["Seismic_Installation_Conditions"]}'
            )

        if cmp_meta['Comments__Notes'] != 'None':
            comments += f'\nNotes: {cmp_meta["Comments__Notes"]}'

        if cmp_meta['Author'] not in ['Not Given', 'By User']:
            comments += f'\nAuthor: {cmp_meta["Author"]}'

        # get the suggested block size and replace the misleading values with ea
        block_size = cmp_meta['Fragility_Unit_of_Measure'].split(' ')[::-1]

        meta_data = {
            'Description': cmp_meta['Component_Name'],
            'Comments': comments,
            'SuggestedComponentBlockSize': ' '.join(block_size),
            'RoundUpToIntegerQuantity': cmp_meta['Round_to_Integer_Unit'],
            'LimitStates': {},
        }

        # now look at each Limit State
        for LS_i, LS_contents in enumerate(DS_setup):
            LS_i = LS_i + 1
            LS_contents = np.atleast_1d(LS_contents)

            ls_meta = {}

            # start with the special cases with multiple DSs in an LS
            if LS_contents[0] in {'MutEx', 'Simul'}:
                # collect the fragility data for the member DSs
                median_demands = []
                dispersions = []
                weights = []
                for ds in LS_contents[1:]:
                    median_demands.append(getattr(cmp, f'DS_{ds[2]}_Median_Demand'))

                    dispersions.append(
                        getattr(cmp, f'DS_{ds[2]}_Total_Dispersion_Beta')
                    )

                    weights.append(getattr(cmp, f'DS_{ds[2]}_Probability'))

                # make sure the specified distribution parameters are appropriate
                if (np.unique(median_demands).size != 1) or (
                    np.unique(dispersions).size != 1
                ):
                    msg = (
                        f'Incorrect mutually exclusive DS '
                        f'definition in component {cmp.Index} at '
                        f'Limit State {LS_i}'
                    )
                    raise ValueError(msg)

                if LS_contents[0] == 'MutEx':
                    # in mutually exclusive cases, make sure the specified DS
                    # weights sum up to one
                    np.testing.assert_allclose(
                        np.sum(np.array(weights, dtype=float)),
                        1.0,
                        err_msg=f'Mutually exclusive Damage State weights do '
                        f'not sum to 1.0 in component {cmp.Index} at '
                        f'Limit State {LS_i}',
                    )

                    # and save all DS metadata under this Limit State
                    for ds in LS_contents[1:]:
                        ds_id = ds[2]

                        repair_action = cmp_meta[f'DS_{ds_id}_Repair_Description']
                        if pd.isna(repair_action):
                            repair_action = '<missing data>'

                        ls_meta.update(
                            {
                                f'DS{ds_id}': {
                                    'Description': cmp_meta[
                                        f'DS_{ds_id}_Description'
                                    ],
                                    'RepairAction': repair_action,
                                }
                            }
                        )

                else:
                    # in simultaneous cases, convert simultaneous weights into
                    # mutexc weights
                    sim_ds_count = len(LS_contents) - 1
                    ds_count = 2 ** (sim_ds_count) - 1

                    sim_weights = []

                    for ds_id in range(1, ds_count + 1):
                        ds_map = format(ds_id, f'0{sim_ds_count}b')

                        sim_weights.append(
                            np.prod(
                                [
                                    (
                                        weights[ds_i]
                                        if ds_map[-ds_i - 1] == '1'
                                        else 1.0 - weights[ds_i]
                                    )
                                    for ds_i in range(sim_ds_count)
                                ]
                            )
                        )

                        # save ds metadata - we need to be clever here
                        # the original metadata is saved for the pure cases
                        # when only one DS is triggered
                        # all other DSs store information about which
                        # combination of pure DSs they represent

                        if ds_map.count('1') == 1:
                            ds_pure_id = ds_map[::-1].find('1') + 1

                            repair_action = cmp_meta[
                                f'DS_{ds_pure_id}_Repair_Description'
                            ]
                            if pd.isna(repair_action):
                                repair_action = '<missing data>'

                            ls_meta.update(
                                {
                                    f'DS{ds_id}': {
                                        'Description': f'Pure DS{ds_pure_id}. '
                                        + cmp_meta[f'DS_{ds_pure_id}_Description'],
                                        'RepairAction': repair_action,
                                    }
                                }
                            )

                        else:
                            ds_combo = [
                                f'DS{_.start() + 1}'
                                for _ in re.finditer('1', ds_map[::-1])
                            ]

                            ls_meta.update(
                                {
                                    f'DS{ds_id}': {
                                        'Description': 'Combination of '
                                        + ' & '.join(ds_combo),
                                        'RepairAction': (
                                            'Combination of pure DS repair actions.'
                                        ),
                                    }
                                }
                            )

                    # adjust weights to respect the assumption that at least
                    # one DS will occur (i.e., the case with all DSs returning
                    # False is not part of the event space)
                    sim_weights_array = np.array(sim_weights) / np.sum(sim_weights)

                    weights = sim_weights_array

                theta_0 = median_demands[0]
                theta_1 = dispersions[0]
                weights_str = ' | '.join([f'{w:.6f}' for w in weights])

                df_db.loc[cmp.Index, f'LS{LS_i}-DamageStateWeights'] = weights_str

            # then look at the sequential DS cases
            elif LS_contents[0].startswith('DS'):
                # this is straightforward, store the data in the table and dict
                ds_id = LS_contents[0][2]

                theta_0 = getattr(cmp, f'DS_{ds_id}_Median_Demand')
                theta_1 = getattr(cmp, f'DS_{ds_id}_Total_Dispersion_Beta')

                repair_action = cmp_meta[f'DS_{ds_id}_Repair_Description']
                if pd.isna(repair_action):
                    repair_action = '<missing data>'

                ls_meta.update(
                    {
                        f'DS{ds_id}': {
                            'Description': cmp_meta[f'DS_{ds_id}_Description'],
                            'RepairAction': repair_action,
                        }
                    }
                )

            # FEMA P58 assumes lognormal distribution for every fragility
            df_db.loc[cmp.Index, f'LS{LS_i}-Family'] = 'lognormal'

            # identify incomplete cases...

            # where theta is missing
            if theta_0 != 'By User':
                df_db.loc[cmp.Index, f'LS{LS_i}-Theta_0'] = theta_0
            else:
                incomplete = True

            # where beta is missing
            if theta_1 != 'By User':
                df_db.loc[cmp.Index, f'LS{LS_i}-Theta_1'] = theta_1
            else:
                incomplete = True

            # store the collected metadata for this limit state
            meta_data['LimitStates'].update({f'LS{LS_i}': ls_meta})

        # store the incomplete flag for this component
        df_db.loc[cmp.Index, 'Incomplete'] = int(incomplete)

        # store the metadata for this component
        meta_dict.update({cmpID: meta_data})

    # assign the Index column as the new ID
    df_db.set_index('Index', inplace=True)

    # rename the index
    df_db.index.name = 'ID'

    # convert to optimal datatypes to reduce file size
    df_db = df_db.convert_dtypes()

    # save the fragility data
    df_db.to_csv(target_data_file)

    # save the metadata
    with open(target_meta_file, 'w+', encoding='utf-8') as f:
        json.dump(meta_dict, f, indent=2)

    print('Successfully parsed and saved the fragility data from FEMA P58')


def create_FEMA_P58_repair_files(  # noqa: C901, N802
    source_file,
    meta_file,
    target_data_file,
    target_meta_file,
):
    """
    Create a repair consequence parameter database based on the FEMA P58 data.

    The method was developed to process v3.1.2 of the FragilityDatabase xls
    that is provided with FEMA P58 2nd edition.

    Parameters
    ----------
    source_file: string
        Path to the fragility database file.
    meta_file: string
        Path to the JSON file with metadata about the database.
    target_data_file: string
        Path where the consequence data file should be saved. A csv file is
        expected.
    target_meta_file: string
        Path where the consequence metadata should be saved. A json file is
        expected.

    """
    # parse the source file
    df = pd.concat(
        [
            pd.read_excel(source_file, sheet_name=sheet, header=2, index_col=1)
            for sheet in ('Summary', 'Cost Summary', 'Env Summary')
        ],
        axis=1,
    )

    # parse the extra metadata file
    if Path(meta_file).is_file():
        with open(meta_file, encoding='utf-8') as f:
            frag_meta = json.load(f)
    else:
        frag_meta = {}

    # remove duplicate columns
    # (there are such because we joined two tables that were read separately)
    df = df.loc[:, ~df.columns.duplicated()]

    # remove empty rows and columns
    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    # filter the columns we need for the repair database
    cols_to_db = [
        'Fragility Unit of Measure',
        'DS Hierarchy',
    ]
    for DS_i in range(1, 6):
        cols_to_db += [
            f'Best Fit, DS{DS_i}',
            f'Lower Qty Mean, DS{DS_i}',
            f'Upper Qty Mean, DS{DS_i}',
            f'Lower Qty Cutoff, DS{DS_i}',
            f'Upper Qty Cutoff, DS{DS_i}',
            f'CV / Dispersion, DS{DS_i}',
            # --------------------------
            f'Best Fit, DS{DS_i}.1',
            f'Lower Qty Mean, DS{DS_i}.1',
            f'Upper Qty Mean, DS{DS_i}.1',
            f'Lower Qty Cutoff, DS{DS_i}.1',
            f'Upper Qty Cutoff, DS{DS_i}.1',
            f'CV / Dispersion, DS{DS_i}.2',
            f'DS {DS_i}, Long Lead Time',
            # --------------------------
            f'Repair Cost, p10, DS{DS_i}',
            f'Repair Cost, p50, DS{DS_i}',
            f'Repair Cost, p90, DS{DS_i}',
            f'Time, p10, DS{DS_i}',
            f'Time, p50, DS{DS_i}',
            f'Time, p90, DS{DS_i}',
            f'Mean Value, DS{DS_i}',
            f'Mean Value, DS{DS_i}.1',
            # --------------------------
            # Columns added for the Environmental loss
            f'DS{DS_i} Best Fit',
            f'DS{DS_i} CV or Beta',
            # --------------------------
            f'DS{DS_i} Best Fit.1',
            f'DS{DS_i} CV or Beta.1',
            # --------------------------
            f'DS{DS_i} Embodied Carbon (kg CO2eq)',
            f'DS{DS_i} Embodied Energy (MJ)',
        ]

    # filter the columns that we need for the metadata
    cols_to_meta = [
        'Component Name',
        'Component Description',
        'Construction Quality:',
        'Seismic Installation Conditions:',
        'Comments / Notes',
        'Author',
        'Fragility Unit of Measure',
        'Round to Integer Unit?',
        'DS 1, Description',
        'DS 1, Repair Description',
        'DS 2, Description',
        'DS 2, Repair Description',
        'DS 3, Description',
        'DS 3, Repair Description',
        'DS 4, Description',
        'DS 4, Repair Description',
        'DS 5, Description',
        'DS 5, Repair Description',
    ]

    # remove special characters to make it easier to work with column names
    str_map = {
        ord(' '): '_',
        ord('.'): '_',
        ord(':'): None,
        ord('('): None,
        ord(')'): None,
        ord('?'): None,
        ord('/'): None,
        ord(','): None,
    }

    df_db_source = df.loc[:, cols_to_db]
    df_db_source.columns = [s.translate(str_map) for s in cols_to_db]
    df_db_source.sort_index(inplace=True)

    df_meta = df.loc[:, cols_to_meta]
    df_meta.columns = [s.translate(str_map) for s in cols_to_meta]

    df_db_source.replace('BY USER', np.nan, inplace=True)

    # initialize the output loss table
    # define the columns
    out_cols = [
        'Index',
        'Incomplete',
        'Quantity-Unit',
        'DV-Unit',
    ]
    for DS_i in range(1, 16):
        out_cols += [
            f'DS{DS_i}-Family',
            f'DS{DS_i}-Theta_0',
            f'DS{DS_i}-Theta_1',
            f'DS{DS_i}-LongLeadTime',
        ]

    # create the MultiIndex
    comps = df_db_source.index.values
    DVs = ['Cost', 'Time', 'Carbon', 'Energy']
    df_MI = pd.MultiIndex.from_product([comps, DVs], names=['ID', 'DV'])

    df_db = pd.DataFrame(columns=out_cols, index=df_MI, dtype=float)

    # initialize the dictionary that stores the loss metadata
    meta_dict = {}

    # add the general information to the meta dict
    if '_GeneralInformation' in frag_meta.keys():
        frag_meta = frag_meta['_GeneralInformation']

        meta_dict.update({'_GeneralInformation': frag_meta})

    convert_family = {'LogNormal': 'lognormal', 'Normal': 'normal'}

    # for each component...
    # (this approach is not efficient, but easy to follow which was considered
    # more important than efficiency.)
    for cmp in df_db_source.itertuples():
        ID = cmp.Index.split('.')
        cmpID = f'{ID[0][0]}.{ID[0][1:3]}.{ID[0][3:5]}.{ID[1]}'

        # store the new index
        df_db.loc[cmp.Index, 'Index'] = cmpID

        # assume the component information is complete
        incomplete_cost = False
        incomplete_time = False
        incomplete_carbon = False
        incomplete_energy = False

        # store units

        df_db.loc[cmp.Index, 'Quantity-Unit'] = ' '.join(
            cmp.Fragility_Unit_of_Measure.split(' ')[::-1]
        ).strip()
        df_db.loc[(cmp.Index, 'Cost'), 'DV-Unit'] = 'USD_2011'
        df_db.loc[(cmp.Index, 'Time'), 'DV-Unit'] = 'worker_day'
        df_db.loc[(cmp.Index, 'Carbon'), 'DV-Unit'] = 'kg'
        df_db.loc[(cmp.Index, 'Energy'), 'DV-Unit'] = 'MJ'

        # get the raw metadata for the component
        cmp_meta = df_meta.loc[cmp.Index, :]

        # store the global (i.e., not DS-specific) metadata

        # start with a comp. description
        if not pd.isna(cmp_meta['Component_Description']):
            comments = cmp_meta['Component_Description']
        else:
            comments = ''

        # the additional fields are added to the description if they exist
        if cmp_meta['Construction_Quality'] != 'Not Specified':
            comments += (
                f'\nConstruction Quality: ' f'{cmp_meta["Construction_Quality"]}'
            )

        if cmp_meta['Seismic_Installation_Conditions'] not in [
            'Not Specified',
            'Not applicable',
            'Unknown',
            'Any',
        ]:
            comments += (
                f'\nSeismic Installation Conditions: '
                f'{cmp_meta["Seismic_Installation_Conditions"]}'
            )

        if cmp_meta['Comments__Notes'] != 'None':
            comments += f'\nNotes: {cmp_meta["Comments__Notes"]}'

        if cmp_meta['Author'] not in ['Not Given', 'By User']:
            comments += f'\nAuthor: {cmp_meta["Author"]}'

        # get the suggested block size and replace the misleading values with ea
        block_size = cmp_meta['Fragility_Unit_of_Measure'].split(' ')[::-1]

        meta_data = {
            'Description': cmp_meta['Component_Name'],
            'Comments': comments,
            'SuggestedComponentBlockSize': ' '.join(block_size),
            'RoundUpToIntegerQuantity': cmp_meta['Round_to_Integer_Unit'],
            'ControllingDemand': 'Damage Quantity',
            'DamageStates': {},
        }

        # Handle components with simultaneous damage states separately
        if 'Simul' in cmp.DS_Hierarchy:
            # Note that we are assuming that all damage states are triggered by
            # a single limit state in these components.
            # This assumption holds for the second edition of FEMA P58, but it
            # might need to be revisited in future editions.

            cost_est = {}
            time_est = {}
            carbon_est = {}
            energy_est = {}

            # get the p10, p50, and p90 estimates for all damage states
            for DS_i in range(1, 6):
                if not pd.isna(getattr(cmp, f'Repair_Cost_p10_DS{DS_i}')):
                    cost_est.update(
                        {
                            f'DS{DS_i}': np.array(
                                [
                                    getattr(cmp, f'Repair_Cost_p10_DS{DS_i}'),
                                    getattr(cmp, f'Repair_Cost_p50_DS{DS_i}'),
                                    getattr(cmp, f'Repair_Cost_p90_DS{DS_i}'),
                                    getattr(cmp, f'Lower_Qty_Mean_DS{DS_i}'),
                                    getattr(cmp, f'Upper_Qty_Mean_DS{DS_i}'),
                                ]
                            )
                        }
                    )

                    time_est.update(
                        {
                            f'DS{DS_i}': np.array(
                                [
                                    getattr(cmp, f'Time_p10_DS{DS_i}'),
                                    getattr(cmp, f'Time_p50_DS{DS_i}'),
                                    getattr(cmp, f'Time_p90_DS{DS_i}'),
                                    getattr(cmp, f'Lower_Qty_Mean_DS{DS_i}_1'),
                                    getattr(cmp, f'Upper_Qty_Mean_DS{DS_i}_1'),
                                    int(
                                        getattr(cmp, f'DS_{DS_i}_Long_Lead_Time')
                                        == 'YES'
                                    ),
                                ]
                            )
                        }
                    )

                if not pd.isna(getattr(cmp, f'DS{DS_i}_Embodied_Carbon_kg_CO2eq')):
                    theta_0, theta_1, family = [
                        getattr(cmp, f'DS{DS_i}_Embodied_Carbon_kg_CO2eq'),
                        getattr(cmp, f'DS{DS_i}_CV_or_Beta'),
                        getattr(cmp, f'DS{DS_i}_Best_Fit'),
                    ]

                    if family == 'Normal':
                        p10, p50, p90 = norm.ppf(
                            [0.1, 0.5, 0.9], loc=theta_0, scale=theta_0 * theta_1
                        )
                    elif family == 'LogNormal':
                        p10, p50, p90 = np.exp(
                            norm.ppf(
                                [0.1, 0.5, 0.9], loc=np.log(theta_0), scale=theta_1
                            )
                        )

                    carbon_est.update({f'DS{DS_i}': np.array([p10, p50, p90])})

                if not pd.isna(getattr(cmp, f'DS{DS_i}_Embodied_Energy_MJ')):
                    theta_0, theta_1, family = [
                        getattr(cmp, f'DS{DS_i}_Embodied_Energy_MJ'),
                        getattr(cmp, f'DS{DS_i}_CV_or_Beta_1'),
                        getattr(cmp, f'DS{DS_i}_Best_Fit_1'),
                    ]

                    if family == 'Normal':
                        p10, p50, p90 = norm.ppf(
                            [0.1, 0.5, 0.9], loc=theta_0, scale=theta_0 * theta_1
                        )
                    elif family == 'LogNormal':
                        p10, p50, p90 = np.exp(
                            norm.ppf(
                                [0.1, 0.5, 0.9], loc=np.log(theta_0), scale=theta_1
                            )
                        )

                    energy_est.update({f'DS{DS_i}': np.array([p10, p50, p90])})

            # now prepare the equivalent mutex damage states
            sim_ds_count = len(cost_est.keys())
            ds_count = 2 ** (sim_ds_count) - 1

            for DS_i in range(1, ds_count + 1):
                ds_map = format(DS_i, f'0{sim_ds_count}b')

                cost_vals = np.sum(
                    [
                        (
                            cost_est[f'DS{ds_i + 1}']
                            if ds_map[-ds_i - 1] == '1'
                            else np.zeros(5)
                        )
                        for ds_i in range(sim_ds_count)
                    ],
                    axis=0,
                )

                time_vals = np.sum(
                    [
                        (
                            time_est[f'DS{ds_i + 1}']
                            if ds_map[-ds_i - 1] == '1'
                            else np.zeros(6)
                        )
                        for ds_i in range(sim_ds_count)
                    ],
                    axis=0,
                )

                carbon_vals = np.sum(
                    [
                        (
                            carbon_est[f'DS{ds_i + 1}']
                            if ds_map[-ds_i - 1] == '1'
                            else np.zeros(3)
                        )
                        for ds_i in range(sim_ds_count)
                    ],
                    axis=0,
                )

                energy_vals = np.sum(
                    [
                        (
                            energy_est[f'DS{ds_i + 1}']
                            if ds_map[-ds_i - 1] == '1'
                            else np.zeros(3)
                        )
                        for ds_i in range(sim_ds_count)
                    ],
                    axis=0,
                )

                # fit a distribution
                family_hat, theta_hat = fit_distribution_to_percentiles(
                    cost_vals[:3], [0.1, 0.5, 0.9], ['normal', 'lognormal']
                )

                cost_theta = theta_hat
                if family_hat == 'normal':
                    cost_theta[1] = cost_theta[1] / cost_theta[0]

                time_theta = [
                    time_vals[1],
                    np.sqrt(cost_theta[1] ** 2.0 + 0.25**2.0),
                ]

                # fit distributions to environmental impact consequences
                (
                    family_hat_carbon,
                    theta_hat_carbon,
                ) = fit_distribution_to_percentiles(
                    carbon_vals[:3], [0.1, 0.5, 0.9], ['normal', 'lognormal']
                )

                carbon_theta = theta_hat_carbon
                if family_hat_carbon == 'normal':
                    carbon_theta[1] = carbon_theta[1] / carbon_theta[0]

                (
                    family_hat_energy,
                    theta_hat_energy,
                ) = fit_distribution_to_percentiles(
                    energy_vals[:3], [0.1, 0.5, 0.9], ['normal', 'lognormal']
                )

                energy_theta = theta_hat_energy
                if family_hat_energy == 'normal':
                    energy_theta[1] = energy_theta[1] / energy_theta[0]

                # Note that here we assume that the cutoff quantities are
                # identical across damage states.
                # This assumption holds for the second edition of FEMA P58, but
                # it might need to be revisited in future editions.
                cost_qnt_low = cmp.Lower_Qty_Cutoff_DS1
                cost_qnt_up = cmp.Upper_Qty_Cutoff_DS1
                time_qnt_low = cmp.Lower_Qty_Cutoff_DS1_1
                time_qnt_up = cmp.Upper_Qty_Cutoff_DS1_1

                # store the results
                df_db.loc[(cmp.Index, 'Cost'), f'DS{DS_i}-Family'] = family_hat

                df_db.loc[(cmp.Index, 'Cost'), f'DS{DS_i}-Theta_0'] = (
                    f'{cost_vals[3]:g},{cost_vals[4]:g}|'
                    f'{cost_qnt_low:g},{cost_qnt_up:g}'
                )

                df_db.loc[(cmp.Index, 'Cost'), f'DS{DS_i}-Theta_1'] = (
                    f'{cost_theta[1]:g}'
                )

                df_db.loc[(cmp.Index, 'Time'), f'DS{DS_i}-Family'] = family_hat

                df_db.loc[(cmp.Index, 'Time'), f'DS{DS_i}-Theta_0'] = (
                    f'{time_vals[3]:g},{time_vals[4]:g}|'
                    f'{time_qnt_low:g},{time_qnt_up:g}'
                )

                df_db.loc[(cmp.Index, 'Time'), f'DS{DS_i}-Theta_1'] = (
                    f'{time_theta[1]:g}'
                )

                df_db.loc[(cmp.Index, 'Time'), f'DS{DS_i}-LongLeadTime'] = int(
                    time_vals[5] > 0
                )

                df_db.loc[(cmp.Index, 'Carbon'), f'DS{DS_i}-Family'] = (
                    family_hat_carbon
                )

                df_db.loc[(cmp.Index, 'Carbon'), f'DS{DS_i}-Theta_0'] = (
                    f'{carbon_theta[0]:g}'
                )

                df_db.loc[(cmp.Index, 'Carbon'), f'DS{DS_i}-Theta_1'] = (
                    f'{carbon_theta[1]:g}'
                )

                df_db.loc[(cmp.Index, 'Energy'), f'DS{DS_i}-Family'] = (
                    family_hat_energy
                )

                df_db.loc[(cmp.Index, 'Energy'), f'DS{DS_i}-Theta_0'] = (
                    f'{energy_theta[0]:g}'
                )

                df_db.loc[(cmp.Index, 'Energy'), f'DS{DS_i}-Theta_1'] = (
                    f'{energy_theta[1]:g}'
                )

                if ds_map.count('1') == 1:
                    ds_pure_id = ds_map[::-1].find('1') + 1

                    repair_action = cmp_meta[f'DS_{ds_pure_id}_Repair_Description']
                    if pd.isna(repair_action):
                        repair_action = '<missing data>'

                    meta_data['DamageStates'].update(
                        {
                            f'DS{DS_i}': {
                                'Description': f'Pure DS{ds_pure_id}. '
                                + cmp_meta[f'DS_{ds_pure_id}_Description'],
                                'RepairAction': repair_action,
                            }
                        }
                    )

                else:
                    ds_combo = [
                        f'DS{_.start() + 1}' for _ in re.finditer('1', ds_map[::-1])
                    ]

                    meta_data['DamageStates'].update(
                        {
                            f'DS{DS_i}': {
                                'Description': 'Combination of '
                                + ' & '.join(ds_combo),
                                'RepairAction': 'Combination of pure DS repair '
                                'actions.',
                            }
                        }
                    )

        # for every other component...
        else:
            # now look at each Damage State
            for DS_i in range(1, 6):
                # cost
                if not pd.isna(getattr(cmp, f'Best_Fit_DS{DS_i}')):
                    df_db.loc[(cmp.Index, 'Cost'), f'DS{DS_i}-Family'] = (
                        convert_family[getattr(cmp, f'Best_Fit_DS{DS_i}')]
                    )

                    if not pd.isna(getattr(cmp, f'Lower_Qty_Mean_DS{DS_i}')):
                        theta_0_low = getattr(cmp, f'Lower_Qty_Mean_DS{DS_i}')
                        theta_0_up = getattr(cmp, f'Upper_Qty_Mean_DS{DS_i}')
                        qnt_low = getattr(cmp, f'Lower_Qty_Cutoff_DS{DS_i}')
                        qnt_up = getattr(cmp, f'Upper_Qty_Cutoff_DS{DS_i}')

                        if theta_0_low == 0.0 and theta_0_up == 0.0:
                            df_db.loc[(cmp.Index, 'Cost'), f'DS{DS_i}-Family'] = (
                                np.nan
                            )

                        else:
                            df_db.loc[(cmp.Index, 'Cost'), f'DS{DS_i}-Theta_0'] = (
                                f'{theta_0_low:g},{theta_0_up:g}|'
                                f'{qnt_low:g},{qnt_up:g}'
                            )

                            df_db.loc[(cmp.Index, 'Cost'), f'DS{DS_i}-Theta_1'] = (
                                f"{getattr(cmp, f'CV__Dispersion_DS{DS_i}'):g}"
                            )

                    else:
                        incomplete_cost = True

                    repair_action = cmp_meta[f'DS_{DS_i}_Repair_Description']
                    if pd.isna(repair_action):
                        repair_action = '<missing data>'

                    meta_data['DamageStates'].update(
                        {
                            f'DS{DS_i}': {
                                'Description': cmp_meta[f'DS_{DS_i}_Description'],
                                'RepairAction': repair_action,
                            }
                        }
                    )

                # time
                if not pd.isna(getattr(cmp, f'Best_Fit_DS{DS_i}_1')):
                    df_db.loc[(cmp.Index, 'Time'), f'DS{DS_i}-Family'] = (
                        convert_family[getattr(cmp, f'Best_Fit_DS{DS_i}_1')]
                    )

                    if not pd.isna(getattr(cmp, f'Lower_Qty_Mean_DS{DS_i}_1')):
                        theta_0_low = getattr(cmp, f'Lower_Qty_Mean_DS{DS_i}_1')
                        theta_0_up = getattr(cmp, f'Upper_Qty_Mean_DS{DS_i}_1')
                        qnt_low = getattr(cmp, f'Lower_Qty_Cutoff_DS{DS_i}_1')
                        qnt_up = getattr(cmp, f'Upper_Qty_Cutoff_DS{DS_i}_1')

                        if theta_0_low == 0.0 and theta_0_up == 0.0:
                            df_db.loc[(cmp.Index, 'Time'), f'DS{DS_i}-Family'] = (
                                np.nan
                            )

                        else:
                            df_db.loc[(cmp.Index, 'Time'), f'DS{DS_i}-Theta_0'] = (
                                f'{theta_0_low:g},{theta_0_up:g}|'
                                f'{qnt_low:g},{qnt_up:g}'
                            )

                            df_db.loc[(cmp.Index, 'Time'), f'DS{DS_i}-Theta_1'] = (
                                f"{getattr(cmp, f'CV__Dispersion_DS{DS_i}_2'):g}"
                            )

                        df_db.loc[(cmp.Index, 'Time'), f'DS{DS_i}-LongLeadTime'] = (
                            int(getattr(cmp, f'DS_{DS_i}_Long_Lead_Time') == 'YES')
                        )

                    else:
                        incomplete_time = True

                # Carbon
                if not pd.isna(getattr(cmp, f'DS{DS_i}_Best_Fit')):
                    df_db.loc[(cmp.Index, 'Carbon'), f'DS{DS_i}-Family'] = (
                        convert_family[getattr(cmp, f'DS{DS_i}_Best_Fit')]
                    )

                    df_db.loc[(cmp.Index, 'Carbon'), f'DS{DS_i}-Theta_0'] = getattr(
                        cmp, f'DS{DS_i}_Embodied_Carbon_kg_CO2eq'
                    )

                    df_db.loc[(cmp.Index, 'Carbon'), f'DS{DS_i}-Theta_1'] = getattr(
                        cmp, f'DS{DS_i}_CV_or_Beta'
                    )

                # Energy
                if not pd.isna(getattr(cmp, f'DS{DS_i}_Best_Fit_1')):
                    df_db.loc[(cmp.Index, 'Energy'), f'DS{DS_i}-Family'] = (
                        convert_family[getattr(cmp, f'DS{DS_i}_Best_Fit_1')]
                    )

                    df_db.loc[(cmp.Index, 'Energy'), f'DS{DS_i}-Theta_0'] = getattr(
                        cmp, f'DS{DS_i}_Embodied_Energy_MJ'
                    )

                    df_db.loc[(cmp.Index, 'Energy'), f'DS{DS_i}-Theta_1'] = getattr(
                        cmp, f'DS{DS_i}_CV_or_Beta_1'
                    )

        df_db.loc[(cmp.Index, 'Cost'), 'Incomplete'] = int(incomplete_cost)
        df_db.loc[(cmp.Index, 'Time'), 'Incomplete'] = int(incomplete_time)
        df_db.loc[(cmp.Index, 'Carbon'), 'Incomplete'] = int(incomplete_carbon)
        df_db.loc[(cmp.Index, 'Energy'), 'Incomplete'] = int(incomplete_energy)
        # store the metadata for this component
        meta_dict.update({cmpID: meta_data})

    # assign the Index column as the new ID
    df_db.index = pd.MultiIndex.from_arrays(
        [df_db['Index'].values, df_db.index.get_level_values(1)]
    )

    df_db.drop('Index', axis=1, inplace=True)

    # review the database and drop rows with no information
    cmp_to_drop = []
    for cmp in df_db.index:
        empty = True

        for DS_i in range(1, 6):
            if not pd.isna(df_db.loc[cmp, f'DS{DS_i}-Family']):
                empty = False
                break

        if empty:
            cmp_to_drop.append(cmp)

    df_db.drop(cmp_to_drop, axis=0, inplace=True)
    for cmp in cmp_to_drop:
        if cmp[0] in meta_dict:
            del meta_dict[cmp[0]]

    # convert to optimal datatypes to reduce file size
    df_db = df_db.convert_dtypes()

    df_db = base.convert_to_SimpleIndex(df_db, 0)

    # rename the index
    df_db.index.name = 'ID'

    # save the consequence data
    df_db.to_csv(target_data_file)

    # save the metadata
    with open(target_meta_file, 'w+', encoding='utf-8') as f:
        json.dump(meta_dict, f, indent=2)

    print('Successfully parsed and saved the repair consequence data from FEMA P58')


def main():
    """Generate FEMA P-58 2nd edition damage and loss library files."""
    create_FEMA_P58_fragility_files(
        source_file=(
            'seismic/building/component/FEMA P-58 2nd Edition/'
            'data_sources/input_files/FEMAP-58_FragilityDatabase_v3.1.2.xls'
        ),
        meta_file=(
            'seismic/building/component/FEMA P-58 2nd Edition/'
            'data_sources/input_files/FEMA_P58_meta.json'
        ),
        target_data_file=(
            'seismic/building/component/FEMA P-58 2nd Edition/fragility.csv'
        ),
        target_meta_file=(
            'seismic/building/component/FEMA P-58 2nd Edition/fragility.json'
        ),
    )

    create_FEMA_P58_repair_files(
        source_file=(
            'seismic/building/component/FEMA P-58 2nd Edition/'
            'data_sources/input_files/FEMAP-58_FragilityDatabase_v3.1.2.xls'
        ),
        meta_file=(
            'seismic/building/component/FEMA P-58 2nd Edition/'
            'data_sources/input_files/FEMA_P58_meta.json'
        ),
        target_data_file=(
            'seismic/building/component/'
            'FEMA P-58 2nd Edition/consequence_repair.csv'
        ),
        target_meta_file=(
            'seismic/building/component/'
            'FEMA P-58 2nd Edition/consequence_repair.json'
        ),
    )


if __name__ == '__main__':
    main()
