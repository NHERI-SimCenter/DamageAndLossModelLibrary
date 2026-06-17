"""Generates HAZUS 5.1 story seismic damage and loss library files."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from pelicun import base


def create_Hazus_EQ_fragility_db(  # noqa: C901, N802
    source_file,
    meta_file,
    target_data_file,
    target_meta_file,
):
    """
    Create a database file based on the HAZUS EQ Technical Manual.

    This method was developed to process a json file with tabulated
    data from v5.1 of the Hazus Earthquake Technical Manual. The json
    file is included under data_sources in the SimCenter
    DamageAndLossModelLibrary repo on GitHub.

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

    """
    # parse the source file
    with open(source_file, encoding='utf-8') as f:  # noqa: PTH123
        raw_data = json.load(f)

    # parse the extra metadata file
    if Path(meta_file).is_file():
        with open(meta_file, encoding='utf-8') as f:  # noqa: PTH123
            frag_meta = json.load(f)
    else:
        frag_meta = {}

    # prepare lists of labels for various building features
    design_levels = list(
        raw_data['Structural_Fragility_Groups']['EDP_limits'].keys()
    )

    building_types = list(
        raw_data['Structural_Fragility_Groups']['P_collapse'].keys()
    )

    convert_design_level = {
        'High_code': 'HC',
        'Moderate_code': 'MC',
        'Low_code': 'LC',
        'Pre_code': 'PC',
    }

    # initialize the fragility table
    df_db = pd.DataFrame(
        columns=[
            'ID',
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
        index=np.arange(len(building_types) * len(design_levels) * 5),
        dtype=float,
    )

    # initialize the dictionary that stores the fragility metadata
    meta_dict = {}

    # add the general information to the meta dict
    if '_GeneralInformation' in frag_meta:
        GI = frag_meta['_GeneralInformation']  # noqa: N806

        # remove the decision variable part from the general info
        GI.pop('DecisionVariables', None)

        for key, item in deepcopy(GI).items():
            if key == 'ComponentGroups_Damage':
                GI.update({'ComponentGroups': item})

            if key.startswith('ComponentGroups'):
                GI.pop(key, None)

        meta_dict.update({'_GeneralInformation': GI})

    counter = 0

    # First, prepare the structural fragilities
    S_data = raw_data['Structural_Fragility_Groups']  # noqa: N806

    for bt in building_types:
        for dl in design_levels:
            if bt in S_data['EDP_limits'][dl]:
                # add a dot in bt between structure and height labels, if needed
                if (len(bt) > 2) and (bt[-1] in {'L', 'M', 'H'}):
                    bt_exp = f'{bt[:-1]}.{bt[-1]}'
                    st = bt[:-1]
                    hc = bt[-1]
                else:
                    bt_exp = bt
                    st = bt
                    hc = None

                # story-level fragilities are based only on the low rise archetypes
                if hc in {'M', 'H'}:
                    continue
                if hc == 'L':
                    bt_exp = st

                # create the component id
                cmp_id = f'STR.{bt_exp}.{convert_design_level[dl]}'
                df_db.loc[counter, 'ID'] = cmp_id

                # store demand specifications
                df_db.loc[counter, 'Demand-Type'] = 'Peak Interstory Drift Ratio'

                df_db.loc[counter, 'Demand-Unit'] = 'rad'
                df_db.loc[counter, 'Demand-Offset'] = 0

                # add metadata
                if hc is not None:
                    cmp_meta = {
                        'Description': (
                            frag_meta['Meta']['Collections']['STR']['Description']
                            + ', '
                            + frag_meta['Meta']['StructuralSystems'][st][
                                'Description'
                            ]
                            + ', '
                            + frag_meta['Meta']['HeightClasses'][hc]['Description']
                            + ', '
                            + frag_meta['Meta']['DesignLevels'][
                                convert_design_level[dl]
                            ]['Description']
                        ),
                        'Comments': (
                            frag_meta['Meta']['Collections']['STR']['Comment']
                            + '\n'
                            + frag_meta['Meta']['StructuralSystems'][st]['Comment']
                            + '\n'
                            + frag_meta['Meta']['HeightClasses'][hc]['Comment']
                            + '\n'
                            + frag_meta['Meta']['DesignLevels'][
                                convert_design_level[dl]
                            ]['Comment']
                        ),
                        'SuggestedComponentBlockSize': '1 EA',
                        'RoundUpToIntegerQuantity': 'True',
                        'LimitStates': {},
                    }
                else:
                    cmp_meta = {
                        'Description': (
                            frag_meta['Meta']['Collections']['STR']['Description']
                            + ', '
                            + frag_meta['Meta']['StructuralSystems'][st][
                                'Description'
                            ]
                            + ', '
                            + frag_meta['Meta']['DesignLevels'][
                                convert_design_level[dl]
                            ]['Description']
                        ),
                        'Comments': (
                            frag_meta['Meta']['Collections']['STR']['Comment']
                            + '\n'
                            + frag_meta['Meta']['StructuralSystems'][st]['Comment']
                            + '\n'
                            + frag_meta['Meta']['DesignLevels'][
                                convert_design_level[dl]
                            ]['Comment']
                        ),
                        'SuggestedComponentBlockSize': '1 EA',
                        'RoundUpToIntegerQuantity': 'True',
                        'LimitStates': {},
                    }

                # store the Limit State parameters
                ds_meta = frag_meta['Meta']['StructuralSystems'][st]['DamageStates']
                for LS_i in range(1, 5):  # noqa: N806
                    df_db.loc[counter, f'LS{LS_i}-Family'] = 'lognormal'
                    df_db.loc[counter, f'LS{LS_i}-Theta_0'] = S_data['EDP_limits'][
                        dl
                    ][bt][LS_i - 1]
                    df_db.loc[counter, f'LS{LS_i}-Theta_1'] = S_data[
                        'Fragility_beta'
                    ][dl]

                    if LS_i == 4:
                        p_coll = S_data['P_collapse'][bt]
                        df_db.loc[counter, f'LS{LS_i}-DamageStateWeights'] = (
                            f'{1.0 - p_coll} | {p_coll}'
                        )

                        cmp_meta['LimitStates'].update(
                            {
                                'LS4': {
                                    'DS4': {'Description': ds_meta['DS4']},
                                    'DS5': {'Description': ds_meta['DS5']},
                                }
                            }
                        )

                    else:
                        cmp_meta['LimitStates'].update(
                            {
                                f'LS{LS_i}': {
                                    f'DS{LS_i}': {
                                        'Description': ds_meta[f'DS{LS_i}']
                                    }
                                }
                            }
                        )

                # store metadata
                meta_dict.update({cmp_id: cmp_meta})

                counter += 1

    # Second, the non-structural drift sensitive one
    NSD_data = raw_data['NonStructural_Drift_Sensitive_Fragility_Groups']  # noqa: N806

    # create the component id
    df_db.loc[counter, 'ID'] = 'NSD'

    # store demand specifications
    df_db.loc[counter, 'Demand-Type'] = 'Peak Interstory Drift Ratio'

    df_db.loc[counter, 'Demand-Unit'] = 'rad'
    df_db.loc[counter, 'Demand-Offset'] = 0

    # add metadata
    cmp_meta = {
        'Description': frag_meta['Meta']['Collections']['NSD']['Description'],
        'Comments': frag_meta['Meta']['Collections']['NSD']['Comment'],
        'SuggestedComponentBlockSize': '1 EA',
        'RoundUpToIntegerQuantity': 'True',
        'LimitStates': {},
    }

    # store the Limit State parameters
    ds_meta = frag_meta['Meta']['Collections']['NSD']['DamageStates']
    for LS_i in range(1, 5):  # noqa: N806
        df_db.loc[counter, f'LS{LS_i}-Family'] = 'lognormal'
        df_db.loc[counter, f'LS{LS_i}-Theta_0'] = NSD_data['EDP_limits'][LS_i - 1]
        df_db.loc[counter, f'LS{LS_i}-Theta_1'] = NSD_data['Fragility_beta']

        # add limit state metadata
        cmp_meta['LimitStates'].update(
            {f'LS{LS_i}': {f'DS{LS_i}': {'Description': ds_meta[f'DS{LS_i}']}}}
        )

    # store metadata
    meta_dict.update({'NSD': cmp_meta})

    counter += 1

    # Third, the non-structural acceleration sensitive fragilities
    NSA_data = raw_data['NonStructural_Acceleration_Sensitive_Fragility_Groups']  # noqa: N806

    for dl in design_levels:
        # create the component id
        cmp_id = f'NSA.{convert_design_level[dl]}'
        df_db.loc[counter, 'ID'] = cmp_id

        # store demand specifications
        df_db.loc[counter, 'Demand-Type'] = 'Peak Floor Acceleration'
        df_db.loc[counter, 'Demand-Unit'] = 'g'
        df_db.loc[counter, 'Demand-Offset'] = 0

        # add metadata
        cmp_meta = {
            'Description': (
                frag_meta['Meta']['Collections']['NSA']['Description']
                + ', '
                + frag_meta['Meta']['DesignLevels'][convert_design_level[dl]][
                    'Description'
                ]
            ),
            'Comments': (
                frag_meta['Meta']['Collections']['NSA']['Comment']
                + '\n'
                + frag_meta['Meta']['DesignLevels'][convert_design_level[dl]][
                    'Comment'
                ]
            ),
            'SuggestedComponentBlockSize': '1 EA',
            'RoundUpToIntegerQuantity': 'True',
            'LimitStates': {},
        }

        # store the Limit State parameters
        ds_meta = frag_meta['Meta']['Collections']['NSA']['DamageStates']
        for LS_i in range(1, 5):  # noqa: N806
            df_db.loc[counter, f'LS{LS_i}-Family'] = 'lognormal'
            df_db.loc[counter, f'LS{LS_i}-Theta_0'] = NSA_data['EDP_limits'][dl][
                LS_i - 1
            ]
            df_db.loc[counter, f'LS{LS_i}-Theta_1'] = NSA_data['Fragility_beta']

            # add limit state metadata
            cmp_meta['LimitStates'].update(
                {f'LS{LS_i}': {f'DS{LS_i}': {'Description': ds_meta[f'DS{LS_i}']}}}
            )

        # store metadata
        meta_dict.update({cmp_id: cmp_meta})

        counter += 1

    # Fourth, the lifeline facilities - only at the building-level resolution

    # Fifth, the ground failure fragilities
    GF_data = raw_data['Ground_Failure']  # noqa: N806

    for direction in ('Horizontal', 'Vertical'):
        for f_depth in ('Shallow', 'Deep'):
            # create the component id
            cmp_id = f'GF.{direction[0]}.{f_depth[0]}'
            df_db.loc[counter, 'ID'] = cmp_id

            # store demand specifications
            df_db.loc[counter, 'Demand-Type'] = 'Permanent Ground Deformation'
            df_db.loc[counter, 'Demand-Unit'] = 'inch'
            df_db.loc[counter, 'Demand-Offset'] = 0

            # add metadata
            cmp_meta = {
                'Description': (
                    frag_meta['Meta']['Collections']['GF']['Description']
                    + f', {direction} Direction, {f_depth} Foundation'
                ),
                'Comments': (frag_meta['Meta']['Collections']['GF']['Comment']),
                'SuggestedComponentBlockSize': '1 EA',
                'RoundUpToIntegerQuantity': 'True',
                'LimitStates': {},
            }

            # store the Limit State parameters
            ds_meta = frag_meta['Meta']['Collections']['GF']['DamageStates']

            df_db.loc[counter, 'LS1-Family'] = 'lognormal'
            df_db.loc[counter, 'LS1-Theta_0'] = GF_data['EDP_limits'][direction][
                f_depth
            ]
            df_db.loc[counter, 'LS1-Theta_1'] = GF_data['Fragility_beta'][direction][
                f_depth
            ]
            p_complete = GF_data['P_Complete']
            df_db.loc[counter, 'LS1-DamageStateWeights'] = (
                f'{1.0 - p_complete} | {p_complete}'
            )

            cmp_meta['LimitStates'].update(
                {
                    'LS1': {
                        'DS1': {'Description': ds_meta['DS1']},
                        'DS2': {'Description': ds_meta['DS2']},
                    }
                }
            )

            # store metadata
            meta_dict.update({cmp_id: cmp_meta})

            counter += 1

    # remove empty rows (from the end)
    df_db.dropna(how='all', inplace=True)  # noqa: PD002

    # All Hazus components have complete fragility info,
    df_db['Incomplete'] = 0

    # none of them are directional,
    df_db['Demand-Directional'] = 0

    # rename the index
    df_db.set_index('ID', inplace=True)  # noqa: PD002

    # convert to optimal datatypes to reduce file size
    df_db = df_db.convert_dtypes()

    # save the fragility data
    df_db.to_csv(target_data_file)

    # save the metadata
    with open(target_meta_file, 'w+', encoding='utf-8') as f:  # noqa: PTH123
        json.dump(meta_dict, f, indent=2)

    print('Successfully parsed and saved the fragility data from Hazus EQ')  # noqa: T201


def create_Hazus_EQ_repair_db(  # noqa: C901, N802
    source_file,
    meta_file,
    target_data_file,
    target_meta_file,
):
    """
    Create a database file based on the HAZUS EQ Technical Manual.

    This method was developed to process a json file with tabulated
    data from v4.2.3 of the Hazus Earthquake Technical Manual. The
    json file is included under data_sources in the SimCenter
    DamageAndLossModelLibrary repo on GitHub.

    Parameters
    ----------
    source_file: string
        Path to the Hazus database file.
    meta_file: string
        Path to the JSON file with metadata about the database.
    target_data_file: string
        Path where the repair DB file should be saved. A csv file is
        expected.
    target_meta_file: string
        Path where the repair DB metadata should be saved. A json file is
        expected.

    """
    # parse the source file
    with open(source_file, encoding='utf-8') as f:  # noqa: PTH123
        raw_data = json.load(f)

    # parse the extra metadata file
    if Path(meta_file).is_file():
        with open(meta_file, encoding='utf-8') as f:  # noqa: PTH123
            frag_meta = json.load(f)
    else:
        frag_meta = {}

    # prepare lists of labels for various building features
    occupancies = list(raw_data['Structural_Fragility_Groups']['Repair_cost'].keys())

    # initialize the output loss table
    # define the columns
    out_cols = [
        'Incomplete',
        'Quantity-Unit',
        'DV-Unit',
    ]
    for DS_i in range(1, 6):  # noqa: N806
        out_cols += [
            f'DS{DS_i}-Theta_0',
        ]

    # create the MultiIndex
    cmp_types = ['STR', 'NSD', 'NSA', 'LF']
    comps = [
        f'{cmp_type}.{occ_type}'
        for cmp_type in cmp_types
        for occ_type in occupancies
    ]
    DVs = ['Cost', 'Time']  # noqa: N806
    df_MI = pd.MultiIndex.from_product([comps, DVs], names=['ID', 'DV'])  # noqa: N806

    df_db = pd.DataFrame(columns=out_cols, index=df_MI, dtype=float)

    # initialize the dictionary that stores the loss metadata
    meta_dict = {}

    # add the general information to the meta dict
    if '_GeneralInformation' in frag_meta:
        GI = frag_meta['_GeneralInformation']  # noqa: N806

        for key, item in deepcopy(GI).items():
            if key == 'ComponentGroups_Loss_Repair':
                GI.update({'ComponentGroups': item})

            if key.startswith('ComponentGroups'):
                GI.pop(key, None)

        meta_dict.update({'_GeneralInformation': GI})

    # First, prepare the structural damage consequences
    S_data = raw_data['Structural_Fragility_Groups']  # noqa: N806

    for occ_type in occupancies:
        # create the component id
        cmp_id = f'STR.{occ_type}'

        cmp_meta = {
            'Description': (
                frag_meta['Meta']['Collections']['STR']['Description']
                + ', '
                + frag_meta['Meta']['OccupancyTypes'][occ_type]['Description']
            ),
            'Comments': (
                frag_meta['Meta']['Collections']['STR']['Comment']
                + '\n'
                + frag_meta['Meta']['OccupancyTypes'][occ_type]['Comment']
            ),
            'SuggestedComponentBlockSize': '1 EA',
            'RoundUpToIntegerQuantity': 'True',
            'DamageStates': {},
        }

        # store the consequence values for each Damage State
        ds_meta = frag_meta['Meta']['Collections']['STR']['DamageStates']
        for DS_i in range(1, 6):  # noqa: N806
            cmp_meta['DamageStates'].update(
                {f'DS{DS_i}': {'Description': ds_meta[f'DS{DS_i}']}}
            )

            # DS4 and DS5 have identical repair consequences
            if DS_i == 5:
                ds_i = 4
            else:
                ds_i = DS_i

            # Convert percentage to ratio.
            df_db.loc[(cmp_id, 'Cost'), f'DS{DS_i}-Theta_0'] = (
                f"{S_data['Repair_cost'][occ_type][ds_i - 1] / 100.00:.3f}"
            )

            df_db.loc[(cmp_id, 'Time'), f'DS{DS_i}-Theta_0'] = S_data['Repair_time'][
                occ_type
            ][ds_i - 1]

        # store metadata
        meta_dict.update({cmp_id: cmp_meta})

    # Second, the non-structural drift sensitive one
    NSD_data = raw_data['NonStructural_Drift_Sensitive_Fragility_Groups']  # noqa: N806

    for occ_type in occupancies:
        # create the component id
        cmp_id = f'NSD.{occ_type}'

        cmp_meta = {
            'Description': (
                frag_meta['Meta']['Collections']['NSD']['Description']
                + ', '
                + frag_meta['Meta']['OccupancyTypes'][occ_type]['Description']
            ),
            'Comments': (
                frag_meta['Meta']['Collections']['NSD']['Comment']
                + '\n'
                + frag_meta['Meta']['OccupancyTypes'][occ_type]['Comment']
            ),
            'SuggestedComponentBlockSize': '1 EA',
            'RoundUpToIntegerQuantity': 'True',
            'DamageStates': {},
        }

        # store the consequence values for each Damage State
        ds_meta = frag_meta['Meta']['Collections']['NSD']['DamageStates']
        for DS_i in range(1, 5):  # noqa: N806
            cmp_meta['DamageStates'].update(
                {f'DS{DS_i}': {'Description': ds_meta[f'DS{DS_i}']}}
            )

            # Convert percentage to ratio.
            df_db.loc[(cmp_id, 'Cost'), f'DS{DS_i}-Theta_0'] = (
                f"{NSD_data['Repair_cost'][occ_type][DS_i - 1] / 100.00:.3f}"
            )

        # store metadata
        meta_dict.update({cmp_id: cmp_meta})

    # Third, the non-structural acceleration sensitive fragilities
    NSA_data = raw_data['NonStructural_Acceleration_Sensitive_Fragility_Groups']  # noqa: N806

    for occ_type in occupancies:
        # create the component id
        cmp_id = f'NSA.{occ_type}'

        cmp_meta = {
            'Description': (
                frag_meta['Meta']['Collections']['NSA']['Description']
                + ', '
                + frag_meta['Meta']['OccupancyTypes'][occ_type]['Description']
            ),
            'Comments': (
                frag_meta['Meta']['Collections']['NSA']['Comment']
                + '\n'
                + frag_meta['Meta']['OccupancyTypes'][occ_type]['Comment']
            ),
            'SuggestedComponentBlockSize': '1 EA',
            'RoundUpToIntegerQuantity': 'True',
            'DamageStates': {},
        }

        # store the consequence values for each Damage State
        ds_meta = frag_meta['Meta']['Collections']['NSA']['DamageStates']
        for DS_i in range(1, 5):  # noqa: N806
            cmp_meta['DamageStates'].update(
                {f'DS{DS_i}': {'Description': ds_meta[f'DS{DS_i}']}}
            )

            # Convert percentage to ratio.
            df_db.loc[(cmp_id, 'Cost'), f'DS{DS_i}-Theta_0'] = (
                f"{NSA_data['Repair_cost'][occ_type][DS_i - 1] / 100.00:.3f}"
            )

        # store metadata
        meta_dict.update({cmp_id: cmp_meta})

    # Fourth, the lifeline facilities - only at the building-level resolution

    # remove empty rows (from the end)
    df_db.dropna(how='all', inplace=True)  # noqa: PD002

    # All Hazus components have complete fragility info,
    df_db['Incomplete'] = 0
    # df_db.loc[:, 'Incomplete'] = 0

    # The damage quantity unit is the same for all consequence values
    df_db.loc[:, 'Quantity-Unit'] = '1 EA'

    # The output units are also identical among all components
    idx = base.idx
    df_db.loc[idx[:, 'Cost'], 'DV-Unit'] = 'loss_ratio'
    df_db.loc[idx[:, 'Time'], 'DV-Unit'] = 'day'

    # convert to simple index
    df_db = base.convert_to_SimpleIndex(df_db, 0)

    # rename the index
    df_db.index.name = 'ID'

    # convert to optimal datatypes to reduce file size
    df_db = df_db.convert_dtypes()

    # save the consequence data
    df_db.to_csv(target_data_file)

    # save the metadata - later
    with open(target_meta_file, 'w+', encoding='utf-8') as f:  # noqa: PTH123
        json.dump(meta_dict, f, indent=2)

    print('Successfully parsed and saved the repair consequence data from Hazus EQ')  # noqa: T201


def main():
    """Generate HAZUS 5.1 story seismic damage and loss library files."""
    create_Hazus_EQ_fragility_db(
        source_file=(
            'seismic/building/portfolio/Hazus v5.1/'
            'data_sources/input_files/hazus_data_eq.json'
        ),
        meta_file=(
            'seismic/building/portfolio/Hazus v5.1/'
            'data_sources/input_files/Hazus_meta.json'
        ),
        target_data_file='seismic/building/subassembly/Hazus v5.1/fragility.csv',
        target_meta_file='seismic/building/subassembly/Hazus v5.1/fragility.json',
    )

    create_Hazus_EQ_repair_db(
        source_file=(
            'seismic/building/portfolio/Hazus v5.1/'
            'data_sources/input_files/hazus_data_eq.json'
        ),
        meta_file=(
            'seismic/building/portfolio/Hazus v5.1/'
            'data_sources/input_files/Hazus_meta.json'
        ),
        target_data_file=(
            'seismic/building/subassembly/Hazus v5.1/consequence_repair.csv'
        ),
        target_meta_file=(
            'seismic/building/subassembly/Hazus v5.1/consequence_repair.json'
        ),
    )


if __name__ == '__main__':
    main()
