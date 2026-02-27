#
# Copyright (c) 2023 Leland Stanford Junior University
# Copyright (c) 2023 The Regents of the University of California
#
# This file is part of pelicun.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# pelicun. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Adam Zsarnóczay

import json
from pathlib import Path

import jsonschema
import pandas as pd
from FloodRulesets import configure_flood_vulnerability
from jsonschema import validate
from pelicun import base


def auto_populate(aim):  # noqa: C901
    """
    Automatically creates a performance model for Hazus Hurricane analysis.

    Parameters
    ----------
    AIM: dict
        Asset Information Model - provides features of the asset that can be
        used to infer attributes of the performance model.

    Returns
    -------
    gi: dict
        The GI from the input AIM. Kept for backwards-compatibility, will be
        removed eventually.
        TODO(adamzs): remove this output once all auto-pop scripts have been
        replaced by mapping scripts.
    dl_ap: dict
        Damage and Loss parameters - these define the performance model and
        details of the calculation.
    comp: DataFrame
        Component assignment - Defines the components (in rows) and their
        location, direction, and quantity (in columns).
    """
    # extract the General Information
    general_information_input = aim.get('GeneralInformation', None)

    # parse the GI data

    # Year built
    alname_yearbuilt = ['YearBuiltNJDEP', 'yearBuilt', 'YearBuiltMODIV']
    yearbuilt = None
    try:
        yearbuilt = general_information_input['YearBuilt']
    except KeyError:
        for i in alname_yearbuilt:
            if i in general_information_input:
                yearbuilt = general_information_input[i]
                break

    # if none of the above works, set a default
    if yearbuilt is None:
        yearbuilt = 1985

    # maps for split level
    auto_populated_split_level = {'NO': 0, 'YES': 1, False: 0, True: 1}

    # maps for design level (Marginal Engineered is mapped to Engineered as default)
    auto_populated_design_level = {'E': 'E', 'NE': 'NE', 'PE': 'PE', 'ME': 'E'}
    design_level = general_information_input.get('DesignLevel', 'E')
    if pd.isna(design_level):
        design_level = 'E'

    foundation = general_information_input.get('FoundationType', 3501)
    if pd.isna(foundation):
        foundation = 3501

    nunits = general_information_input.get('NoUnits', 1)
    if pd.isna(nunits):
        nunits = 1

    # maps for flood zone
    auto_populated_flood_zone_mapping = {
        # Coastal areas with a 1% or greater chance of flooding and an
        # additional hazard associated with storm waves.
        6101: 'VE',
        6102: 'VE',
        6103: 'AE',
        6104: 'AE',
        6105: 'AO',
        6106: 'AE',
        6107: 'AH',
        6108: 'AO',
        6109: 'A',
        6110: 'X',
        6111: 'X',
        6112: 'X',
        6113: 'OW',
        6114: 'D',
        6115: 'NA',
        6119: 'NA',
    }
    if isinstance(general_information_input['FloodZone'], int):
        # NJDEP code for flood zone (conversion to the FEMA designations)
        floodzone_fema = auto_populated_flood_zone_mapping[
            general_information_input['FloodZone']
        ]
    else:
        # standard input should follow the FEMA flood zone designations
        floodzone_fema = general_information_input['FloodZone']

    # add the parsed data to the BIM dict
    general_information_auto_populated = general_information_input.copy()
    general_information_auto_populated.update(
        {
            'YearBuilt': int(yearbuilt),
            'DesignLevel': str(
                auto_populated_design_level[design_level]
            ),  # default engineered
            'NumberOfUnits': int(nunits),
            'FirstFloorElevation': float(
                general_information_input.get('FirstFloorHt1', 10.0)
            ),
            'SplitLevel': bool(
                auto_populated_split_level[
                    general_information_input.get('SplitLevel', 'NO')
                ]
            ),  # default: no
            'FoundationType': int(foundation),  # default: pile
            'City': general_information_input.get('City', 'NA'),
            'FloodZone': str(floodzone_fema),
        }
    )

    # prepare the flood rulesets
    fld_config = configure_flood_vulnerability(general_information_auto_populated)

    if fld_config is None:
        info_dict = {
            key: general_information_auto_populated.get(key, '')
            for key in [
                'OccupancyClass',
                'NumberOfStories',
                'FloodType',
                'BasementType',
                'PostFIRM',
            ]
        }

        # TODO (azs): implement a logging system instead of printing these messages
        msg = (
            f'No matching flood archetype configuration available for the '
            f'following attributes:\n'
            f'{info_dict}'
        )
        print(msg)  # noqa: T201
        # raise ValueError(msg)

    # prepare the component assignment
    comp = pd.DataFrame(
        {f'{fld_config}': ['ea', 1, 1, 1, 'N/A']},
        index=['Units', 'Location', 'Direction', 'Theta_0', 'Family'],
    ).T

    damage_loss_config_auto_populated = {
        'Asset': {
            'ComponentAssignmentFile': 'CMP_QNT.csv',
            'ComponentDatabase': 'None',
            'NumberOfStories': 1,
        },
        'Demands': {},
        'Losses': {
            'Repair': {
                'ConsequenceDatabase': 'Hazus Hurricane Storm Surge - Buildings',
                'MapApproach': 'Automatic',
                # "MapFilePath": "loss_map.csv",
                'DecisionVariables': {
                    'Cost': True,
                    'Carbon': False,
                    'Energy': False,
                    'Time': False,
                },
            }
        },
        'Options': {
            'NonDirectionalMultipliers': {'ALL': 1.0},
        },
    }

    # adjust demands to consider first floor elevation

    # get the length unit
    ft_to_demand_unit = base.convert_units(
        1.0,
        unit='ft',
        to_unit=general_information_auto_populated['units']['length'],
        category='length',
    )

    demand_file = Path(aim['DL']['Demands']['DemandFilePath']).resolve()
    original_demands = pd.read_csv(demand_file, index_col=0)
    for col in original_demands.columns:
        if 'PIH' in col:
            extension = original_demands[col]
            break
    col_original = col

    col_parts = col_original.split('-')
    if col_parts[1] == 'PIH':
        col_parts[2] = '0'
    else:
        col_parts[1] = '0'
    col_mod = '-'.join(col_parts)

    # move the original demands to location 0
    original_demands[col_mod] = extension.to_numpy().copy()
    original_demands[col_original] = (
        extension.to_numpy()
        - general_information_auto_populated['FirstFloorElevation']
        * ft_to_demand_unit
    )

    original_demands.to_csv(demand_file)

    return (
        general_information_auto_populated,
        damage_loss_config_auto_populated,
        comp,
    )
