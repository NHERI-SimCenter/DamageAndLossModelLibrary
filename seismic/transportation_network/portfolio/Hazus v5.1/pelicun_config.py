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
"""Hazus Earthquake IM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pelicun
from pelicun.assessment import DLCalculationAssessment


# Convert common length units
def convert_length_units(value, unit_in, unit_out):
    """Convert units."""
    aval_types = ['m', 'mm', 'cm', 'km', 'inch', 'ft', 'mile']
    m = 1.0
    mm = 0.001 * m
    cm = 0.01 * m
    km = 1000 * m
    inch = 0.0254 * m
    ft = 12.0 * inch
    mile = 5280.0 * ft
    scale_map = {
        'm': m,
        'mm': mm,
        'cm': cm,
        'km': km,
        'inch': inch,
        'ft': ft,
        'mile': mile,
    }
    if (unit_in not in aval_types) or (unit_out not in aval_types):
        # TODO (azs): implement a logging system instead of printing these messages
        print(  # noqa: T201
            f'The unit {unit_in} or {unit_out} '
            f'are used in auto_population but not supported'
        )
        return None
    return value * scale_map[unit_in] / scale_map[unit_out]


def get_hazus_bridge_k3d_modifier(hazus_class, aim):
    """
    Calculate K_3D modifier for Hazus bridge seismic analysis.

    This function computes the K_3D factor based on the bridge class and
    number of spans according to Hazus methodology. The K_3D factor modifies
    the piers' 2-dimensional capacity to allow for the 3-dimensional arch
    action in the deck.

    Parameters
    ----------
    hazus_class : str
        Hazus bridge classification (e.g., 'HWB1', 'HWB2', etc.).
    aim : dict
        Asset Information Model containing bridge characteristics,
        specifically 'NumOfSpans'.

    Returns
    -------
    float
        K_3D modifier value for the given bridge class and span configuration.
    """
    # In HAZUS, the K_3D for HWB28 is undefined, so we return 1, i.e., no scaling
    # The K-3D factors for HWB3 and HWB4 are defined as EQ1, which leads to division by zero
    # This is an error in the HAZUS documentation, and we assume that the factors are 1 for these classes
    mapping = {
        'HWB1': 1,
        'HWB2': 1,
        'HWB3': 1,
        'HWB4': 1,
        'HWB5': 1,
        'HWB6': 1,
        'HWB7': 1,
        'HWB8': 2,
        'HWB9': 3,
        'HWB10': 2,
        'HWB11': 3,
        'HWB12': 4,
        'HWB13': 4,
        'HWB14': 1,
        'HWB15': 5,
        'HWB16': 3,
        'HWB17': 1,
        'HWB18': 1,
        'HWB19': 1,
        'HWB20': 2,
        'HWB21': 3,
        'HWB22': 2,
        'HWB23': 3,
        'HWB24': 6,
        'HWB25': 6,
        'HWB26': 7,
        'HWB27': 7,
        'HWB28': 8,
    }
    factors = {
        1: (0.25, 1),
        2: (0.33, 0),
        3: (0.33, 1),
        4: (0.09, 1),
        5: (0.05, 0),
        6: (0.2, 1),
        7: (0.1, 0),
    }
    if hazus_class in ['HWB3', 'HWB4', 'HWB28']:
        return 1
    n = aim['NumOfSpans']
    if n < 2:
        return 1
    a = factors[mapping[hazus_class]][0]
    b = factors[mapping[hazus_class]][1]
    return 1 + a / (n - b)  # This is the original form in Mander and Basoz (1999)


def classify_bridge_for_hazus(aim):  # noqa: C901
    """
    Classify bridge into Hazus bridge categories.

    This function determines the appropriate Hazus bridge classification
    (HWB1-HWB28) based on bridge characteristics including structure type,
    state code, year built, number of spans, and maximum span length.

    Parameters
    ----------
    aim : dict
        Asset Information Model containing bridge characteristics:
        - BridgeClass: Structure type code
        - StateCode: State where bridge is located
        - YearBuilt: Year of construction
        - NumOfSpans: Number of spans
        - MaxSpanLength: Length of maximum span
        - units: Dictionary containing length units

    Returns
    -------
    str
        Hazus bridge classification code (e.g., 'HWB1', 'HWB2', etc.).
    """
    structure_type = aim['BridgeClass']
    # if (
    #     type(structure_type) == str
    #     and len(structure_type) > 3
    #     and structure_type[:3] == "HWB"
    #     and 0 < int(structure_type[3:])
    #     and 29 > int(structure_type[3:])
    # ):
    #     return AIM["bridge_class"]
    state = aim['StateCode']
    yr_built = aim['YearBuilt']
    num_span = aim['NumOfSpans']
    len_max_span = aim['MaxSpanLength']
    len_unit = aim['units']['length']
    len_max_span = convert_length_units(len_max_span, len_unit, 'm')

    seismic = (int(state) == 6 and int(yr_built) >= 1975) or (
        int(state) != 6 and int(yr_built) >= 1990
    )
    # Use a catch-all, other class by default
    bridge_class = 'HWB28'

    if len_max_span > 150:
        if not seismic:
            bridge_class = 'HWB1'
        else:
            bridge_class = 'HWB2'

    elif num_span == 1:
        if not seismic:
            bridge_class = 'HWB3'
        else:
            bridge_class = 'HWB4'

    elif structure_type in list(range(101, 107)):
        if not seismic:
            if state != 6:
                bridge_class = 'HWB5'
            else:
                bridge_class = 'HWB6'
        else:
            bridge_class = 'HWB7'

    elif structure_type in [205, 206]:
        if not seismic:
            bridge_class = 'HWB8'
        else:
            bridge_class = 'HWB9'

    elif structure_type in list(range(201, 207)):
        if not seismic:
            bridge_class = 'HWB10'
        else:
            bridge_class = 'HWB11'

    elif structure_type in list(range(301, 307)):
        if not seismic:
            if len_max_span >= 20:
                if state != 6:
                    bridge_class = 'HWB12'
                else:
                    bridge_class = 'HWB13'
            elif state != 6:
                bridge_class = 'HWB24'
            else:
                bridge_class = 'HWB25'
        else:
            bridge_class = 'HWB14'

    elif structure_type in list(range(402, 411)):
        if not seismic:
            if len_max_span >= 20:
                bridge_class = 'HWB15'
            elif state != 6:
                bridge_class = 'HWB26'
            else:
                bridge_class = 'HWB27'
        else:
            bridge_class = 'HWB16'

    elif structure_type in list(range(501, 507)):
        if not seismic:
            if state != 6:
                bridge_class = 'HWB17'
            else:
                bridge_class = 'HWB18'
        else:
            bridge_class = 'HWB19'

    elif structure_type in [605, 606]:
        if not seismic:
            bridge_class = 'HWB20'
        else:
            bridge_class = 'HWB21'

    elif structure_type in list(range(601, 608)):
        if not seismic:
            bridge_class = 'HWB22'
        else:
            bridge_class = 'HWB23'

    # TODO: review and add HWB24-27 rules  # noqa: TD002
    # TODO: also double check rules for HWB10-11 and HWB22-23  # noqa: TD002

    return bridge_class


def get_hazus_bridge_pgd_modifier(hazus_class, aim):
    """
    Calculate PGD modifier for Hazus bridge seismic analysis.

    This function computes the f1 and f2 factors that scale the Permanent
    Ground Deformation (PGD) capacity of bridges in damage states 2-4. The
    factors are based on bridge geometry and skew angle according to Hazus
    methodology.

    Parameters
    ----------
    hazus_class : str
        Hazus bridge classification (e.g., 'HWB1', 'HWB2', etc.).
    aim : dict
        Asset Information Model containing bridge characteristics:
        - DeckWidth: Width of bridge deck
        - NumOfSpans: Number of spans
        - Skew: Skew angle of bridge
        - StructureLength: Total length of structure

    Returns
    -------
    tuple of float
        The f1 and f2 factors to modify PGD capacity for the given bridge class
        and geometry.
    """
    # This is the original modifier in HAZUS, which gives inf if Skew is 0
    # modifier1 = 0.5*AIM['StructureLength']/(AIM['DeckWidth']*AIM['NumOfSpans']*np.sin(AIM['Skew']/180.0*np.pi))
    # Use the modifier that is corrected from HAZUS manual to achieve the asymptotic behavior
    # Where longer bridges, narrower bridges, less span and higher skew leads to lower modifier (i.e., more fragile bridges)
    modifier1 = (
        aim['DeckWidth']
        * aim['NumOfSpans']
        * np.sin((90 - aim['Skew']) / 180.0 * np.pi)
        / (aim['StructureLength'] * 0.5)
    )
    modifier2 = np.sin((90 - aim['Skew']) / 180.0 * np.pi)
    mapping = {
        'HWB1': (1, 1),
        'HWB2': (1, 1),
        'HWB3': (1, 1),
        'HWB4': (1, 1),
        'HWB5': (modifier1, modifier1),
        'HWB6': (modifier1, modifier1),
        'HWB7': (modifier1, modifier1),
        'HWB8': (1, modifier2),
        'HWB9': (1, modifier2),
        'HWB10': (1, modifier2),
        'HWB11': (1, modifier2),
        'HWB12': (modifier1, modifier1),
        'HWB13': (modifier1, modifier1),
        'HWB14': (modifier1, modifier1),
        'HWB15': (1, modifier2),
        'HWB16': (1, modifier2),
        'HWB17': (modifier1, modifier1),
        'HWB18': (modifier1, modifier1),
        'HWB19': (modifier1, modifier1),
        'HWB20': (1, modifier2),
        'HWB21': (1, modifier2),
        'HWB22': (modifier1, modifier1),
        'HWB23': (modifier1, modifier1),
        'HWB24': (modifier1, modifier1),
        'HWB25': (modifier1, modifier1),
        'HWB26': (1, modifier2),
        'HWB27': (1, modifier2),
        'HWB28': (1, 1),
    }
    return mapping[hazus_class][0], mapping[hazus_class][1]


def classify_tunnel_for_hazus(aim) -> str:
    """
    Classify tunnel into Hazus tunnel categories.

    This function determines the appropriate Hazus tunnel classification
    based on construction type.

    Parameters
    ----------
    aim : dict
        Asset Information Model containing tunnel characteristics,
        specifically 'ConstructType'.

    Returns
    -------
    str
        Hazus tunnel classification code ('HTU1' or 'HTU2').
    """
    if 'Bored' in aim['ConstructType'] or 'Drilled' in aim['ConstructType']:
        return 'HTU1'
    if 'Cut' in aim['ConstructType'] or 'Cover' in aim['ConstructType']:
        return 'HTU2'
    # Select HTU2 for unclassified tunnels because it is more conservative.
    return 'HTU2'


def classify_road_for_hazus(aim) -> str:
    """
    Classify road into Hazus road categories.

    This function determines the appropriate Hazus road classification
    based on road type.

    Parameters
    ----------
    aim : dict
        Asset Information Model containing road characteristics,
        specifically 'RoadType'.

    Returns
    -------
    str
        Hazus road classification code ('HRD1' or 'HRD2').
    """
    if aim['RoadType'] in ['Primary', 'Secondary']:
        return 'HRD1'

    if aim['RoadType'] == 'Residential':
        return 'HRD2'

    # many unclassified roads are urban roads
    return 'HRD2'


def get_hazus_bridge_slight_damage_modifier(hazus_class, aim):
    """
    Calculate slight damage modifier for Hazus bridge seismic analysis.

    This function computes slight damage modifiers for specific Hazus bridge
    classes based on spectral acceleration ratios. The modifier converts short
    periods to an equivalent spectral amplitude at T=1.0 second. Since this
    modifier is a function of the spectral shape of the ground motion demand,
    its value is specific to each demand realization. This function returns
    a specific factor for each realization in the form of an operation string
    the Pelicun can apply to modify the demand sample during the analysis.

    Parameters
    ----------
    hazus_class : str
        Hazus bridge classification (e.g., 'HWB1', 'HWB2', etc.).
    aim : dict
        Asset Information Model containing demand and configuration data.

    Returns
    -------
    list of str or None
        List of operation strings for sample-wise modification, or None
        if the bridge class doesn't require modification.
    """
    if hazus_class in [
        'HWB1',
        'HWB2',
        'HWB5',
        'HWB6',
        'HWB7',
        'HWB8',
        'HWB9',
        'HWB12',
        'HWB13',
        'HWB14',
        'HWB17',
        'HWB18',
        'HWB19',
        'HWB20',
        'HWB21',
        'HWB24',
        'HWB25',
        'HWB28',
    ]:
        return None
    demand_path = Path(aim['DL']['Demands']['DemandFilePath']).resolve()
    sample_size = int(aim['DL']['Demands']['SampleSize'])
    length_unit = aim['GeneralInformation']['units']['length']
    coupled_demands = aim['Applications']['DL']['ApplicationData']['coupled_EDP']
    assessment = DLCalculationAssessment(config_options=None)
    assessment.calculate_demand(
        demand_path=demand_path,
        collapse_limits=None,
        length_unit=length_unit,
        demand_calibration=None,
        sample_size=sample_size,
        demand_cloning=None,
        residual_drift_inference=None,
        coupled_demands=coupled_demands,
    )
    demand_sample, _ = assessment.demand.save_sample(save_units=True)
    edp_types = demand_sample.columns.get_level_values(level='type')
    if (edp_types == 'SA_0.3').sum() != 1:
        msg = (
            'The demand file does not contain the required EDP type SA_0.3'
            ' or contains multiple instances of it.'
        )
        raise ValueError(msg)
    sa_0p3 = demand_sample.loc[  # noqa: PD011
        :, demand_sample.columns.get_level_values(level='type') == 'SA_0.3'
    ].values.flatten()
    if (edp_types == 'SA_1.0').sum() != 1:
        msg = (
            'The demand file does not contain the required EDP type SA_1.0'
            ' or contains multiple instances of it.'
        )
        raise ValueError(msg)
    sa_1p0 = demand_sample.loc[  # noqa: PD011
        :, demand_sample.columns.get_level_values(level='type') == 'SA_1.0'
    ].values.flatten()

    ratio = 2.5 * sa_1p0 / sa_0p3
    operation = [
        f'*{ratio[i]}' if ratio[i] <= 1.0 else '*1.0' for i in range(len(ratio))
    ]

    assert len(operation) == sample_size

    return operation


def auto_populate(aim):
    """
    Automatically creates a performance model for PGA-based Hazus EQ analysis.

    Parameters
    ----------
    AIM: dict
        Asset Information Model - provides features of the asset that can be
        used to infer attributes of the performance model.

    Returns
    -------
    GI_ap: dict
        Extended General Information - extends the GI from the input AIM with
        additional inferred features. These features are typically used in
        intermediate steps during the auto-population and are not required
        for the performance assessment. They are returned to allow reviewing
        how these latent variables affect the final results.
    DL_ap: dict
        Damage and Loss parameters - these define the performance model and
        details of the calculation.
    CMP: DataFrame
        Component assignment - Defines the components (in rows) and their
        location, direction, and quantity (in columns).
    """
    # extract the General Information
    gi = aim.get('GeneralInformation', None)

    if gi is None:
        # TODO: show an error message  # noqa: TD002
        pass

    # initialize the auto-populated GI
    gi_ap = gi.copy()

    dl_app_data = aim['Applications']['DL']['ApplicationData']
    ground_failure = dl_app_data['ground_failure']

    inf_type = gi['assetSubtype']

    if inf_type == 'HwyBridge':
        # If Skew is labeled as 99, it means there is a major variation in skews of substructure units. (Per NBI coding guide)
        # Assume a number of 45 as the "average" skew for the bridge.
        if gi['Skew'] == 99:
            gi['Skew'] = 45

        # get the bridge class
        bt = classify_bridge_for_hazus(gi)
        gi_ap['BridgeHazusClass'] = bt

        # fmt: off
        comp = pd.DataFrame(
            {f'HWB.GS.{bt[3:]}': [  'ea',         1,          1,        1,   'N/A']},
            index = [            'Units', 'Location', 'Direction', 'Theta_0', 'Family']
        ).T
        # fmt: on

        # scaling_specification
        k_skew = np.sqrt(np.sin((90 - gi['Skew']) * np.pi / 180.0))
        k_3d = get_hazus_bridge_k3d_modifier(bt, gi)
        k_shape = get_hazus_bridge_slight_damage_modifier(bt, aim)
        scaling_specification = {
            f'HWB.GS.{bt[3:]}-1-1': {
                'LS2': f'*{k_skew * k_3d}',
                'LS3': f'*{k_skew * k_3d}',
                'LS4': f'*{k_skew * k_3d}',
            }
        }
        if k_shape is not None:
            scaling_specification[f'HWB.GS.{bt[3:]}-1-1']['LS1'] = k_shape
        # if needed, add components to simulate damage from ground failure
        if ground_failure:
            # fmt: off
            comp_gf = pd.DataFrame(
                {f'HWB.GF':          [  'ea',         1,          1,        1,   'N/A']},  # noqa: F541
                index = [     'Units', 'Location', 'Direction', 'Theta_0', 'Family']
            ).T
            # fmt: on

            comp = pd.concat([comp, comp_gf], axis=0)

            f1, f2 = get_hazus_bridge_pgd_modifier(bt, gi)

            scaling_specification.update(
                {
                    'HWB.GF-1-1': {
                        'LS2': f'*{f1}',
                        'LS3': f'*{f1}',
                        'LS4': f'*{f2}',
                    }
                }
            )

        dl_ap = {
            'Asset': {
                'ComponentAssignmentFile': 'CMP_QNT.csv',
                'ComponentDatabase': 'Hazus Earthquake - Transportation',
                'BridgeHazusClass': bt,
                'PlanArea': '1',
            },
            'Damage': {
                'DamageProcess': 'Hazus Earthquake',
                'ScalingSpecification': scaling_specification,
            },
            'Demands': {},
            'Losses': {
                'Repair': {
                    'ConsequenceDatabase': 'Hazus Earthquake - Transportation',
                    'MapApproach': 'Automatic',
                }
            },
            'Options': {
                'NonDirectionalMultipliers': {'ALL': 1.0},
            },
        }

    elif inf_type == 'HwyTunnel':
        # get the tunnel class
        tt = classify_tunnel_for_hazus(gi)
        gi_ap['TunnelHazusClass'] = tt

        # fmt: off
        comp = pd.DataFrame(
            {f'HTU.GS.{tt[3:]}': [  'ea',         1,          1,        1,   'N/A']},
            index = [            'Units','Location','Direction','Theta_0','Family']
        ).T
        # fmt: on
        # if needed, add components to simulate damage from ground failure
        if ground_failure:
            # fmt: off
            comp_gf = pd.DataFrame(
                {f'HTU.GF':          [  'ea',         1,          1,        1,   'N/A']},  # noqa: F541
                index = [     'Units','Location','Direction','Theta_0','Family']
            ).T
            # fmt: on

            comp = pd.concat([comp, comp_gf], axis=0)

        dl_ap = {
            'Asset': {
                'ComponentAssignmentFile': 'CMP_QNT.csv',
                'ComponentDatabase': 'Hazus Earthquake - Transportation',
                'TunnelHazusClass': tt,
                'PlanArea': '1',
            },
            'Damage': {'DamageProcess': 'Hazus Earthquake'},
            'Demands': {},
            'Losses': {
                'Repair': {
                    'ConsequenceDatabase': 'Hazus Earthquake - Transportation',
                    'MapApproach': 'Automatic',
                }
            },
            'Options': {
                'NonDirectionalMultipliers': {'ALL': 1.0},
            },
        }
    elif inf_type == 'Roadway':
        # get the road class
        rt = classify_road_for_hazus(gi)
        gi_ap['RoadHazusClass'] = rt

        # fmt: off
        comp = pd.DataFrame(
            {},
            index = [           'Units','Location','Direction','Theta_0','Family']
        ).T
        # fmt: on

        if ground_failure:
            # fmt: off
            comp_gf = pd.DataFrame(
                {f'HRD.GF.{rt[3:]}':[  'ea',         1,          1,        1,   'N/A']},
                index = [     'Units','Location','Direction','Theta_0','Family']
            ).T
            # fmt: on

            comp = pd.concat([comp, comp_gf], axis=0)

        dl_ap = {
            'Asset': {
                'ComponentAssignmentFile': 'CMP_QNT.csv',
                'ComponentDatabase': 'Hazus Earthquake - Transportation',
                'RoadHazusClass': rt,
                'PlanArea': '1',
            },
            'Damage': {'DamageProcess': 'Hazus Earthquake'},
            'Demands': {},
            'Losses': {
                'Repair': {
                    'ConsequenceDatabase': 'Hazus Earthquake - Transportation',
                    'MapApproach': 'Automatic',
                }
            },
            'Options': {
                'NonDirectionalMultipliers': {'ALL': 1.0},
            },
        }
    else:
        # TODO (azs): implement a logging system instead of printing these messages
        print('subtype not supported in HWY')  # noqa: T201

    return gi_ap, dl_ap, comp
