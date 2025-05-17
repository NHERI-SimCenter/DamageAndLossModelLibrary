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

def auto_populate(aim):  # noqa: C901
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
        # TODO: show an error message
        pass

    # initialize the auto-populated GI
    gi_ap = gi.copy()

    asset_type = aim['assetType']
    dl_app_data = aim['Applications']['DL']['ApplicationData']
    ground_failure = dl_app_data['ground_failure']

    pipe_material_map = {
        'CI': 'B',
        'AC': 'B',
        'RCC': 'B',
        'DI': 'D',
        'PVC': 'D',
        'DS': 'D',
        'BS': 'B',
    }

    # GI = AIM.get("GeneralInformation", None)
    # if GI==None:

    # initialize the auto-populated GI
    wdn_element_type = gi_ap.get('type', 'MISSING')
    asset_name = gi_ap.get('AIM_id', None)

    if wdn_element_type == 'Pipe':
        pipe_construction_year = gi_ap.get('year', None)
        pipe_diameter = gi_ap.get('Diam', None)
        # diamaeter value is a fundamental part of hydraulic
        # performance assessment
        if pipe_diameter is None:
            msg = f'pipe diameter in asset type {asset_type}, \
                             asset id "{asset_name}" has no diameter \
                                 value.'
            raise ValueError(msg)

        pipe_length = gi_ap.get('Len', None)
        # length value is a fundamental part of
        # hydraulic performance assessment
        if pipe_diameter is None:
            msg = f'pipe length in asset type {asset_type}, \
                             asset id "{asset_name}" has no diameter \
                                 value.'
            raise ValueError(msg)

        pipe_material = gi_ap.get('material', None)

        # pipe material can be not available or named "missing" in
        # both case, pipe flexibility will be set to "missing"

        """
        The assumed logic (rullset) is that if the material is
        missing, if the pipe is smaller than or equal to 20
        inches, the material is Cast Iron (CI) otherwise the pipe
        material is steel.
            If the material is steel (ST), either based on user
        specified input or the assumption due to the lack of the
        user-input, the year that the pipe is constructed define
        the flexibility status per HAZUS instructions. If the pipe
        is built in 1935 or after, it is, the pipe is Ductile
        Steel (DS), and otherwise it is Brittle Steel (BS).
            If the pipe is missing construction year and is built
        by steel, we assume consevatively that the pipe is brittle
        (i.e., BS)
        """
        if pipe_material is None:
            if pipe_diameter > 20 * 0.0254:  # 20 inches in meter
                print(
                    f'Asset {asset_name} is missing material. '
                    'Material is assumed to be Cast Iron'
                )
                pipe_material = 'CI'
            else:
                print(
                    f'Asset {asset_name} is missing material. Material is '
                    f'assumed to be Steel (ST)'
                )
                pipe_material = 'ST'

        if pipe_material == 'ST':
            if (pipe_construction_year is not None) and (
                pipe_construction_year >= 1935
            ):
                msg = (
                    f'Asset {asset_name} has material of "ST" '
                    'is assumed to be Ductile Steel.'
                )

                print(msg)
                pipe_material = 'DS'

            else:
                msg = (
                    f'Asset {asset_name} has material of "ST" '
                    'is assumed to be Brittle Steel.'
                )

                print(msg)
                pipe_material = 'BS'

        pipe_flexibility = pipe_material_map.get(pipe_material, 'missing')

        gi_ap['material flexibility'] = pipe_flexibility
        gi_ap['material'] = pipe_material

        # Pipes are broken into 20ft segments (rounding up) and
        # each segment is represented by an individual entry in
        # the performance model, `CMP`. The damage capacity of each
        # segment is assumed to be independent and driven by the
        # same EDP. We therefore replicate the EDP associated with
        # the pipe to the various locations assigned to the
        # segments.

        # Determine number of segments

        pipe_length_unit = gi_ap['units']['length']
        pipe_length_ft = pelicun.base.convert_units(
            pipe_length, unit=pipe_length_unit, to_unit='ft', category='length'
        )
        reference_length = 20.00  # 20 ft
        if pipe_length_ft % reference_length < 1e-2:
            # If the lengths are equal, then that's one segment, not two.
            num_segments = int(pipe_length_ft / reference_length)
        else:
            # In all other cases, round up.
            num_segments = int(pipe_length_ft / reference_length) + 1
        location_string = f'1--{num_segments}' if num_segments > 1 else '1'

        # Define performance model
        # fmt: off

        pipe_fl = f'PWP.{pipe_flexibility}'
        comp = pd.DataFrame(
            {pipe_fl + '.GS': ['ea', location_string, '0', 1, 'N/A'],
             pipe_fl + '.GF': ['ea', location_string, '0', 1, 'N/A'],
             'aggregate': ['ea', location_string, '0', 1, 'N/A']},
            index=['Units', 'Location', 'Direction', 'Theta_0', 'Family']
        ).T
        # fmt: on

        # Set up the demand cloning configuration for the pipe
        # segments, if required.
        demand_config = {}
        if num_segments > 1:
            # determine the EDP tags available for cloning
            response_data = pelicun.file_io.load_data('response.csv', None)
            num_header_entries = len(response_data.columns.names)
            # if 4, assume a hazard level tag is present and remove it
            if num_header_entries == 4:
                response_data.columns = pd.MultiIndex.from_tuples(
                    [x[1::] for x in response_data.columns]
                )
            demand_cloning_config = {}
            for edp in response_data.columns:
                tag, location, direction = edp  # noqa: F841

                demand_cloning_config['-'.join(edp)] = [
                    f'{tag}-{x}-{direction}'
                    for x in [f'{i + 1}' for i in range(num_segments)]
                ]
            demand_config = {'DemandCloning': demand_cloning_config}

        # Create damage process
        dmg_process = {
            f'1_PWP.{pipe_flexibility}.GS-LOC': {'DS1': 'aggregate_DS1'},
            f'2_PWP.{pipe_flexibility}.GF-LOC': {'DS1': 'aggregate_DS1'},
            f'3_PWP.{pipe_flexibility}.GS-LOC': {'DS2': 'aggregate_DS2'},
            f'4_PWP.{pipe_flexibility}.GF-LOC': {'DS2': 'aggregate_DS2'},
        }
        dmg_process_filename = 'dmg_process.json'
        with open(dmg_process_filename, 'w', encoding='utf-8') as f:
            json.dump(dmg_process, f, indent=2)

        # Define the auto-populated config
        dl_ap = {
            'Asset': {
                'ComponentAssignmentFile': 'CMP_QNT.csv',
                'ComponentDatabase': 'Hazus Earthquake - Potable Water',
                'Material Flexibility': pipe_flexibility,
                'PlanArea': '1',  # Sina: does not make sense for water.
                # Kept it here since it was also
                # kept here for Transportation
            },
            'Damage': {
                'DamageProcess': 'User Defined',
                'DamageProcessFilePath': 'dmg_process.json',
            },
            'Demands': demand_config,
        }

    elif wdn_element_type == 'Tank':
        tank_cmp_lines = {
            ('OG', 'C', 1): {'PST.G.C.A.GS': ['ea', 1, 1, 1, 'N/A']},
            ('OG', 'C', 0): {'PST.G.C.U.GS': ['ea', 1, 1, 1, 'N/A']},
            ('OG', 'S', 1): {'PST.G.S.A.GS': ['ea', 1, 1, 1, 'N/A']},
            ('OG', 'S', 0): {'PST.G.S.U.GS': ['ea', 1, 1, 1, 'N/A']},
            # Anchored status and Wood is not defined for On Ground tanks
            ('OG', 'W', 0): {'PST.G.W.GS': ['ea', 1, 1, 1, 'N/A']},
            # Anchored status and Steel is not defined for
            # Above Ground tanks
            ('AG', 'S', 0): {'PST.A.S.GS': ['ea', 1, 1, 1, 'N/A']},
            # Anchored status and Concrete is not defined for Buried tanks.
            ('B', 'C', 0): {'PST.B.C.GF': ['ea', 1, 1, 1, 'N/A']},
        }

        # The default values are assumed: material = Concrete (C),
        # location= On Ground (OG), and Anchored = 1
        tank_material = gi_ap.get('material', 'C')
        tank_location = gi_ap.get('location', 'OG')
        tank_anchored = gi_ap.get('anchored', 1)

        tank_material_allowable = {'C', 'S'}
        if tank_material not in tank_material_allowable:
            msg = f'Tank\'s material = "{tank_material}" is \
                 not allowable in tank {asset_name}. The \
                 material must be either C for concrete or S \
                 for steel.'
            raise ValueError(msg)

        tank_location_allowable = {'AG', 'OG', 'B'}
        if tank_location not in tank_location_allowable:
            msg = f'Tank\'s location = "{tank_location}" is \
                 not allowable in tank {asset_name}. The \
                 location must be either "AG" for Above \
                 ground, "OG" for On Ground or "BG" for \
                 Below Ground (buried) Tanks.'
            raise ValueError(msg)

        tank_anchored_allowable = {0, 1}
        if tank_anchored not in tank_anchored_allowable:
            msg = f'Tank\'s anchored status = "{tank_location}\
                 " is not allowable in tank {asset_name}. \
                 The anchored status must be either integer\
                 value 0 for unachored, or 1 for anchored'
            raise ValueError(msg)

        if tank_location == 'AG' and tank_material == 'C':
            msg = (
                f'The tank {asset_name} is Above Ground (i.e., AG), '
                'but the material type is Concrete ("C"). '
                'Tank type "C" is not defined for AG tanks. '
                'The tank is assumed to be Steel ("S").'
            )

            print(msg)
            tank_material = 'S'

        if tank_location == 'AG' and tank_material == 'W':
            msg = (
                f'The tank {asset_name} is Above Ground (i.e., AG), but'
                ' the material type is Wood ("W"). '
                'Tank type "W" is not defined for AG tanks. '
                'The tank is assumed to be Steel ("S").'
            )

            print(msg)
            tank_material = 'S'

        if tank_location == 'B' and tank_material == 'S':
            msg = (
                f'The tank {asset_name} is buried (i.e., B), but the '
                'material type is Steel ("S"). Tank type "S" is '
                'not defined for "B" tanks. '
                'The tank is assumed to be Concrete ("C").'
            )

            print(msg)
            tank_material = 'C'

        if tank_location == 'B' and tank_material == 'W':
            msg = (
                f'The tank {asset_name} is buried (i.e., B), but the'
                'material type is Wood ("W"). Tank type "W" is '
                'not defined for B tanks. The tank is assumed '
                'to be Concrete ("C")'
            )

            print(msg)
            tank_material = 'C'

        if tank_anchored == 1:
            # Since anchore status does nto matter, there is no need to
            # print a warning
            tank_anchored = 0

        cur_tank_cmp_line = tank_cmp_lines[
            tank_location, tank_material, tank_anchored
        ]

        comp = pd.DataFrame(
            cur_tank_cmp_line,
            index=['Units', 'Location', 'Direction', 'Theta_0', 'Family'],
        ).T

        dl_ap = {
            'Asset': {
                'ComponentAssignmentFile': 'CMP_QNT.csv',
                'ComponentDatabase': 'Hazus Earthquake - Potable Water',
                'Material': tank_material,
                'Location': tank_location,
                'Anchored': tank_anchored,
                'PlanArea': '1',  # Sina: does not make sense for water.
                # Kept it here since it was also kept here for
                # Transportation
            },
            'Demands': {},
        }

    else:
        print(
            f'Water Distribution network element type {wdn_element_type} '
            f'is not supported in Hazus Earthquake - Potable Water'
        )
        dl_ap = 'N/A'
        comp = None


    return gi_ap, dl_ap, comp
