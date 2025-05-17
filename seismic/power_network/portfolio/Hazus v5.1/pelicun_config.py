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

    # initialize the auto-populated GI
    power_asset_type = gi_ap.get('type', 'MISSING')
    asset_name = gi_ap.get('AIM_id', None)

    if power_asset_type == 'Substation':
        ep_s_size = ''
        ep_s_anchored = ''
        substation_voltage = gi_ap.get('Voltage', None)
        if substation_voltage is None:
            msg = (
                'Substation feature "Voltage" is missing. '
                f' substation "{asset_name}" assumed to be '
                '"  Low Voltage".'
            )
            print(msg)
            substation_voltage = 'low'

        if isinstance(substation_voltage, str):
            if substation_voltage.lower() == 'low':
                ep_s_size = 'L'
            elif substation_voltage.lower() == 'medium':
                ep_s_size = 'M'
            elif substation_voltage.lower() == 'high':
                ep_s_size = 'H'
            else:
                msg = (
                    'substation Voltage value is = '
                    f'{substation_voltage}. '
                    'The value must be either "low" '
                    ', " medium", or " high".'
                )
                raise ValueError(msg)

        elif isinstance(substation_voltage, (float, int)):
            # Substation Voltage unit is kV. Any number smaller than
            # 34 kV is not supported by HAZUS methodlogy. Furthermore,
            # values significantly larger may refer to a voltage value in
            # different unit. The upper bound value is set ro 1200 kV.

            if substation_voltage < 34:
                msg = (
                    f'The subtation Voltage for asset "{asset_name}" '
                    f'is too low({substation_voltage}). The current '
                    'methodology support voltage between 34 kV and 1200'
                    ' kV. Please make sure that the units are in kV.'
                )
                raise ValueError(msg)

            if substation_voltage > 1200:
                msg = (
                    f'The subtation Voltage for asset "{asset_name}"'
                    f'is too high({substation_voltage}). The current '
                    'methodology support voltage between 34 kV and 1200'
                    ' kV. Please make sure that the units are in kV.'
                )
                raise ValueError(msg)

            if substation_voltage <= 150:
                ep_s_size = 'L'
            elif substation_voltage <= 230:
                ep_s_size = 'M'
            elif substation_voltage >= 500:
                ep_s_size = 'H'
            else:
                msg = (
                    'This should never have happed. Please '
                    'report this to the developer(SimCenter)'
                    f'. (Value = {substation_voltage}).'
                )
                raise RuntimeError(msg)
        else:
            msg = (
                'substation Voltage value is = '
                f'{substation_voltage}. It should be '
                'string or a number. For more information, '
                'refer to the documentation please.'
            )
            raise ValueError(msg)

        substation_anchored = gi_ap.get('Anchored', None)

        if substation_anchored is None:
            print(
                'Substation feature "Anchored" is missing. '
                f' substation "{asset_name}" assumed to be '
                '"  Unanchored".'
            )

            substation_anchored = False

        if isinstance(substation_anchored, str):
            if substation_anchored.lower() in [
                'a',
                'anchored',
                'yes',
                'true',
                'positive',
                '1',
            ]:
                ep_s_anchored = 'A'
            elif substation_anchored.lower() in [
                'u',
                'unanchored',
                'no',
                'false',
                'negative',
                '0',
            ]:
                ep_s_anchored = 'U'
        elif isinstance(substation_anchored, (bool, int, float)):
            if abs(substation_anchored - True) < 0.001:
                ep_s_anchored = 'A'
            elif abs(substation_anchored) < 0.001:
                ep_s_anchored = 'U'
            else:
                msg = (
                    'This should never have happed. Please '
                    'report this to the developer(SimCenter)'
                    f'. (Value = {substation_anchored}).'
                )
                raise RuntimeError(msg)

        if ep_s_anchored is None:
            msg = (
                'Substation anchored value is = '
                f'{substation_anchored}. It should be '
                'string, boolean, or a number representing '
                'True or False. For more information, '
                'refer to the documentation please.'
            )
            raise ValueError(msg)

        # Define performance model
        # fmt: off
        substation_type = f'EP.S.{ep_s_size}.{ep_s_anchored}'
        comp = pd.DataFrame(
            {substation_type: ['ea', 1, 1, 1, 'N/A']},
            index=['Units', 'Location', 'Direction', 'Theta_0', 'Family']
        ).T

        # Define the auto-populated config
        dl_ap = {
            "Asset": {
                "ComponentAssignmentFile": "CMP_QNT.csv",
                "ComponentDatabase": "Hazus Earthquake - Electric Power",
                "Substation Voltage": ep_s_size,
                "Substation Anchored": ep_s_anchored,
            },
            "Damage": {"DamageProcess": "Hazus Earthquake"},
            "Demands": {},
            "Losses": {},
        }

    elif power_asset_type == 'Circuit':
        circuit_anchored = gi_ap.get('Anchored', None)

        ep_c_anchored = None
        if circuit_anchored is None:
            print(
                'Circuit feature "Anchored" is missing. '
                f' Circuit "{asset_name}" assumed to be '
                '"  Unanchored".'
            )

            circuit_anchored = False

        if isinstance(circuit_anchored, str):
            if circuit_anchored.lower() in [
                'a',
                'anchored',
                'yes',
                'true',
                'positive',
                '1',
            ]:
                ep_c_anchored = 'A'
            elif circuit_anchored.lower() in [
                'u',
                'unanchored',
                'no',
                'false',
                'negative',
                '0',
            ]:
                ep_c_anchored = 'U'
        elif isinstance(circuit_anchored, (bool, int, float)):
            if abs(circuit_anchored - True) < 0.001:
                ep_c_anchored = 'A'
            elif abs(circuit_anchored) < 0.001:
                ep_c_anchored = 'U'
            else:
                msg = (
                    'This should never have happed. Please '
                    'report this to the developer(SimCenter)'
                    f'. (Value = {circuit_anchored}).'
                )
                raise RuntimeError(msg)

        if ep_c_anchored is None:
            msg = (
                'Circuit anchored value is = '
                f'{circuit_anchored}. It should be '
                'string, boolean, or a number representing '
                'True or False. For more information, '
                'refer to the documentation please.'
            )
            raise ValueError(msg)

        # Define performance model
        # fmt: off
        circuit_type = f'EP.C.{ep_c_anchored}'
        comp = pd.DataFrame(
            {circuit_type: ['ea', 1, 1, 1, 'N/A']},
            index=['Units', 'Location', 'Direction', 'Theta_0', 'Family']
        ).T

        # Define the auto-populated config
        dl_ap = {
            "Asset": {
                "ComponentAssignmentFile": "CMP_QNT.csv",
                "ComponentDatabase": "Hazus Earthquake - Electric Power",
                "Circuit Anchored": ep_c_anchored,
            },
            "Damage": {"DamageProcess": "Hazus Earthquake"},
            "Demands": {},
            "Losses": {},
        }

    elif power_asset_type == 'Generation':
        ep_g_size = ''
        generation_output = gi_ap.get('Output', None)
        if generation_output is None:
            msg = (
                'Generation feature "Output" is missing. '
                f' Generation "{asset_name}" assumed to be '
                '"Small".'
            )
            print(msg)
            # if the power feature is missing, the generation is assumed
            # to be small
            ep_g_size = 'small'

        if isinstance(generation_output, str):
            generation_output = generation_output.lower()
            generation_output = generation_output.strip()
            acceptable_power_unit = ('w', 'kw', 'mw', 'gw')

            units_exist = [
                unit in generation_output for unit in acceptable_power_unit
            ]

            power_unit = None

            if True in units_exist:
                power_unit = acceptable_power_unit[units_exist.index(True)]

                if generation_output.endswith(power_unit):
                    generation_output = generation_output.strip(power_unit)
                    generation_output = generation_output.strip()
            else:
                msg = (
                    "Generation feature doesn't have a unit for "
                    '"Output" value. The unit for Generation '
                    f'"{asset_name}"  is assumed to be "MW".'
                )
                print(msg)

                power_unit = 'mw'

            try:
                generation_output = float(generation_output)

                if power_unit == 'w':
                    generation_output = generation_output / 10**6
                elif power_unit == 'kw':
                    generation_output = generation_output / 10**3
                elif power_unit == 'mw':
                    # just for the sake of completeness, we don't
                    # need to convert here, since MW is our base unit
                    pass
                elif power_unit == 'gw':
                    generation_output = generation_output * 1000

                if generation_output < 200:
                    ep_g_size = 'small'
                elif 200 < generation_output < 500:
                    ep_g_size = 'medium'
                else:
                    ep_g_size = 'large'

            except ValueError as e:
                # check if the exception is for value not being a float
                not_float_str = 'could not convert string to float:'
                if not str(e).startswith(not_float_str):
                    raise
                # otherwise
                msg = (
                    'Generation feature has an unrecognizable "Output"'
                    f' value. Generation "{asset_name}" = '
                    f'{generation_output}, instead of a numerical '
                    'value. So the size of the Generation is assumed '
                    'to be "Small".'
                )
                print(msg)

                ep_g_size = 'small'

            if ep_g_size == 'small':
                ep_g_size = 'S'
            elif ep_g_size in ('medium', 'large'):
                # because medium and large size generation plants are
                # categorized in the same category.
                ep_g_size = 'ML'
            else:
                msg = (
                    'This should never have happed. Please '
                    'report this to the developer(SimCenter)'
                    f'. (Value = {ep_g_size}).'
                )
                raise ValueError(msg)

        generation_anchored = gi_ap.get('Anchored', None)

        if generation_anchored is None:
            msg = (
                'Generation feature "Anchored" is missing. '
                f' Circuit "{asset_name}" assumed to be '
                '"  Unanchored".'
            )
            print(msg)

            generation_anchored = False

        ep_g_anchored = None
        if isinstance(generation_anchored, str):
            if generation_anchored.lower() in [
                'a',
                'anchored',
                'yes',
                'true',
                'positive',
                '1',
            ]:
                ep_g_anchored = 'A'
            elif generation_anchored.lower() in [
                'u',
                'unanchored',
                'no',
                'false',
                'negative',
                '0',
            ]:
                ep_g_anchored = 'U'
        elif isinstance(generation_anchored, (bool, int, float)):
            if abs(generation_anchored - True) < 0.001:
                ep_g_anchored = 'A'
            elif abs(generation_anchored) < 0.001:
                ep_g_anchored = 'U'
            else:
                msg = (
                    'This should never have happed. Please '
                    'report this to the developer(SimCenter)'
                    f'. (Value = {generation_anchored}).'
                )
                raise RuntimeError(msg)

        if ep_g_anchored is None:
            msg = (
                'Circuit anchored value is = '
                f'{circuit_anchored}. It should be '
                'string, boolean, or a number representing '
                'True or False. For more information, '
                'refer to the documentation please.'
            )
            raise ValueError(msg)

        # Define performance model
        # fmt: off
        generation_type = f'EP.G.{ep_g_size}.{ep_g_anchored}'
        comp = pd.DataFrame(
            {generation_type: ['ea', 1, 1, 1, 'N/A']},
            index=['Units', 'Location', 'Direction', 'Theta_0', 'Family']
        ).T

        # Define the auto-populated config
        dl_ap = {
            "Asset": {
                "ComponentAssignmentFile": "CMP_QNT.csv",
                "ComponentDatabase": "Hazus Earthquake - Electric Power",
                "Generation Size": ep_g_size,
                "Generation Anchored": ep_g_anchored,
            },
            "Damage": {"DamageProcess": "Hazus Earthquake"},
            "Demands": {},
            "Losses": {},
        }

    return gi_ap, dl_ap, comp
