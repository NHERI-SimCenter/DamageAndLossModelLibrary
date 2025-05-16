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
from jsonschema import validate
import pandas as pd


def get_feature(feature, building_info):
    """
    Parse and map building characteristics into features used in Hazus.

    Parameters
    ----------
    feature: string
        Name of feature to be mapped.

    """

    if feature == 'building_type':
        building_type_map = {
            "Wood": "W", 
            "Masonry": "M", 
            "Concrete": "C", 
            "Steel": "S",
            "Manufactured Housing": "MH",
            "Essential Facility": "HUEF"
        }
        return building_type_map[building_info["BuildingType"]]

    elif feature == 'structure_type':
        structure_type_map = {
            "Single Family Housing": "SF",
            "Multi-Unit Housing": "MUH",
            "Low-Rise Strip Mall": "LRM",
            "Low-Rise Industrial Building": "LRI",
            "Engineered Residential Building": "ERB",
            "Engineered Commercial Building": "ECB",
            "Pre-Engineered Metal Building": "PMB",
            "Pre-HUD": "PHUD",
            "1976 HUD": "76HUD",
            "1994 HUD Zone 1": "94HUDI",
            "1994 HUD Zone 2": "94HUDII",
            "1994 HUD Zone 3": "94HUDIII",
            "Fire Station": "FS",
            "Police Station": "PS",
            "Emergency Operation Center": "EO",
            "Hospital": "H",
            "School": "S"
        }
        return structure_type_map[building_info["StructureType"]]

    elif feature == 'height_class':

        structure_type = get_feature('structure_type', building_info)

        if structure_type in ['SF', 'MUH', 'ERB', 'ECB', 'S', 'H']:
            number_of_stories = int(building_info["NumberOfStories"])

        if structure_type in ["LRM"]:
            height = float(building_info["Height"])

        if structure_type in ['PMB']:
            plan_area = float(building_info["PlanArea"])

        if structure_type == 'SF':
            if number_of_stories == 1:
                return "1" # 1 Story
            else:
                return "2" # 2 or more Stories

        elif structure_type == 'MUH':
            if number_of_stories == 1:
                return "1" # 1 Story
            elif number_of_stories == 2:
                return "2" # 2 Stories
            else:
                return "3" # 3 or more Stories

        elif structure_type == 'LRM':
            if height <= 15.0:
                return "1" # Up to 15 ft high
            else:
                return "2" # More than 15 ft high

        elif structure_type in ['ERB', 'ECB']:
            if number_of_stories in [1, 2]:
                return "L" # 1-2 Stories
            elif number_of_stories in [3, 4, 5]:
                return "M" # 3-5 Stories
            else:
                return "H" # 6 or more Stories

        elif structure_type == 'S':
            if number_of_stories == 1:
                return "S" # Small
            elif number_of_stories == 2:
                return "M" # Medium
            else:
                return "L" # Large

        elif structure_type == 'H':
            if number_of_stories <= 2:
                return "S" # Small
            elif number_of_stories <= 6:
                return "M" # Medium
            else:
                return "L" # Large

        elif structure_type == 'PMB':
            if plan_area < 10000:
                return "S" # Small, less than 10,000 ft2
            elif plan_area < 100000:
                return "M" # Medium, less than 100,000 ft2
            else:
                return "L" # Large

    elif feature == 'roof_shape':
        roof_shape_map = {
            "Hip": "hip",
            "Gable": "gab",
            "Flat": "flt"
        }
        return roof_shape_map[building_info.get("RoofShape")]

    elif feature == 'roof_cover':
        roof_cover_map = {
            "Single-Ply Membrane": "spm",
            "Built-Up Roof": "bur",
            "Sheet Metal": "smtl",
            "Composite Shingle": "cshl"
        }
        return roof_cover_map[building_info.get("RoofCover")]

    elif feature == 'roof_quality':
        roof_quality_map = {
            "Good": "god",
            "Poor": "por",
        }
        return roof_quality_map[building_info.get("RoofQuality")]

    elif feature == 'roof_system':
        roof_system_map = {
            "Truss": 'trs',
            "Open-Web Steel Joists": 'ows'
        }
        return roof_system_map[building_info.get("RoofSystem")]

    elif feature == 'roof_deck_attachment':
        roof_deck_attachment_map = {
            '6d': '6d',
            '6s': '6s',
            '8d': '8d',
            '8s': '8s',
            'Standard': 'std',
            'Superior': 'sup'
        }
        return roof_deck_attachment_map[building_info.get("RoofDeckAttachment")]        

    elif feature == 'roof_wall_connection':
        roof_wall_connection_map = {
            "Strap": "strap",
            "Toe-nail": "tnail"
        }
        return roof_wall_connection_map[building_info.get("RoofToWallConnection")]

    elif feature == 'secondary_water_resistance':
        return int(building_info.get("SecondaryWaterResistance"))

    elif feature == 'shutters':
        return int(building_info.get("Shutters"))

    elif feature == 'garage':
        garage_map = {
            "No": "no",
            "Standard": "std",
            "Weak": "wkd",
            "Superior": "sup"
        }
        return garage_map[building_info.get("Garage")]

    elif feature == 'wind_debris_class':
        return building_info.get("WindDebrisClass")

    elif feature == 'unit_class':
        number_of_units = building_info.get("NumberOfUnits")

        if number_of_units == 1:
            return "sgl"
        else:
            return "mlt"

    elif feature == 'joist_spacing':
        return building_info.get("JoistSpacing")

    elif feature == 'masonry_reinforcing':
        return int(building_info.get("MasonryReinforcing"))

    elif feature == 'tie_downs':
        return int(building_info.get("TieDowns"))

    elif feature == 'window_area':
        window_area = building_info.get("WindowArea")
        if window_area < 0.33:
            return 'low' # Low
        elif window_area < 0.5:
            return 'med' # Medium
        else:
            return 'hig' # High

    elif feature == 'terrain_roughness':
        terrain_roughness_map = {
            "Open": 3,
            "Light Suburban": 15,
            "Suburban": 35,
            "Light Trees": 70,
            "Trees": 100
        }
        return terrain_roughness_map[building_info.get("LandCover")]

    return "null"

def auto_populate(aim):
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
    gi = aim.get("GeneralInformation")

    # make sure missing data is properly represented as null in the JSON
    for key, item in gi.items():
        if pd.isna(item) or item=="":
            gi[key] = None

    # load the schema assuming it is called "input_schema.json" and it is
    # stored next to the mapping script
    current_file_path = Path(__file__)
    current_directory = current_file_path.parent

    with Path(current_directory / "input_schema.json").open(encoding="utf-8") as f:
        input_schema = json.load(f)

    # validate the provided features against the required inputs
    try:
        validate(instance=gi, schema=input_schema)
    except jsonschema.exceptions.ValidationError as exc:  # type: ignore
        msg = (
            "The provided building information does not conform to the input"
            " requirements for the chosen damage and loss model."
        )

        raise ValueError(msg) from exc

    model_id = ".".join([get_feature(feature, gi) for feature in ['building_type','structure_type']])

    if model_id == 'W.SF':
        model_features = [
            "height_class",
            "roof_shape", 
            "secondary_water_resistance", 
            "roof_deck_attachment", 
            "roof_wall_connection",
            "garage",
            "shutters",
            "terrain_roughness"
        ]

    elif model_id == 'W.MUH':
        roof_shape = get_feature("roof_shape", gi)

        if roof_shape == 'flt':
            model_features = [
                "height_class",
                "roof_shape",
                "roof_cover",
                "roof_quality",
                "null",
                "roof_deck_attachment",
                "roof_wall_connection",
                "shutters",
                "terrain_roughness"
            ]

        else:
            model_features = [
                "height_class",
                "roof_shape",
                "null",
                "null",
                "secondary_water_resistance",
                "roof_deck_attachment",
                "roof_wall_connection",
                "shutters",
                "terrain_roughness"
            ]

    elif model_id == 'M.SF':
        roof_system = get_feature("roof_system", gi)

        if roof_system == 'trs':
            model_features = [
                "height_class",
                "roof_shape",
                "roof_wall_connection",
                "roof_system",
                "roof_deck_attachment",
                "shutters",
                "secondary_water_resistance",
                "garage",
                "masonry_reinforcing",
                "null",
                "terrain_roughness"
            ]

        else:
            roof_cover = get_feature("roof_cover", gi)
            roof_deck_attachment = get_feature("roof_deck_attachment", gi)

            if roof_cover == 'cshl' and roof_deck_attachment == 'su':
                model_features = [
                    "height_class",
                    "roof_shape",
                    "roof_wall_connection",
                    "roof_system",
                    "roof_deck_attachment",
                    "shutters",
                    "secondary_water_resistance",
                    "null",
                    "null",
                    "roof_cover",
                    "terrain_roughness"
                ]

            else:
                model_features = [
                    "height_class",
                    "roof_shape",
                    "roof_wall_connection",
                    "roof_system",
                    "roof_deck_attachment",
                    "shutters",
                    "null",
                    "null",
                    "null",
                    "roof_cover",
                    "terrain_roughness"
                ]
    
    elif model_id == 'M.MUH':
        roof_shape = get_feature("roof_shape", gi)

        if roof_shape == 'flt':
            model_features = [
                "height_class",
                "roof_shape",
                "null",
                "roof_cover",
                "roof_quality",
                "roof_deck_attachment",
                "roof_wall_connection",
                "shutters",
                "masonry_reinforcing",
                "terrain_roughness"
            ]

        else:
            model_features = [
                "height_class",
                "roof_shape",
                "secondary_water_resistance",
                "null",
                "null",
                "roof_deck_attachment",
                "roof_wall_connection",
                "shutters",
                "masonry_reinforcing",
                "terrain_roughness"
            ]
    
    elif model_id.startswith('M.LRM'):
        height_class = get_feature("height_class", gi)
        roof_system = get_feature("roof_system", gi)

        if height_class == '1':
            if roof_system == 'trs':
                model_features = [
                    "height_class",
                    "roof_cover",
                    "shutters",
                    "masonry_reinforcing",
                    "wind_debris_class",
                    "roof_system",
                    "roof_deck_attachment",
                    "roof_wall_connection",
                    "null",
                    "null",
                    "terrain_roughness"
                ]

            else:
                model_features = [
                    "height_class",
                    "roof_cover",
                    "shutters",
                    "masonry_reinforcing",
                    "wind_debris_class",
                    "roof_system",
                    "null",
                    "null",
                    "roof_quality",
                    "roof_deck_attachment",
                    "terrain_roughness"
                ]

        else:
            if roof_system == 'trs':
                model_features = [
                    "height_class",
                    "roof_cover",
                    "shutters",
                    "masonry_reinforcing",
                    "wind_debris_class",
                    "roof_system",
                    "roof_deck_attachment",
                    "roof_wall_connection",
                    "null",
                    "null",
                    "null",
                    "null",
                    "terrain_roughness"
                ] 

            else:
                unit_class = get_feature("unit_class", gi)

                if unit_class == 'sgl':
                    model_features = [
                        "height_class",
                        "roof_cover",
                        "shutters",
                        "masonry_reinforcing",
                        "wind_debris_class",
                        "roof_system",
                        "null",
                        "null",
                        "roof_quality",
                        "roof_deck_attachment",
                        "unit_class",
                        "null",
                        "terrain_roughness"
                    ]  

                else:
                    model_features = [
                        "height_class",
                        "roof_cover",
                        "shutters",
                        "masonry_reinforcing",
                        "wind_debris_class",
                        "roof_system",
                        "null",
                        "null",
                        "roof_quality",
                        "roof_deck_attachment",
                        "unit_class",
                        "joist_spacing",
                        "terrain_roughness"
                    ]     
    
    elif model_id == 'M.LRI':
        model_features = [
            "shutters",
            "masonry_reinforcing",
            "roof_quality",
            "roof_deck_attachment",
            "terrain_roughness"
        ]
    
    elif model_id.startswith('M.E'):
        model_features = [
            "height_class",
            "roof_cover",
            "shutters",
            "wind_debris_class",
            "roof_deck_attachment",
            "window_area",
            "terrain_roughness"
        ] 
    
    elif model_id.startswith('C.E'):
        model_features = [
            "height_class",
            "roof_cover",
            "shutters",
            "wind_debris_class",
            "window_area",
            "terrain_roughness"
        ]
    
    elif model_id.startswith('S.E'):
        model_features = [
            "height_class",
            "roof_cover",
            "shutters",
            "wind_debris_class",
            "roof_deck_attachment",
            "window_area",
            "terrain_roughness"
        ]
    
    elif model_id == 'S.PMB':
        model_features = [
            "height_class",
            "shutters",
            "roof_quality",
            "roof_deck_attachment",
            "terrain_roughness"
        ]
    
    elif model_id.startswith('MH.'):
        model_features = [
            "shutters",
            "tie_downs",
            "terrain_roughness"
        ]
    
    elif model_id.startswith('HUEF.FS'):
        model_features = [
            "roof_cover",
            "shutters",
            "wind_debris_class",
            "roof_quality",
            "roof_deck_attachment",
            "terrain_roughness"
        ]

    elif model_id.startswith('HUEF.S'):
        height_class = get_feature("height_class", gi)

        if height_class == 'S':
            model_features = [
                "height_class",
                "roof_cover",
                "shutters",
                "wind_debris_class",
                "roof_quality",
                "roof_deck_attachment",
                "terrain_roughness"
            ]

        else:
            model_features = [
                "height_class",
                "roof_cover",
                "shutters",
                "wind_debris_class",
                "null",
                "roof_deck_attachment",
                "terrain_roughness"
            ]
 
    
    elif model_id.startswith('HUEF.H'):
        model_features = [
            "height_class",
            "roof_cover",
            "wind_debris_class",
            "roof_deck_attachment",
            "shutters",
            "terrain_roughness"
        ]
    
    elif model_id.startswith(('HUEF.PS', 'HUEF.EO')):
        model_features = [
            "roof_cover",
            "shutters",
            "wind_debris_class",
            "roof_deck_attachment",
            "window_area",
            "terrain_roughness"
        ]
    
    model_id += "." + ".".join([f"{get_feature(feature, gi)}" for feature in model_features])

    #print("- - - - - - - MODEL ID - - - - - - -")
    #print("current: ", model_id)
    #print("original:", gi.get('Wind_Config',''))
    #print("- - - - - - - MODEL ID - - - - - - -")

    comp = pd.DataFrame(
        {f"{model_id}": ["ea", 1, 1, 1, "N/A"]},  # noqa: E241
        index=["Units", "Location", "Direction", "Theta_0", "Family"],  # noqa: E231, E251
    ).T

    dl_ap = {
        "Asset": {
            "ComponentAssignmentFile": "CMP_QNT.csv",
            "ComponentDatabase": "Hazus Hurricane Wind - Buildings",
            "NumberOfStories": 1,  # there is only one component in a building-level resolution
        },
        "Damage": {"DamageProcess": "Hazus Hurricane"},
        "Demands": {},
        "Losses": {
            "Repair": {
                "ConsequenceDatabase": "Hazus Hurricane Wind - Buildings",
                "MapApproach": "Automatic",
                "DecisionVariables": {
                    "Cost": True,
                    "Carbon": False,
                    "Energy": False,
                    "Time": False
                }
            }
        },
        "Options": {
            "NonDirectionalMultipliers": {"ALL": 1.0},
        },
    }

    return gi, dl_ap, comp
