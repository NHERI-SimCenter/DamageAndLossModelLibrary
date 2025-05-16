# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Leland Stanford Junior University
# Copyright (c) 2018 The Regents of the University of California
#
# This file is part of the SimCenter Backend Applications
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
# this file. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Adam Zsarnóczay
# Kuanshi Zhong
#
# Based on rulesets developed by:
# Karen Angeles
# Meredith Lockhead
# Tracy Kijewski-Correa

import numpy as np

def FL_config(BIM):
    """
    Rules to identify the flood vunerability category

    Parameters
    ----------
    BIM: dictionary
        Information about the building characteristics.

    Returns
    -------
    config: str
        A string that identifies a specific configration within this buidling
        class.
    """
    year = BIM['YearBuilt'] # just for the sake of brevity

    # Flood Type
    if BIM['FloodZone'] == 'AO':
        flood_type = 'raz' # Riverine/A-Zone
    elif BIM['FloodZone'] in ['A', 'AE', 'AH']:
        flood_type = 'caz' # Costal-Zone A
    elif BIM['FloodZone'].startswith('V'):
        flood_type = 'cvz' # Costal-Zone V
    else:
        flood_type = 'caz' # Default

    # First Floor Elevation (FFE)
    # For A Zone, top of finished floor; 
    # for V Zone, bottom of floor beam of lowest floor; 
    # define X based on typical depth of girders assuming bottom of door is used to 
    # estimate first floor ht 
    # (https://www.apawood.org/Data/Sites/1/documents/raised-wood-floor-foundations-guide.pdf) 
    # -- take X=1 ft as average value of different options (depths)
    if flood_type in ['raz', 'caz']:
        FFE = BIM['FirstFloorElevation']
    else:
        FFE = BIM['FirstFloorElevation'] - 1.0

    # PostFIRM
    #Based on FEMA FLOOD INSURANCE STUDY NUMBER 34001CV000A (Atlantic County, NJ )
    # Version Number 2.1.1.1 (See Table 9)
    # Yes=Post-FIRM, No=Pre-FIRM
    PostFIRM_year_by_city = {
        'Absecon': 1976,
        'Atlantic': 1971,
        'Brigantine': 1971,
        'Buena': 1983,
        'Buena Vista': 1979,
        'Corbin City': 1981,
        'Egg Harbor City': 1982,
        'Egg Harbor': 1983,
        'Estell Manor': 1978,
        'Folsom': 1982,
        'Galloway': 1983,
        'Hamilton': 1977,
        'Hammonton': 1982,
        'Linwood': 1983,
        'Longport': 1974,
        'Margate City': 1974,
        'Mullica': 1982,
        'Northfield': 1979,
        'Pleasantville': 1983,
        'Port Republic': 1983,
        'Somers Point': 1982,
        'Ventnor City': 1971,
        'Weymouth':1979 
    }
    if BIM['City'] in PostFIRM_year_by_city:
        PostFIRM_year = PostFIRM_year_by_city[BIM['City']]
        PostFIRM = year > PostFIRM_year
    else:
        PostFIRM = False

    # Basement Type
    if BIM['SplitLevel'] and (BIM['FoundationType'] == 3504):
        basement_type = 'spt' # Split-Level Basement
    elif BIM['FoundationType'] in [3501, 3502, 3503, 3505, 3506, 3507]:
        basement_type = 'bn' # No Basement
    elif (not BIM['SplitLevel']) and (BIM['FoundationType'] == 3504):
        basement_type = 'bw' # Basement
    else:
        basement_type = 'bw' # Default

    # Duration
    # The New Orleans District has developed expert opinion damage functions for
    # the flood control feasibility study in Jefferson and Orleans Parishes (GEC,
    # 1996), and for the Lower Atchafalaya Re-evaluation (GEC, 1997). Depth-
    # damage functions include residential and non-residential structure and
    # contents damage for four types of flooding:
    # • Hurricane flooding, long duration (one week), salt water
    # • Hurricane flooding, short duration (one day), salt water
    # So everything we do in NJ is short duration according to the damage curves.
    dur = 'short'

    # Occupancy Type
    if BIM['OccupancyClass'] == 'RES1':
        if BIM['NumberOfStories'] == 1:
            if flood_type == 'raz':
                occupancy_type = 'SF1XA'
            elif flood_type == 'cvz':
                occupancy_type = 'SF1XV'
        else:
            if basement_type == 'nav':
                if flood_type == 'raz':
                    occupancy_type = 'SF2XA'
                elif flood_type == 'cvz':
                    occupancy_type = 'SF2XV'
            elif basement_type == 'bmt':
                if flood_type == 'raz':
                    occupancy_type = 'SF2BA'
                elif flood_type == 'cvz':
                    occupancy_type = 'SF2BV'
            elif basement_type == 'spt':
                if flood_type == 'raz':
                    occupancy_type = 'SF2SA'
                elif flood_type == 'cvz':
                    occupancy_type = 'SF2SV'
    elif 'RES3' in BIM['OccupancyClass']:
        occupancy_type = 'APT'
    else:
        ap_ot = {
            'RES2': 'MH',
            'RES4': 'HOT',
            'RES5': 'NURSE',
            'RES6': 'NURSE',
            'COM1': 'RETAL',
            'COM2': 'WHOLE',
            'COM3': 'SERVICE',
            'COM4': 'OFFICE',
            'COM5': 'BANK',
            'COM6': 'HOSP',
            'COM7': 'MED',
            'COM8': 'REC',
            'COM9': 'THEAT',
            'COM10': 'GARAGE',
            'IND1': 'INDH',
            'IND2': 'INDL',
            'IND3': 'CHEM',
            'IND4': 'PROC',
            'IND5': 'CHEM',
            'IND6': 'CONST',
            'AGR1': 'AGRI',
            'REL1': 'RELIG',
            'GOV1': 'CITY',
            'GOV2': 'EMERG',
            'EDU1': 'SCHOOL',
            'EDU2': 'SCHOOL'
        }
        occupancy_type = ap_ot[BIM['OccupancyClass']]


    fl_config = None
    if BIM['OccupancyClass'] == 'RES1':
        if flood_type == 'raz':
            if BIM['SplitLevel']:
                if basement_type == 'bn':
                    fl_config = 'structural.111.RES1.FIA.split_level.no_basement.a_zone'
                else:
                    fl_config = 'structural.112.RES1.FIA_Modified.split_level.with_basement.a_zone'

            elif BIM['NumberOfStories'] == 1:
                if basement_type == 'bn':
                    fl_config = 'structural.129.RES1.USACE_IWR.one_story.no_basement'
                else:
                    fl_config = 'structural.704.RES1.BCAR_Jan_2011.one_story.with_basement.b14'

            elif BIM['NumberOfStories'] == 2:
                if basement_type == 'bn':
                    fl_config = 'structural.107.RES1.FIA.two_floors.no_basement.a_zone'
                else:
                    fl_config = 'structural.108.RES1.FIA_Modified.two_floors.with_basement.a_zone'

            elif BIM['NumberOfStories'] == 3:
                if basement_type == 'bn':
                    fl_config = 'structural.109.RES1.FIA.three_or_more_floors.no_basement.a_zone'
                else:
                    fl_config = 'structural.110.RES1.FIA_Modified.three_or_more_floors.with_basement.a_zone'

        elif flood_type == 'cvz':
            if BIM['SplitLevel']:
                if basement_type == 'bn':
                    fl_config = 'structural.658.RES1.BCAR_Jan_2011.all_floors.slab_no_basement.coastal_a_or_v_zone'
                else:
                    fl_config = 'structural.120.RES1.FIA_Modified.split_level.with_basement.v_zone'

            elif BIM['NumberOfStories'] == 1:
                if basement_type == 'bn':
                    fl_config = 'structural.658.RES1.BCAR_Jan_2011.all_floors.slab_no_basement.coastal_a_or_v_zone'
                else:
                    fl_config = 'structural.114.RES1.FIA_Modified.one_floor.with_basement.v_zone'

            elif BIM['NumberOfStories'] == 2:
                if basement_type == 'bn':
                    fl_config = 'structural.115.RES1.FIA.two_floors.no_basement.v_zone'
                else:
                    fl_config = 'structural.116.RES1.FIA_Modified.two_floors.with_basement.v_zone'

            elif BIM['NumberOfStories'] == 3:
                if basement_type == 'bn':
                    fl_config = 'structural.117.RES1.FIA.three_or_more_floors.no_basement.v_zone'
                else:
                    fl_config = 'structural.118.RES1.FIA_Modified.three_or_more_floors.with_basement.v_zone'

        elif flood_type == 'caz':
            if BIM['SplitLevel']:
                if basement_type == 'bn':
                    # copied from Coastal V zone as per Hazus guidelines
                    fl_config = 'structural.658.RES1.BCAR_Jan_2011.all_floors.slab_no_basement.coastal_a_or_v_zone'
                else:
                    fl_config = 'structural.112.RES1.FIA_Modified.split_level.with_basement.a_zone'

            elif BIM['NumberOfStories'] == 1:
                if basement_type == 'bn':
                    # copied from Coastal V zone as per Hazus guidelines
                    fl_config = 'structural.658.RES1.BCAR_Jan_2011.all_floors.slab_no_basement.coastal_a_or_v_zone'
                else:
                    # copied from Coastal V zone as per Hazus guidelines
                    fl_config = 'structural.114.RES1.FIA_Modified.one_floor.with_basement.v_zone'

            elif BIM['NumberOfStories'] == 2:
                if basement_type == 'bn':
                    fl_config = 'structural.107.RES1.FIA.two_floors.no_basement.a_zone'
                else:
                    fl_config = 'structural.108.RES1.FIA_Modified.two_floors.with_basement.a_zone'

            elif BIM['NumberOfStories'] == 3:
                if basement_type == 'bn':
                    fl_config = 'structural.109.RES1.FIA.three_or_more_floors.no_basement.a_zone'
                else:
                    fl_config = 'structural.110.RES1.FIA_Modified.three_or_more_floors.with_basement.a_zone'


    elif BIM['OccupancyClass'] == 'RES2':
        if BIM['NumberOfStories'] == 1:
            if flood_type == 'rvz':
                fl_config = 'structural.189.RES2.FIA.mobile_home.a_zone'

            elif flood_type == 'cvz':
                if basement_type == 'bn':
                    fl_config = 'structural.667.RES2.BCAR_Jan_2011.manufactured_home_mobile.coastal_a_or_v_zone'
                else:
                    fl_config = 'structural.190.RES2.FIA.mobile_home.v_zone'

            elif flood_type == 'caz':
                fl_config = 'structural.189.RES2.FIA.mobile_home.a_zone'

    elif 'RES3' in BIM['OccupancyClass']:

        # the following rules are used for all flood-types as a default and replaced with a
        # more appropriate one if possible
        if basement_type =='bn':
            fl_config = 'structural.204.RES3.USACE_Chicago.apartment_unit_grade'
        else:
            fl_config = 'structural.205.RES3.USACE_Chicago.apartment_unit_sub_grade'

        if flood_type == 'cvz':
            if BIM['NumberOfStories'] in [1, 2]:
                if basement_type == 'bn':
                    if BIM['OccupancyClass'] == 'RES3A':
                        fl_config = 'structural.659.RES3A.BCAR_Jan_2011.1to2_stories.slab_no_basement.coastal_a_or_v_zone'
                    elif BIM['OccupancyClass'] == 'RES3B':
                        fl_config = 'structural.660.RES3B.BCAR_Jan_2011.1to2_stories.slab_no_basement.coastal_a_or_v_zone'

    # the following rules are used for all flood-types as a default
    elif BIM['OccupancyClass'] == 'RES4':
        fl_config = 'structural.209.RES4.USACE_Galveston.average_hotel_&_motel'

    elif BIM['OccupancyClass'] == 'RES5':
        fl_config = 'structural.214.RES5.USACE_Galveston.average_institutional_dormitory'

    elif BIM['OccupancyClass'] == 'RES6':
        fl_config = 'structural.215.RES6.USACE_Galveston.nursing_home'

    elif BIM['OccupancyClass'] == 'COM1':
        fl_config = 'structural.217.COM1.USACE_Galveston.average_retail'

    elif BIM['OccupancyClass'] == 'COM2':
        fl_config = 'structural.341.COM2.USACE_Galveston.average_wholesale'

    elif BIM['OccupancyClass'] == 'COM3':
        fl_config = 'structural.375.COM3.USACE_Galveston.average_personal_&_repair_services'

    elif BIM['OccupancyClass'] == 'COM4':
        fl_config = 'structural.431.COM4.USACE_Galveston.average_prof/tech_services'

    elif BIM['OccupancyClass'] == 'COM5':
        fl_config = 'structural.467.COM5.USACE_Galveston.bank'

    elif BIM['OccupancyClass'] == 'COM6':
        fl_config = 'structural.474.COM6.USACE_Galveston.hospital'

    elif BIM['OccupancyClass'] == 'COM7':
        fl_config = 'structural.475.COM7.USACE_Galveston.average_medical_office'

    elif BIM['OccupancyClass'] == 'COM8':
        fl_config = 'structural.493.COM8.USACE_Galveston.average_entertainment/recreation'

    elif BIM['OccupancyClass'] == 'COM9':
        fl_config = 'structural.532.COM9.USACE_Galveston.average_theatre'

    elif BIM['OccupancyClass'] == 'COM10':
        fl_config = 'structural.543.COM10.USACE_Galveston.garage'

    elif BIM['OccupancyClass'] == 'IND1':
        fl_config = 'structural.545.IND1.USACE_Galveston.average_heavy_industrial'

    elif BIM['OccupancyClass'] == 'IND2':
        fl_config = 'structural.559.IND2.USACE_Galveston.average_light_industrial'

    elif BIM['OccupancyClass'] == 'IND3':
        fl_config = 'structural.575.IND3.USACE_Galveston.average_food/drug/chem'

    elif BIM['OccupancyClass'] == 'IND4':
        fl_config = 'structural.586.IND4.USACE_Galveston.average_metals/minerals_processing'

    elif BIM['OccupancyClass'] == 'IND5':
        fl_config = 'structural.591.IND5.USACE_Galveston.average_high_technology'

    elif BIM['OccupancyClass'] == 'IND6':
        fl_config = 'structural.592.IND6.USACE_Galveston.average_construction'

    elif BIM['OccupancyClass'] == 'AGR1':
        fl_config = 'structural.616.AGR1.USACE_Galveston.average_agriculture'

    elif BIM['OccupancyClass'] == 'REL1':
        fl_config = 'structural.624.REL1.USACE_Galveston.church'

    elif BIM['OccupancyClass'] == 'GOV1':
        fl_config = 'structural.631.GOV1.USACE_Galveston.average_government_services'

    elif BIM['OccupancyClass'] == 'GOV2':
        fl_config = 'structural.640.GOV2.USACE_Galveston.average_emergency_response'

    elif BIM['OccupancyClass'] == 'EDU1':
        fl_config = 'structural.643.EDU1.USACE_Galveston.average_school'

    elif BIM['OccupancyClass'] == 'EDU2':
        fl_config = 'structural.652.EDU2.USACE_Galveston.average_college/university'

    # extend the BIM dictionary
    BIM.update(dict(
        FloodType = flood_type,
        BasementType=basement_type,
        PostFIRM=PostFIRM,
        ))

    return fl_config

