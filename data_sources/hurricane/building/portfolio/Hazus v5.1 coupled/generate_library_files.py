"""Generates Hazus Hurricane damage and loss database files."""

# This code was written before we began enforcing more strict linting
# standards. pylint warnings are ignored for this file.

# pylint: skip-file

from __future__ import annotations

import json
import shutil
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from tqdm import tqdm

warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', FutureWarning)


def parse_description(descr, parsed_data):  # noqa: C901, PLR0912, PLR0915
    """
    Parse the descr string and store params in row Series.

    Parameters
    ----------
    descr: string
        Hurricane archetype description from Hazus's raw database.
    parsed_data: pd.Series
        A pandas Series to collect parsed data.

    """
    # Roof Shape
    if 'rsflt' in descr:
        parsed_data['roof_shape'] = 'flt'
        descr = descr.replace('rsflt', '')
    elif 'rsgab' in descr:
        parsed_data['roof_shape'] = 'gab'
        descr = descr.replace('rsgab', '')
    elif 'rship' in descr:
        parsed_data['roof_shape'] = 'hip'
        descr = descr.replace('rship', '')

    # Secondary Water Resistance
    if 'swrys' in descr:
        parsed_data['sec_water_res'] = True
        descr = descr.replace('swrys', '')
    elif 'swrno' in descr:
        parsed_data['sec_water_res'] = False
        descr = descr.replace('swrno', '')

    # Roof Deck Attachment
    if 'rda6d' in descr:
        parsed_data['roof_deck_attch'] = '6d'
        descr = descr.replace('rda6d', '')
    elif 'rda8d' in descr:
        parsed_data['roof_deck_attch'] = '8d'
        descr = descr.replace('rda8d', '')
    elif 'rda6s' in descr:
        parsed_data['roof_deck_attch'] = '6s'
        descr = descr.replace('rda6s', '')
    elif 'rda8s' in descr:
        parsed_data['roof_deck_attch'] = '8s'
        descr = descr.replace('rda8s', '')

    # Roof Deck Attachment - Alternative Description
    if 'rdast' in descr:
        parsed_data['roof_deck_attch'] = 'std'  # standard
        descr = descr.replace('rdast', '')
    elif 'rdasu' in descr:
        parsed_data['roof_deck_attch'] = 'sup'  # superior
        descr = descr.replace('rdasu', '')

    # Roof-Wall Connection
    if 'tnail' in descr:
        parsed_data['roof_wall_conn'] = 'tnail'
        descr = descr.replace('tnail', '')
    elif 'strap' in descr:
        parsed_data['roof_wall_conn'] = 'strap'
        descr = descr.replace('strap', '')

    # Garage
    if 'gdnod' in descr:
        parsed_data['garage'] = 'no'
        descr = descr.replace('gdnod', '')
    elif 'gdno2' in descr:
        parsed_data['garage'] = 'no'
        descr = descr.replace('gdno2', '')
    elif 'gdstd' in descr:
        parsed_data['garage'] = 'std'
        descr = descr.replace('gdstd', '')
    elif 'gdwkd' in descr:
        parsed_data['garage'] = 'wkd'
        descr = descr.replace('gdwkd', '')
    elif 'gdsup' in descr:
        parsed_data['garage'] = 'sup'
        descr = descr.replace('gdsup', '')

    # Shutters
    if 'shtys' in descr:
        parsed_data['shutters'] = True
        descr = descr.replace('shtys', '')
    elif 'shtno' in descr:
        parsed_data['shutters'] = False
        descr = descr.replace('shtno', '')

    # Roof Cover
    if 'rcbur' in descr:
        parsed_data['roof_cover'] = 'bur'
        descr = descr.replace('rcbur', '')
    elif 'rcspm' in descr:
        parsed_data['roof_cover'] = 'spm'
        descr = descr.replace('rcspm', '')

    # Roof Cover - Alternative Description
    if 'rcshl' in descr:
        parsed_data['roof_cover'] = 'cshl'  # cover, shingle
        descr = descr.replace('rcshl', '')
    elif 'rsmtl' in descr:
        parsed_data['roof_cover'] = 'smtl'  # sheet metal
        descr = descr.replace('rsmtl', '')

    # Roof Quality
    if 'rqgod' in descr:
        parsed_data['roof_quality'] = 'god'
        descr = descr.replace('rqgod', '')
    elif 'rqpor' in descr:
        parsed_data['roof_quality'] = 'por'
        descr = descr.replace('rqpor', '')

    # Masonry Reinforcing
    if 'rmfys' in descr:
        parsed_data['masonry_reinforcing'] = True
        descr = descr.replace('rmfys', '')
    elif 'rmfno' in descr:
        parsed_data['masonry_reinforcing'] = False
        descr = descr.replace('rmfno', '')

    # Roof Frame Type
    if 'rftrs' in descr:
        parsed_data['roof_frame_type'] = 'trs'  # wood truss
        descr = descr.replace('rftrs', '')
    elif 'rfows' in descr:
        parsed_data['roof_frame_type'] = 'ows'  # OWSJ
        descr = descr.replace('rfows', '')

    # Wind Debris Environment
    if 'widdA' in descr:
        parsed_data['wind_debris'] = 'A'  # res/comm.
        descr = descr.replace('widdA', '')
    elif 'widdB' in descr:
        parsed_data['wind_debris'] = 'B'  # varies by direction
        descr = descr.replace('widdB', '')
    elif 'widdC' in descr:
        parsed_data['wind_debris'] = 'C'  # residential
        descr = descr.replace('widdC', '')
    elif 'widdD' in descr:
        parsed_data['wind_debris'] = 'D'  # none
        descr = descr.replace('widdD', '')

    # Roof Deck Age
    if 'dqgod' in descr:
        parsed_data['roof_deck_age'] = 'god'  # new or average
        descr = descr.replace('dqgod', '')
    elif 'dqpor' in descr:
        parsed_data['roof_deck_age'] = 'por'  # old
        descr = descr.replace('dqpor', '')

    # Metal Roof Deck Attachment
    if 'rd100' in descr:
        parsed_data['metal_rda'] = 'std'  # standard
        descr = descr.replace('rd100', '')
    elif 'rd110' in descr:
        parsed_data['metal_rda'] = 'sup'  # superior
        descr = descr.replace('rd110', '')

    # Number of Units
    if 'nusgl' in descr:
        parsed_data['num_of_units'] = 'sgl'
        descr = descr.replace('nusgl', '')
    elif 'numlt' in descr:
        parsed_data['num_of_units'] = 'mlt'
        descr = descr.replace('numlt', '')

    # Joist Spacing
    if 'jspa4' in descr:
        parsed_data['joist_spacing'] = '4'
        descr = descr.replace('jspa4', '')
    elif 'jspa6' in descr:
        parsed_data['joist_spacing'] = '6'
        descr = descr.replace('jspa6', '')

    # Window Area
    if 'walow' in descr:
        parsed_data['window_area'] = 'low'
        descr = descr.replace('walow', '')
    elif 'wamed' in descr:
        parsed_data['window_area'] = 'med'
        descr = descr.replace('wamed', '')
    elif 'wahig' in descr:
        parsed_data['window_area'] = 'hig'
        descr = descr.replace('wahig', '')

    # ----- unknown attributes ---------

    if 'uprys' in descr:
        parsed_data['upgrade_??'] = True
        descr = descr.replace('uprys', '')
    elif 'uprno' in descr:
        parsed_data['upgrade_??'] = False
        descr = descr.replace('uprno', '')

    if 'wcdbl' in descr:
        parsed_data['wall_cover_??'] = 'dbl'
        descr = descr.replace('wcdbl', '')
    elif 'wcsgl' in descr:
        parsed_data['wall_cover_??'] = 'sgl'
        descr = descr.replace('wcsgl', '')

    if 'tspa2' in descr:
        parsed_data['tspa_??'] = '2'
        descr = descr.replace('tspa2', '')
    elif 'tspa4' in descr:
        parsed_data['tspa_??'] = '4'
        descr = descr.replace('tspa4', '')

    if 'mtdys' in descr:
        parsed_data['tie_downs'] = True
        descr = descr.replace('mtdys', '')
    elif 'mtdno' in descr:
        parsed_data['tie_downs'] = False
        descr = descr.replace('mtdno', '')

    return descr


def create_hazus_hurricane_damage_loss_files(  # noqa: C901, PLR0912, PLR0915
    fit_parameters=True,  # noqa: FBT002
    root_path='hurricane/building/portfolio/Hazus v5.1 coupled/',
):
    """
    Create Hazus hurricane damage and loss library files.

    This function processes raw Hazus hurricane data to generate damage and loss
    library files. It can either fit new normal or lognormal functions to the
    raw data or load existing fitted parameters from a fitted_parameters.csv
    file.

    Parameters
    ----------
    fit_parameters : bool, optional
        If True, fits new parameters to raw data. If False, loads existing
        fitted parameters from fitted_parameters.csv. Default is True.
    root_path : str, optional
        Path to the directory containing the Hazus data files.
        Default is 'hurricane/building/portfolio/Hazus v5.1 coupled/'.

    Returns
    -------
    None
        The function generates library files in the specified directory structure.
    """
    root_path = Path(root_path)

    # The original path points to the folder where the original parameters are
    # stored.
    original_path = root_path.parent / 'Hazus v5.1 original/'

    if fit_parameters:
        # Load RAW Hazus data

        raw_data_path = root_path / 'data_sources/input_files/'

        # read bldg data

        bldg_df_ST = pd.read_excel(  # noqa: N806
            raw_data_path / 'huListOfWindBldgTypes.xlsx', index_col=0
        )
        bldg_df_EF = pd.read_excel(  # noqa: N806
            raw_data_path / 'huListOfWindBldgTypesEF.xlsx', index_col=0
        )

        # make sure the column headers are in sync
        bldg_df_EF.columns = ['sbtName', *bldg_df_EF.columns[1:]]

        # offset the EF building IDs to ensure each archetype has a unique ID
        bldg_df_EF.index = max(bldg_df_ST.index) + bldg_df_EF.index
        bldg_df_EF.sort_index(inplace=True)  # noqa: PD002

        bldg_df = pd.concat([bldg_df_ST, bldg_df_EF], axis=0)

        # read fragility data

        frag_df_ST = pd.read_excel(raw_data_path / 'huDamLossFun.xlsx')  # noqa: N806

        frag_df_EF = pd.read_excel(raw_data_path / 'huDamLossFunEF.xlsx')  # noqa: N806
        frag_df_EF['wbID'] += max(bldg_df_ST.index)

        frag_df = pd.concat([frag_df_ST, frag_df_EF], axis=0, ignore_index=True)

        frag_df.sort_values(['wbID', 'TERRAINID', 'DamLossDescID'], inplace=True)  # noqa: PD002

        frag_df.reset_index(drop=True, inplace=True)  # noqa: PD002

        # Fix errors and fill missing data in the raw fragility database

        # ## Incorrect Damage State labels
        #
        # **Problem**
        # Fragility curve data is stored with the wrong Damage State label
        # for some archetypes. This leads to a lower damage state having
        # higher corresponding capacity than a higher damage state.
        #
        # **Scope**
        # The problem only affects 40 archetypes. In all cases, the data
        # under Damage State 4 seems to belong to Damage State 1.
        #
        # **Fix**
        # We offset the Damage State 1-3 data by one DS and move what is
        # under Damage State 4 to DS1. This yields a plausible set of
        # fragility curves.
        #
        # Note: When identifying which archetypes are affected, we compare
        # the probabilities of exceeding each damage state at every
        # discrete wind speed in the database. We look for instances where
        # a lower damage state has lower probability of exceedance than a
        # higher damage state. When comparing probabilities of exceedance,
        # we recognize that the data in the Hazus database is noisy and
        # consider an absolute 2% tolerance to accommodate this noise and
        # avoid false positives.
        #

        # get labels of columns in frag_df with damage and loss data
        wind_speeds_str = [c for c in frag_df.columns if 'WS' in c]

        # also get a list of floats based on the above labels
        wind_speeds = np.array([float(ws[2:]) for ws in wind_speeds_str])

        # set the max wind speed of interest
        max_speed = 200
        max_speed_id = max(np.where(wind_speeds <= max_speed)[0]) + 1

        DS_data = [  # noqa: N806
            frag_df[frag_df['DamLossDescID'] == ds].loc[:, wind_speeds_str]
            for ds in range(1, 5)
        ]

        # the problem affects DS4 probabilities
        archetypes = (DS_data[2] - DS_data[3].to_numpy() < -0.02).max(axis=1)
        # go through each affected archetype and fix the problem
        for frag_id in archetypes[archetypes == True].index:  # noqa: E712
            # get the wbID and terrain_id
            wbID, terrain_id = frag_df.loc[frag_id, ['wbID', 'TERRAINID']]  # noqa: N806

            # load the fragility info for the archetype
            frag_df_arch = frag_df.loc[
                (frag_df['wbID'] == wbID) & (frag_df['TERRAINID'] == terrain_id)
            ]

            # check which DS is stored as DS4
            # we do this by looking at the median capacities at each DS
            # through simple interpolation
            median_capacities = [
                np.interp(
                    0.5,
                    frag_df_arch[wind_speeds_str].iloc[ds].to_numpy(),
                    wind_speeds,
                )
                for ds in range(4)
            ]

            # then check where to store the values at DS4 to maintain
            # ascending exceedance probabilities
            target_DS = np.where(np.argsort(median_capacities) == 3)[0][0]  # noqa: N806

            # since this is always DS1 in the current database,
            # the script below works with that assumption and checks for exceptions
            if target_DS == 0:
                # first, extract the probabilities stored at DS4
                DS4_probs = frag_df_arch[wind_speeds_str].iloc[3].to_numpy()  # noqa: N806

                # then offset the probabilities of DS1-3 by one level
                for ds in [3, 2, 1]:
                    source_DS_index = frag_df_arch.index[ds - 1]  # noqa: N806
                    target_DS_index = frag_df_arch.index[ds]  # noqa: N806

                    frag_df.loc[target_DS_index, wind_speeds_str] = frag_df.loc[
                        source_DS_index, wind_speeds_str
                    ].to_numpy()

                # finally store the DS4 probs at the DS1 cells
                target_DS_index = frag_df_arch.index[0]  # noqa: N806

                frag_df.loc[target_DS_index, wind_speeds_str] = DS4_probs

        # ## Missing Damage State probabilities
        #
        # **Problem**
        # Some archetypes have only zeros in the cells that store Damage State
        # 4 exceedance probabilities at various wind speeds.
        #
        # **Scope**
        # This problem affects **346** building types in at least one but
        # typically all five terrain types. Altogether 1453 archetypes miss
        # their DS4 information in the raw data. As shown below, only DS4 is
        # affected, other damage states have some information for all
        # archetypes.
        #
        # **Fix**
        # We overwrite the zeros in the DS4 cells by copying the DS3
        # probabilities there. This leads to assuming zero probability of
        # exceeding DS4, which is still almost surely wrong. The **Hazus team
        # has been contacted** to provide the missing data or guidance on how
        # to improve this fix.

        # start by checking which archetypes have no damage data for at least
        # one Damage State

        # get the damage data for all archetypes
        DS_data = frag_df[frag_df['DamLossDescID'].isin([1, 2, 3, 4])].loc[  # noqa: N806
            :, wind_speeds_str
        ]

        # # check for invalid values
        # print(f'Global minimum value: {np.min(DS_data.to_numpy())}')
        # print(f'Global maximum value: {np.max(DS_data.to_numpy())}')

        # sum up the probabilities of exceeding each DS at various wind speeds
        DS_zero = DS_data.sum(axis=1)  # noqa: N806

        # and look for the lines where the sum is zero - i.e., all values are zero
        no_DS_info = frag_df.loc[DS_zero[DS_zero == 0].index]  # noqa: N806

        def overwrite_ds4_data():
            # now go through the building types in no_DS_info
            for wbID in no_DS_info['wbID'].unique():  # noqa: N806
                # and each terrain type that is affected
                for terrain_id in no_DS_info.loc[
                    no_DS_info['wbID'] == wbID, 'TERRAINID'
                ].to_numpy():
                    # get the fragility data for each archetype
                    frag_df_arch = frag_df.loc[
                        (frag_df['wbID'] == wbID)
                        & (frag_df['TERRAINID'] == terrain_id)
                    ]

                    # extract the DS3 information
                    DS3_data = frag_df_arch.loc[  # noqa: N806
                        frag_df['DamLossDescID'] == 3, wind_speeds_str
                    ].to_numpy()

                    # and overwrite the DS4 values in the original dataset
                    DS4_index = frag_df_arch.loc[frag_df['DamLossDescID'] == 4].index  # noqa: N806
                    frag_df.loc[DS4_index, wind_speeds_str] = DS3_data

        overwrite_ds4_data()

        # Fit fragility curves to discrete points

        # pre_calc

        flt_bldg_df = bldg_df.copy()

        # # this allows you to test it with a few archetypes before
        # # running the whole thing
        # flt_bldg_df = bldg_df.iloc[:100]

        # labels for all features, damage state data, and loss data
        column_names = [
            'bldg_type',
            'roof_shape',
            'roof_cover',
            'roof_quality',
            'sec_water_res',
            'roof_deck_attch',
            'roof_wall_conn',
            'garage',
            'shutters',
            'terr_rough',
            'upgrade_??',
            'wall_cover_??',
            'tspa_??',
            'masonry_reinforcing',
            'roof_frame_type',
            'wind_debris',
            'roof_deck_age',
            'metal_rda',
            'num_of_units',
            'joist_spacing',
            'window_area',
            'tie_downs',
            'DS1_dist',
            'DS1_mu',
            'DS1_sig',
            'DS1_fit',
            'DS1_meps',
            'DS2_dist',
            'DS2_mu',
            'DS2_sig',
            'DS2_fit',
            'DS2_meps',
            'DS3_dist',
            'DS3_mu',
            'DS3_sig',
            'DS3_fit',
            'DS3_meps',
            'DS4_dist',
            'DS4_mu',
            'DS4_sig',
            'DS4_fit',
            'DS4_meps',
            'L1',
            'L2',
            'L3',
            'L4',
            'L_fit',
            'L_meps',
            'DS1_original',
            'DS2_original',
            'DS3_original',
            'DS4_original',
            'L_original',
        ]

        # resulting dataframe
        new_df = pd.DataFrame(
            None, columns=column_names, index=np.arange(len(flt_bldg_df.index) * 5)
        )

        rows = []

        # calculation
        for index, row in tqdm(list(flt_bldg_df.iterrows())):
            # initialize the row for the archetype
            new_row = pd.Series(index=new_df.columns, dtype=np.float64)

            # store building type
            new_row['bldg_type'] = row['sbtName']

            # then parse the description and store the recognized parameter values
            descr = parse_description(row['charDescription'].strip(), new_row)

            # check if any part of the description remained unparsed
            if descr != '':
                print('WARNING', index, descr)  # noqa: T201

            # filter only those parts of the frag_df that correspond to
            # this archetype
            frag_df_arch = frag_df[frag_df['wbID'] == index]

            # cycle through the five terrain types in Hazus
            for terrain_id, roughness in enumerate([0.03, 0.15, 0.35, 0.7, 1.0]):
                # Hazus array indexing is 1-based
                terrain_id += 1  # noqa: PLW2901

                new_row_terrain = new_row.copy()

                # store the roughness length
                new_row_terrain['terr_rough'] = roughness

                # filter only those parts of the frag_df_arch that correspond
                # to this terrain type
                frag_df_arch_terrain = frag_df_arch[
                    frag_df_arch['TERRAINID'] == terrain_id
                ]

                mu_min = 0

                # for each damage state
                for DS in [1, 2, 3, 4]:  # noqa: N806
                    # get the exceedence probabilities for this DS of this
                    # archetype
                    P_exc = np.asarray(  # noqa: N806
                        frag_df_arch_terrain.loc[
                            frag_df_arch_terrain['DamLossDescID'] == DS,
                            wind_speeds_str,
                        ].to_numpy()[0]
                    )
                    multilinear_CDF_parameters = (  # noqa: N806
                        ','.join([str(x) for x in P_exc])
                        + '|'
                        + ','.join([str(x) for x in wind_speeds])
                    )

                    mu_0 = max(
                        [wind_speeds[np.argsort(abs(P_exc - 0.5))[0]], mu_min * 1.01]
                    )
                    sig_0 = 20
                    beta_0 = 0.2

                    median_id = max(np.where(wind_speeds <= mu_0)[0]) + 1
                    min_speed_id = max(np.where(wind_speeds <= 100)[0]) + 1
                    max_speed_id_mod = max(
                        [min([median_id, max_speed_id]), min_speed_id]
                    )

                    # define the two error measures to be minimized

                    # assuming Normal distribution for building capacity
                    def MSE_normal(params, mu_min, res_type='MSE'):  # noqa: N802
                        # unpack the parameters
                        mu, sig = params

                        # penalize invalid params
                        if (np.round(mu, decimals=1) <= mu_min) or (sig <= 0):
                            return 1e10

                        eps = (norm.cdf(wind_speeds, loc=mu, scale=sig) - P_exc)[  # noqa: B023
                            :max_speed_id_mod  # noqa: B023
                        ]

                        if res_type == 'MSE':
                            return sum(eps**2.0)

                        if res_type == 'max abs eps':
                            return max(abs(eps))

                        if res_type == 'eps':  # noqa: RET503
                            return eps

                    # assuming Lognormal distribution for building capacity
                    def MSE_lognormal(params, mu_min, res_type='MSE'):  # noqa: N802
                        # unpack the parameters
                        mu, beta = params

                        # penalize invalid params
                        if (np.round(mu, decimals=1) <= mu_min) or (beta <= 0):
                            return 1e10

                        eps = (
                            norm.cdf(np.log(wind_speeds), loc=np.log(mu), scale=beta)
                            - P_exc  # noqa: B023
                        )[
                            :max_speed_id_mod  # noqa: B023
                        ]

                        if res_type == 'MSE':
                            return sum(eps**2.0)

                        if res_type == 'max abs eps':
                            return max(abs(eps))

                        if res_type == 'eps':  # noqa: RET503
                            return eps

                    # minimize MSE assuming Normal distribution
                    res_normal = minimize(
                        MSE_normal,
                        [mu_0, sig_0],
                        args=(mu_min),
                        method='BFGS',
                        options={'maxiter': 50},
                    )

                    res_normal.x = np.array(
                        [
                            np.round(res_normal.x[0], decimals=1),
                            np.round(res_normal.x[1], decimals=1),
                        ]
                    )

                    # store MSE @ optimized location in fun
                    res_normal.fun = np.sqrt(
                        MSE_normal(res_normal.x, mu_min) / max_speed_id_mod
                    )
                    # and hijack maxcv to store max eps within res object
                    res_normal.maxcv = MSE_normal(
                        res_normal.x, mu_min, res_type='max abs eps'
                    )

                    # minimize MSE assuming Lognormal distribution
                    res_lognormal = minimize(
                        MSE_lognormal,
                        [mu_0, beta_0],
                        args=(mu_min),
                        method='BFGS',
                        options={'maxiter': 50},
                    )

                    res_lognormal.x = np.array(
                        [
                            np.round(res_lognormal.x[0], decimals=1),
                            np.round(res_lognormal.x[1], decimals=2),
                        ]
                    )

                    # store the MSE @ optimized location in fun
                    res_lognormal.fun = np.sqrt(
                        MSE_lognormal(res_lognormal.x, mu_min) / max_speed_id_mod
                    )
                    # and hijack maxcv to store max eps within res object
                    res_lognormal.maxcv = MSE_lognormal(
                        res_lognormal.x, mu_min, res_type='max abs eps'
                    )

                    # show a warning if either model could not be fit
                    if (res_normal.status != 0) and (res_lognormal.status != 0):
                        if (res_normal.fun < 1) or (res_lognormal.fun < 1):
                            pass

                        else:
                            print(  # noqa: T201
                                f'WARNING: Error in CDF fitting '
                                f'for {index}, {terrain_id}, {DS}'
                            )
                            # display(res_normal)
                            # display(res_lognormal)

                    # keep the better fit:
                    # first, check the mean absolute error
                    if res_normal.fun < res_lognormal.fun:
                        # If the mean absolute errors in the two models
                        # are very close AND one model has substantially
                        # smaller maximum error than the other, then
                        # choose the model with the smaller maximum error
                        if (np.log(res_lognormal.fun / res_normal.fun) < 0.1) and (
                            np.log(res_normal.maxcv / res_lognormal.maxcv) > 0.1
                        ):
                            dist_type = 'lognormal'
                            res = res_lognormal

                        else:
                            dist_type = 'normal'
                            res = res_normal

                    elif (np.log(res_normal.fun / res_lognormal.fun) < 0.1) and (
                        np.log(res_lognormal.maxcv / res_normal.maxcv) > 0.1
                    ):
                        dist_type = 'normal'
                        res = res_normal

                    else:
                        dist_type = 'lognormal'
                        res = res_lognormal

                    # store the parameters
                    new_row_terrain[f'DS{DS}_dist'] = dist_type
                    new_row_terrain[f'DS{DS}_mu'] = res.x[0]
                    new_row_terrain[f'DS{DS}_sig'] = res.x[1]
                    new_row_terrain[f'DS{DS}_fit'] = res.fun
                    new_row_terrain[f'DS{DS}_meps'] = res.maxcv
                    new_row_terrain[f'DS{DS}_original'] = multilinear_CDF_parameters

                    # consecutive damage states should have increasing capacities
                    mu_min = res.x[0]

                # Now we have the damages, continue with Losses

                # Focus on "Building losses" first
                L_ref = np.asarray(  # noqa: N806
                    frag_df_arch_terrain.loc[
                        frag_df_arch_terrain['DamLossDescID'] == 5, wind_speeds_str
                    ].to_numpy()[0]
                )

                multilinear_CDF_parameters = (  # noqa: N806
                    ','.join([str(x) for x in L_ref])
                    + '|'
                    + ','.join([str(x) for x in wind_speeds])
                )

                # We'll need the probability of each Damage State across the
                # pre-defined wind speeds
                DS_probs = np.zeros((4, len(wind_speeds)))  # noqa: N806

                for DS_id, DS in enumerate([1, 2, 3, 4]):  # noqa: N806
                    if new_row_terrain[f'DS{DS}_dist'] == 'normal':
                        DS_probs[DS_id] = norm.cdf(
                            wind_speeds,
                            loc=new_row_terrain[f'DS{DS}_mu'],
                            scale=new_row_terrain[f'DS{DS}_sig'],
                        )
                    else:
                        DS_probs[DS_id] = norm.cdf(
                            np.log(wind_speeds),
                            loc=np.log(new_row_terrain[f'DS{DS}_mu']),
                            scale=new_row_terrain[f'DS{DS}_sig'],
                        )

                # The losses for DS4 are calculated based on outcomes at the
                # highest wind speeds
                L_max = frag_df_arch_terrain.loc[  # noqa: N806
                    frag_df_arch_terrain['DamLossDescID'] == 5, 'WS250'
                ].to_numpy()[0]
                DS4_max = DS_probs[3][-1]  # noqa: N806

                L4 = np.round(min(L_max / DS4_max, 1.0), decimals=3)  # noqa: N806

                # if L4 < 0.75:
                #    print(index, terrain_id, L_max, DS4_max, L4)

                for i in range(3):
                    DS_probs[i] = DS_probs[i] - DS_probs[i + 1]

                # define the loss error measures to be minimized

                def SSE_loss(params, res_type='SSE'):  # noqa: N802
                    loss_ratios = params.copy()

                    # assume 1.0 for DS4
                    loss_ratios = np.append(loss_ratios, L4)  # noqa: B023

                    L_est = np.sum(loss_ratios * DS_probs.T, axis=1)  # noqa: B023, N806

                    # the error is the difference between reference and estimated losses
                    eps = (L_est - L_ref)[:max_speed_id]  # noqa: B023

                    if res_type == 'SSE':
                        # calculate the sum of squared errors across wind speeds
                        SSE = sum(eps**2.0)  # noqa: N806

                    elif res_type == 'max abs eps':
                        return max(abs(eps))

                    return SSE

                cons = (
                    {'type': 'ineq', 'fun': lambda x: x[1] - x[0] - 0.002},
                    {'type': 'ineq', 'fun': lambda x: x[2] - x[1] - 0.002},
                    {'type': 'ineq', 'fun': lambda x: L4 - x[2] - 0.002},  # noqa: B023
                )

                res = minimize(
                    SSE_loss,
                    [0.02, 0.1, 0.44],
                    bounds=((0.001, 1.0), (0.001, 1.0), (0.001, 1.0)),
                    constraints=cons,
                )

                res.x = np.round(res.x, decimals=3)

                # store MSE @ optimized location in fun
                res.fun = np.sqrt(SSE_loss(res.x) / len(L_ref))
                # and hijack maxcv to store max eps within res object
                res.maxcv = SSE_loss(res.x, res_type='max abs eps')

                # if (res.fun > 0.1) or (res.maxcv > 0.2):
                #    print(index, terrain_id, max_speed_id_mod)
                #    print(res.x, res.fun, res.maxcv)

                # store the parameters
                new_row_terrain['L1'] = res.x[0]
                new_row_terrain['L2'] = res.x[1]
                new_row_terrain['L3'] = res.x[2]
                new_row_terrain['L4'] = L4
                new_row_terrain['L_original'] = multilinear_CDF_parameters
                new_row_terrain['L_fit'] = res.fun
                new_row_terrain['L_meps'] = res.maxcv

                # display(new_row.to_frame().T)

                rows.append(new_row_terrain.to_frame().T)

        main_df = pd.concat(rows, axis=0, ignore_index=True)

        main_df.to_csv(root_path / 'data_sources/fitted_parameters.csv')

    main_df = pd.read_csv(
        root_path / 'data_sources/fitted_parameters.csv',
        index_col=0,
        low_memory=False,
        dtype={'joist_spacing': str},
    )

    # Prepare the Damage and Loss Model Data Files
    #
    # ## Prepare archetype IDs

    # This dict maps the building IDs used in HAZUS to the dotted
    # style we use in Pelicun

    bldg_type_map = {
        'WSF1': 'W.SF.1',
        'WSF2': 'W.SF.2',
        'WMUH1': 'W.MUH.1',
        'WMUH2': 'W.MUH.2',
        'WMUH3': 'W.MUH.3',
        'MSF1': 'M.SF.1',
        'MSF2': 'M.SF.2',
        'MMUH1': 'M.MUH.1',
        'MMUH2': 'M.MUH.2',
        'MMUH3': 'M.MUH.3',
        'MLRM1': 'M.LRM.1',
        'MLRM2': 'M.LRM.2',
        'MLRI': 'M.LRI',
        'MERBL': 'M.ERB.L',
        'MERBM': 'M.ERB.M',
        'MERBH': 'M.ERB.H',
        'MECBL': 'M.ECB.L',
        'MECBM': 'M.ECB.M',
        'MECBH': 'M.ECB.H',
        'CERBL': 'C.ERB.L',
        'CERBM': 'C.ERB.M',
        'CERBH': 'C.ERB.H',
        'CECBL': 'C.ECB.L',
        'CECBM': 'C.ECB.M',
        'CECBH': 'C.ECB.H',
        'SPMBS': 'S.PMB.S',
        'SPMBM': 'S.PMB.M',
        'SPMBL': 'S.PMB.L',
        'SERBL': 'S.ERB.L',
        'SERBM': 'S.ERB.M',
        'SERBH': 'S.ERB.H',
        'SECBL': 'S.ECB.L',
        'SECBM': 'S.ECB.M',
        'SECBH': 'S.ECB.H',
        'MHPHUD': 'MH.PHUD',
        'MH76HUD': 'MH.76HUD',
        'MH94HUDI': 'MH.94HUDI',
        'MH94HUDII': 'MH.94HUDII',
        'MH94HUDIII': 'MH.94HUDIII',
        'HUEFFS': 'HUEF.FS',
        'HUEFHS': 'HUEF.H.S',
        'HUEFHM': 'HUEF.H.M',
        'HUEFHL': 'HUEF.H.L',
        'HUEFSS': 'HUEF.S.S',
        'HUEFSM': 'HUEF.S.M',
        'HUEFSL': 'HUEF.S.L',
        'HUEFEO': 'HUEF.EO',
        'HUEFPS': 'HUEF.PS',
    }

    # beyond the building ID, we also want to capture the detailed
    # features of each archetype
    # with human-readable text (as opposed to what we had to parse in the
    # parse_description function earlier

    out_df = main_df.copy()
    out_df['ID'] = ''

    # some general formatting to make file name generation easier
    out_df['shutters'] = out_df['shutters'].astype(int)
    out_df['terr_rough'] = (out_df['terr_rough'] * 100.0).astype(int)

    for index, row in tqdm(list(out_df.iterrows())):
        # define the name of the building damage and loss configuration
        bldg_type = row['bldg_type']
        critical_cols = []

        if bldg_type[:3] == 'WSF':
            cols_of_interest = [
                'bldg_type',
                'roof_shape',
                'sec_water_res',
                'roof_deck_attch',
                'roof_wall_conn',
                'garage',
                'shutters',
                'terr_rough',
            ]
            critical_cols = ['roof_deck_attch', 'garage']

        elif bldg_type[:4] == 'WMUH':
            cols_of_interest = [
                'bldg_type',
                'roof_shape',
                'roof_cover',
                'roof_quality',
                'sec_water_res',
                'roof_deck_attch',
                'roof_wall_conn',
                'shutters',
                'terr_rough',
            ]

        elif bldg_type[:3] == 'MSF':
            cols_of_interest = [
                'bldg_type',
                'roof_shape',
                'roof_wall_conn',
                'roof_frame_type',
                'roof_deck_attch',
                'shutters',
                'sec_water_res',
                'garage',
                'masonry_reinforcing',
                'roof_cover',
                'terr_rough',
            ]

        elif bldg_type[:4] == 'MMUH':
            cols_of_interest = [
                'bldg_type',
                'roof_shape',
                'sec_water_res',
                'roof_cover',
                'roof_quality',
                'roof_deck_attch',
                'roof_wall_conn',
                'shutters',
                'masonry_reinforcing',
                'terr_rough',
            ]

        elif bldg_type[:5] == 'MLRM1':
            cols_of_interest = [
                'bldg_type',
                'roof_cover',
                'shutters',
                'masonry_reinforcing',
                'wind_debris',
                'roof_frame_type',
                'roof_deck_attch',
                'roof_wall_conn',
                'roof_deck_age',
                'metal_rda',
                'terr_rough',
            ]

        elif bldg_type[:5] == 'MLRM2':
            cols_of_interest = [
                'bldg_type',
                'roof_cover',
                'shutters',
                'masonry_reinforcing',
                'wind_debris',
                'roof_frame_type',
                'roof_deck_attch',
                'roof_wall_conn',
                'roof_deck_age',
                'metal_rda',
                'num_of_units',
                'joist_spacing',
                'terr_rough',
            ]

        elif bldg_type[:4] == 'MLRI':
            cols_of_interest = [
                'bldg_type',
                'shutters',
                'masonry_reinforcing',
                'roof_deck_age',
                'metal_rda',
                'terr_rough',
            ]

        elif bldg_type[:4] == 'MERB' or bldg_type[:4] == 'MECB':
            cols_of_interest = [
                'bldg_type',
                'roof_cover',
                'shutters',
                'wind_debris',
                'metal_rda',
                'window_area',
                'terr_rough',
            ]

        elif bldg_type[:4] == 'CERB' or bldg_type[:4] == 'CECB':
            cols_of_interest = [
                'bldg_type',
                'roof_cover',
                'shutters',
                'wind_debris',
                'window_area',
                'terr_rough',
            ]

        elif bldg_type[:4] == 'SERB' or bldg_type[:4] == 'SECB':
            cols_of_interest = [
                'bldg_type',
                'roof_cover',
                'shutters',
                'wind_debris',
                'metal_rda',
                'window_area',
                'terr_rough',
            ]

        elif bldg_type[:4] == 'SPMB':
            cols_of_interest = [
                'bldg_type',
                'shutters',
                'roof_deck_age',
                'metal_rda',
                'terr_rough',
            ]

        elif bldg_type[:2] == 'MH':
            cols_of_interest = ['bldg_type', 'shutters', 'tie_downs', 'terr_rough']

            critical_cols = [
                'tie_downs',
            ]

        elif bldg_type[:6] == 'HUEFFS':
            cols_of_interest = [
                'bldg_type',
                'roof_cover',
                'shutters',
                'wind_debris',
                'roof_deck_age',
                'metal_rda',
                'terr_rough',
            ]

        elif bldg_type[:6] == 'HUEFPS' or bldg_type[:6] == 'HUEFEO':
            cols_of_interest = [
                'bldg_type',
                'roof_cover',
                'shutters',
                'wind_debris',
                'metal_rda',
                'window_area',
                'terr_rough',
            ]

        elif bldg_type[:5] == 'HUEFH':
            cols_of_interest = [
                'bldg_type',
                'roof_cover',
                'wind_debris',
                'metal_rda',
                'shutters',
                'terr_rough',
            ]

        elif bldg_type[:5] == 'HUEFS':
            cols_of_interest = [
                'bldg_type',
                'roof_cover',
                'shutters',
                'wind_debris',
                'roof_deck_age',
                'metal_rda',
                'terr_rough',
            ]

        else:
            continue

        bldg_chars = row[cols_of_interest]

        # If a critical feature is undefined, we consider the archetype
        # unreliable and skip it
        skip_archetype = False
        for critical_col in critical_cols:
            if (critical_col in cols_of_interest) and (
                pd.isna(bldg_chars[critical_col])
            ):
                skip_archetype = True

        if skip_archetype:
            continue

        # Replace boolean strings with int equivalents
        bool_cols = ['sec_water_res', 'masonry_reinforcing', 'tie_downs']
        for col in bool_cols:
            if col in cols_of_interest:
                if pd.isna(bldg_chars[col]):
                    bldg_chars[col] = 'null'
                else:
                    bldg_chars[col] = int(bldg_chars[col])

        # Replace missing values with null
        for col in cols_of_interest:
            if pd.isna(bldg_chars[col]):
                bldg_chars[col] = 'null'

        # Add roof frame type info to MSF archetypes
        if bldg_type.startswith('MSF'):
            if bldg_chars['roof_cover'] in ['smtl', 'cshl']:
                bldg_chars['roof_frame_type'] = 'ows'
            else:
                bldg_chars['roof_frame_type'] = 'trs'

        bldg_chars['bldg_type'] = bldg_type_map[bldg_chars['bldg_type']]

        out_df.loc[index, 'ID'] = '.'.join(bldg_chars.astype(str))

        # We also need to make sure the raw multilinear CDF parameters are valid
        # and convert them to the format used in Pelicun

        for LS_i in range(1, 5):  # noqa: N806
            cdf_y, cdf_x = [
                np.array(vals.split(','), dtype=float)
                for vals in row[f'DS{LS_i}_original'].split('|')
            ]

            exclude_bottom = np.where(cdf_y < 0.0)[0]
            if len(exclude_bottom) > 0:
                cdf_y = cdf_y[exclude_bottom[-1] + 1 :]
                cdf_x = cdf_x[exclude_bottom[-1] + 1 :]

                print('Invalid values trimmed from bottom')  # noqa: T201

            exclude_top = np.where(cdf_y > 1.0)[0]
            if len(exclude_top) > 0:
                cdf_y = cdf_y[: exclude_top[-1]]
                cdf_x = cdf_x[: exclude_top[-1]]

                print('Invalid values trimmed from top')  # noqa: T201

            bottom_list = np.where(cdf_y <= np.nextafter(0.0, 1))[0]
            if len(bottom_list) > 0:
                cdf_y = cdf_y[bottom_list[-1] :]
                cdf_x = cdf_x[bottom_list[-1] :]
            else:
                cdf_y = np.insert(cdf_y, 0, 0.0)
                cdf_x = np.insert(cdf_x, 0, cdf_x[0] - 5.0)
                # This assumes wind speeds are provided at 5 mph increments

            top_list = np.where(cdf_y >= np.nextafter(1.0, -1))[0]
            if len(top_list) > 0:
                cdf_y = cdf_y[: top_list[0] + 1]
                cdf_x = cdf_x[: top_list[0] + 1]
            else:
                cdf_y = np.append(cdf_y, 1.0)
                cdf_x = np.append(cdf_x, cdf_x[-1] + 5.0)
                # This assumes wind speeds are provided at 5 mph increments

            if not np.array_equal(np.sort(cdf_x), cdf_x):
                print('Invalid CDF_x values')  # noqa: T201

            if not np.array_equal(np.sort(cdf_y), cdf_y):
                print('Invalid CDF_y values')  # noqa: T201

            out_df.loc[index, f'DS{LS_i}_original'] = '|'.join(
                [
                    ','.join(cdf_x.astype(int).astype(str)),
                    ','.join(cdf_y.astype(str)),
                ]
            )

    # ## Export to CSV file

    # initialize the fragility table
    df_db_fit = pd.DataFrame(
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
            'LS2-Family',
            'LS2-Theta_0',
            'LS2-Theta_1',
            'LS3-Family',
            'LS3-Theta_0',
            'LS3-Theta_1',
            'LS4-Family',
            'LS4-Theta_0',
            'LS4-Theta_1',
        ],
        index=out_df.index,
        dtype=float,
    )

    df_db_original = pd.DataFrame(
        columns=[
            'ID',
            'Incomplete',
            'Demand-Type',
            'Demand-Unit',
            'Demand-Offset',
            'Demand-Directional',
            'LS1-Family',
            'LS1-Theta_0',
            'LS2-Family',
            'LS2-Theta_0',
            'LS3-Family',
            'LS3-Theta_0',
            'LS4-Family',
            'LS4-Theta_0',
        ],
        index=out_df.index,
        dtype=float,
    )

    for df_db in (df_db_original, df_db_fit):
        df_db['ID'] = out_df['ID']
        df_db['Incomplete'] = 0
        df_db['Demand-Type'] = 'Peak Gust Wind Speed'
        df_db['Demand-Unit'] = 'mph'
        df_db['Demand-Offset'] = 0
        df_db['Demand-Directional'] = 0

    for LS_i in range(1, 5):  # noqa: N806
        df_db_original[f'LS{LS_i}-Family'] = 'multilinear_CDF'
        df_db_original[f'LS{LS_i}-Theta_0'] = out_df[f'DS{LS_i}_original']

        df_db_fit[f'LS{LS_i}-Family'] = out_df[f'DS{LS_i}_dist']
        df_db_fit[f'LS{LS_i}-Theta_0'] = out_df[f'DS{LS_i}_mu'].astype(float)
        df_db_fit[f'LS{LS_i}-Theta_1'] = out_df[f'DS{LS_i}_sig'].astype(float)

        # store the COV for normal distributions (instead of the std)
        normal_mask = df_db_fit[f'LS{LS_i}-Family'] == 'normal'
        df_db_fit.loc[normal_mask, f'LS{LS_i}-Theta_1'] /= df_db_fit.loc[
            normal_mask, f'LS{LS_i}-Theta_0'
        ]
        df_db_fit = df_db_fit.round({f'LS{LS_i}-Theta_1': 2})

    df_db_original, df_db_fit = [
        df_i.loc[df_i['ID'] != ''].set_index('ID').sort_index().convert_dtypes()
        for df_i in (df_db_original, df_db_fit)
    ]

    df_db_fit.to_csv(root_path / 'fragility.csv')
    df_db_original.to_csv(original_path / 'fragility.csv')

    # initialize the output loss table
    # define the columns
    out_cols = [
        'ID',
        'Incomplete',
        'Demand-Type',
        'Demand-Unit',
        'Demand-Offset',
        'Demand-Directional',
        'DV-Unit',
        'LossFunction-Theta_0',
    ]
    df_db_original = pd.DataFrame(columns=out_cols, index=out_df.index, dtype=float)
    df_db_original['ID'] = [f'{x}-Cost' for x in out_df['ID']]
    df_db_original['Incomplete'] = 0
    df_db_original['Demand-Type'] = 'Peak Gust Wind Speed'
    df_db_original['Demand-Unit'] = 'mph'
    df_db_original['Demand-Offset'] = 0
    df_db_original['Demand-Directional'] = 0
    df_db_original['DV-Unit'] = 'loss_ratio'
    df_db_original['LossFunction-Theta_0'] = out_df['L_original']
    df_db_original = df_db_original.loc[df_db_original['ID'] != '-Cost']
    df_db_original = df_db_original.set_index('ID').sort_index().convert_dtypes()

    out_cols = [
        'Incomplete',
        'Quantity-Unit',
        'DV-Unit',
    ]
    for DS_i in range(1, 5):  # noqa: N806
        out_cols += [f'DS{DS_i}-Theta_0']
    df_db_fit = pd.DataFrame(columns=out_cols, index=out_df.index, dtype=float)
    df_db_fit['ID'] = [f'{x}-Cost' for x in out_df['ID']]
    df_db_fit['Incomplete'] = 0
    df_db_fit['Quantity-Unit'] = '1 EA'
    df_db_fit['DV-Unit'] = 'loss_ratio'
    for LS_i in range(1, 5):  # noqa: N806
        df_db_fit[f'DS{LS_i}-Theta_0'] = out_df[f'L{LS_i}']
    df_db_fit = df_db_fit.loc[df_db_fit['ID'] != '-Cost']
    df_db_fit = df_db_fit.set_index('ID').sort_index().convert_dtypes()

    df_db_fit.to_csv(root_path / 'consequence_repair.csv')
    df_db_original.to_csv(original_path / 'loss_repair.csv')


def create_hazus_hurricane_metadata_files(  # noqa: C901
    source_file: str = 'fragility.csv',
    meta_file: str = 'data_sources/input_files/metadata.json',
    target_meta_file_damage: str = 'fragility.json',
    target_meta_file_loss: str = 'consequence_repair.json',
    target_meta_file_damage_original: str = 'fragility.json',
    target_meta_file_loss_original: str = 'loss_repair.json',
    root_path='hurricane/building/portfolio/Hazus v5.1 coupled/',
) -> None:
    """
    Create a database metadata file for the HAZUS Hurricane fragilities.

    This method was developed to add a json file with metadata
    accompanying `damage_DB_SimCenter_Hazus_HU_bldg.csv`. That file
    contains fragility curves fitted to Hazus Hurricane data relaetd
    to the Hazus Hurricane Technical Manual v4.2.

    Parameters
    ----------
    source_file: string
        Path to the Hazus Hurricane fragility data.
    meta_file: string
        Path to a predefined fragility metadata file.
    target_meta_file_damage: string
        Path where the fitted fragility metadata should be saved. A
        json file is expected.
    target_meta_file_loss: string
        Path where the fitted loss damage consequence metadata should
        be saved. A json file is expected.
    target_meta_file_damage_original: string
        Path where the unmodified fragility metadata should be
        saved. A json file is expected.
    target_meta_file_loss_original: string
        Path where the unmodified loss function metadata should be
        saved. A json file is expected.
    root_path: string
        Path to the root folder - everything above is relative to this.

    """
    # Procedure Overview:
    # (1) We define several dictionaries mapping chunks of the
    # composite asset ID (the parts between periods) to human-readable
    # (`-h` for short) representations.
    # (2) We define -h asset type descriptions and map them to the
    # first-most relevant ID chunks (`primary chunks`)
    # (3) We map asset class codes with general asset classes
    # (4) We define the required dictionaries from (1) that decode the
    # ID chunks after the `primary chunks` for each general asset
    # class
    # (5) We decode:
    # ID -> asset class -> general asset class -> dictionaries
    # -> ID turns to -h text by combining the description of the asset class
    # from the `primary chunks` and the decoded description of the
    # following chunks using the dictionaries.

    root_path = Path(root_path)

    # The original path points to the folder where the original parameters are
    # stored.
    original_path = Path(root_path).parent / 'Hazus v5.1 original/'

    # Combine paths:
    source_file = root_path / source_file
    meta_file = root_path / meta_file
    target_meta_file_damage = root_path / target_meta_file_damage
    target_meta_file_loss = root_path / target_meta_file_loss
    target_meta_file_damage_original = (
        original_path / target_meta_file_damage_original
    )
    target_meta_file_loss_original = original_path / target_meta_file_loss_original

    #
    # (1) Dictionaries
    #

    roof_shape = {
        'flt': 'Flat roof.',
        'gab': 'Gable roof.',
        'hip': 'Hip roof.',
    }

    secondary_water_resistance = {
        '1': 'Secondary water resistance.',
        '0': 'No secondary water resistance.',
        'null': 'No information on secondary water resistance.',
    }

    roof_deck_attachment = {
        '6d': '6d roof deck nails.',
        '6s': '6s roof deck nails.',
        '8d': '8d roof deck nails.',
        '8s': '8s roof deck nails.',
        'std': 'Standard roof deck attachment.',
        'sup': 'Superior roof deck attachment.',
        'null': 'Missing roof deck attachment information.',
    }

    roof_wall_connection = {
        'tnail': 'Roof-to-wall toe nails.',
        'strap': 'Roof-to-wall straps.',
        'null': 'Missing roof-to-wall connection information.',
    }

    garage_presence = {
        'no': 'No garage.',
        'wkd': 'Weak garage door.',
        'std': 'Standard garage door.',
        'sup': 'Strong garage door.',
        'null': 'No information on garage.',
    }

    shutters = {'1': 'Has Shutters.', '0': 'No shutters.'}

    roof_cover = {
        'bur': 'Built-up roof cover.',
        'spm': 'Single-ply membrane roof cover.',
        'smtl': 'Sheet metal roof cover.',
        'cshl': 'Shingle roof cover.',
        'null': 'No information on roof cover.',
    }

    roof_quality = {
        'god': 'Good roof quality.',
        'por': 'Poor roof quality.',
        'null': 'No information on roof quality.',
    }

    masonry_reinforcing = {
        '1': 'Has masonry reinforcing.',
        '0': 'No masonry reinforcing.',
        'null': 'Unknown information on masonry reinforcing.',
    }

    roof_frame_type = {
        'trs': 'Wood truss roof frame.',
        'ows': 'OWSJ roof frame.',
    }

    wind_debris_environment = {
        'A': 'Residentiao/commercial wind debris environment.',
        'B': 'Wind debris environment varies by direction.',
        'C': 'Residential wind debris environment.',
        'D': 'No wind debris environment.',
    }

    roof_deck_age = {
        'god': 'New or average roof age.',
        'por': 'Old roof age.',
        'null': 'Missing roof age information.',
    }

    roof_metal_deck_attachment_quality = {
        'std': 'Standard metal deck roof attachment.',
        'sup': 'Superior metal deck roof attachment.',
        'null': 'Missing roof attachment quality information.',
    }

    number_of_units = {
        'sgl': 'Single unit.',
        'mlt': 'Multi-unit.',
        'null': 'Unknown number of units.',
    }

    joist_spacing = {
        '4': '4 ft joist spacing.',
        '6': '6 ft foot joist spacing.',
        'null': 'Unknown joist spacing.',
    }

    window_area = {
        'low': 'Low window area.',
        'med': 'Medium window area.',
        'hig': 'High window area.',
    }

    tie_downs = {'1': 'Tie downs.', '0': 'No tie downs.'}

    terrain_surface_roughness = {
        '3': 'Terrain surface roughness: 0.03 m.',
        '15': 'Terrain surface roughness: 0.15 m.',
        '35': 'Terrain surface roughness: 0.35 m.',
        '70': 'Terrain surface roughness: 0.7 m.',
        '100': 'Terrain surface roughness: 1 m.',
    }

    #
    # (2) Asset type descriptions
    #

    # maps class type code to -h description
    class_types = {
        # ------------------------
        'W.SF.1': 'Wood, Single-family, One-story.',
        'W.SF.2': 'Wood, Single-family, Two or More Stories.',
        # ------------------------
        'W.MUH.1': 'Wood, Multi-Unit Housing, One-story.',
        'W.MUH.2': 'Wood, Multi-Unit Housing, Two Stories.',
        'W.MUH.3': 'Wood, Multi-Unit Housing, Three or More Stories.',
        # ------------------------
        'M.SF.1': 'Masonry, Single-family, One-story.',
        'M.SF.2': 'Masonry, Single-family, Two or More Stories.',
        # ------------------------
        'M.MUH.1': 'Masonry, Multi-Unit Housing, One-story.',
        'M.MUH.2': 'Masonry, Multi-Unit Housing, Two Stories.',
        'M.MUH.3': 'Masonry, Multi-Unit Housing, Three or More Stories.',
        # ------------------------
        'M.LRM.1': 'Masonry, Low-Rise Strip Mall, Up to 15 Feet.',
        'M.LRM.2': 'Masonry, Low-Rise Strip Mall, More than 15 Feet.',
        # ------------------------
        'M.LRI': 'Masonry, Low-Rise Industrial/Warehouse/Factory Buildings.',
        # ------------------------
        'M.ERB.L': (
            'Masonry, Engineered Residential Building, Low-Rise (1-2 Stories).'
        ),
        'M.ERB.M': (
            'Masonry, Engineered Residential Building, Mid-Rise (3-5 Stories).'
        ),
        'M.ERB.H': (
            'Masonry, Engineered Residential Building, High-Rise (6+ Stories).'
        ),
        # ------------------------
        'M.ECB.L': (
            'Masonry, Engineered Commercial Building, Low-Rise (1-2 Stories).'
        ),
        'M.ECB.M': (
            'Masonry, Engineered Commercial Building, Mid-Rise (3-5 Stories).'
        ),
        'M.ECB.H': (
            'Masonry, Engineered Commercial Building, High-Rise (6+ Stories).'
        ),
        # ------------------------
        'C.ERB.L': (
            'Concrete, Engineered Residential Building, Low-Rise (1-2 Stories).'
        ),
        'C.ERB.M': (
            'Concrete, Engineered Residential Building, Mid-Rise (3-5 Stories).'
        ),
        'C.ERB.H': (
            'Concrete, Engineered Residential Building, High-Rise (6+ Stories).'
        ),
        # ------------------------
        'C.ECB.L': (
            'Concrete, Engineered Commercial Building, Low-Rise (1-2 Stories).'
        ),
        'C.ECB.M': (
            'Concrete, Engineered Commercial Building, Mid-Rise (3-5 Stories).'
        ),
        'C.ECB.H': (
            'Concrete, Engineered Commercial Building, High-Rise (6+ Stories).'
        ),
        # ------------------------
        'S.PMB.S': 'Steel, Pre-Engineered Metal Building, Small.',
        'S.PMB.M': 'Steel, Pre-Engineered Metal Building, Medium.',
        'S.PMB.L': 'Steel, Pre-Engineered Metal Building, Large.',
        # ------------------------
        'S.ERB.L': 'Steel, Engineered Residential Building, Low-Rise (1-2 Stories).',
        'S.ERB.M': 'Steel, Engineered Residential Building, Mid-Rise (3-5 Stories).',
        'S.ERB.H': 'Steel, Engineered Residential Building, High-Rise (6+ Stories).',
        # ------------------------
        'S.ECB.L': 'Steel, Engineered Commercial Building, Low-Rise (1-2 Stories).',
        'S.ECB.M': 'Steel, Engineered Commercial Building, Mid-Rise (3-5 Stories).',
        'S.ECB.H': 'Steel, Engineered Commercial Building, High-Rise (6+ Stories).',
        # ------------------------
        'MH.PHUD': 'Manufactured Home, Pre-Housing and Urban Development (HUD).',
        'MH.76HUD': 'Manufactured Home, 1976 HUD.',
        'MH.94HUDI': 'Manufactured Home, 1994 HUD - Wind Zone I.',
        'MH.94HUDII': 'Manufactured Home, 1994 HUD - Wind Zone II.',
        'MH.94HUDIII': 'Manufactured Home, 1994 HUD - Wind Zone III.',
        # ------------------------
        'HUEF.H.S': 'Small Hospital, Hospital with fewer than 50 Beds.',
        'HUEF.H.M': 'Medium Hospital, Hospital with beds between 50 & 150.',
        'HUEF.H.L': 'Large Hospital, Hospital with more than 150 Beds.',
        # ------------------------
        'HUEF.S.S': 'Elementary School.',
        'HUEF.S.M': 'High school, two-story.',
        'HUEF.S.L': 'Large high school, three-story.',
        # ------------------------
        'HUEF.EO': 'Emergency Operation Centers.',
        'HUEF.FS': 'Fire Station.',
        'HUEF.PS': 'Police Station.',
        # ------------------------
    }

    def find_class_type(entry: str) -> str | None:
        """
        Find the class type code.

        Find the class type code from an entry string based on
        predefined patterns.

        Parameters
        ----------
        entry : str
            A string representing the entry, consisting of delimited
            segments that correspond to various attributes of an
            asset.

        Returns
        -------
        str or None
            The class type code if a matching pattern is found;
            otherwise, None if no pattern matches the input string.

        """
        entry_elements = entry.split('.')
        for nper in range(1, len(entry_elements)):
            first_parts = '.'.join(entry_elements[:nper])
            if first_parts in class_types:
                return first_parts
        return None

    #
    # (3) General asset class
    #

    # maps class code type to general class code
    general_classes = {
        # ------------------------
        'W.SF.1': 'WSF',
        'W.SF.2': 'WSF',
        # ------------------------
        'W.MUH.1': 'WMUH',
        'W.MUH.2': 'WMUH',
        'W.MUH.3': 'WMUH',
        # ------------------------
        'M.SF.1': 'MSF',
        'M.SF.2': 'MSF',
        # ------------------------
        'M.MUH.1': 'MMUH',
        'M.MUH.2': 'MMUH',
        'M.MUH.3': 'MMUH',
        # ------------------------
        'M.LRM.1': 'MLRM1',
        'M.LRM.2': 'MLRM2',
        # ------------------------
        'M.LRI': 'MLRI',
        # ------------------------
        'M.ERB.L': 'MERB',
        'M.ERB.M': 'MERB',
        'M.ERB.H': 'MERB',
        # ------------------------
        'M.ECB.L': 'MECB',
        'M.ECB.M': 'MECB',
        'M.ECB.H': 'MECB',
        # ------------------------
        'C.ERB.L': 'CERB',
        'C.ERB.M': 'CERB',
        'C.ERB.H': 'CERB',
        # ------------------------
        'C.ECB.L': 'CECB',
        'C.ECB.M': 'CECB',
        'C.ECB.H': 'CECB',
        # ------------------------
        'S.PMB.S': 'SPMB',
        'S.PMB.M': 'SPMB',
        'S.PMB.L': 'SPMB',
        # ------------------------
        'S.ERB.L': 'SERB',
        'S.ERB.M': 'SERB',
        'S.ERB.H': 'SERB',
        # ------------------------
        'S.ECB.L': 'SECB',
        'S.ECB.M': 'SECB',
        'S.ECB.H': 'SECB',
        # ------------------------
        'MH.PHUD': 'MH',
        'MH.76HUD': 'MH',
        'MH.94HUDI': 'MH',
        'MH.94HUDII': 'MH',
        'MH.94HUDIII': 'MH',
        # ------------------------
        'HUEF.H.S': 'HUEFH',
        'HUEF.H.M': 'HUEFH',
        'HUEF.H.L': 'HUEFH',
        # ------------------------
        'HUEF.S.S': 'HUEFS',
        'HUEF.S.M': 'HUEFS',
        'HUEF.S.L': 'HUEFS',
        # ------------------------
        'HUEF.EO': 'HUEFEO',
        'HUEF.FS': 'HUEFFS',
        'HUEF.PS': 'HUEFPS',
        # ------------------------
    }

    #
    # (4) Relevant dictionaries
    #

    # maps general class code to list of dicts where the -h attribute
    # descriptions will be pulled from
    dictionaries_of_interest = {
        'WSF': [
            roof_shape,
            secondary_water_resistance,
            roof_deck_attachment,
            roof_wall_connection,
            garage_presence,
            shutters,
            terrain_surface_roughness,
        ],
        'WMUH': [
            roof_shape,
            roof_cover,
            roof_quality,
            secondary_water_resistance,
            roof_deck_attachment,
            roof_wall_connection,
            shutters,
            terrain_surface_roughness,
        ],
        'MSF': [
            roof_shape,
            roof_wall_connection,
            roof_frame_type,
            roof_deck_attachment,
            shutters,
            secondary_water_resistance,
            garage_presence,
            masonry_reinforcing,
            roof_cover,
            terrain_surface_roughness,
        ],
        'MMUH': [
            roof_shape,
            secondary_water_resistance,
            roof_cover,
            roof_quality,
            roof_deck_attachment,
            roof_wall_connection,
            shutters,
            masonry_reinforcing,
            terrain_surface_roughness,
        ],
        'MLRM1': [
            roof_cover,
            shutters,
            masonry_reinforcing,
            wind_debris_environment,
            roof_frame_type,
            roof_deck_attachment,
            roof_wall_connection,
            roof_deck_age,
            roof_metal_deck_attachment_quality,
            terrain_surface_roughness,
        ],
        'MLRM2': [
            roof_cover,
            shutters,
            masonry_reinforcing,
            wind_debris_environment,
            roof_frame_type,
            roof_deck_attachment,
            roof_wall_connection,
            roof_deck_age,
            roof_metal_deck_attachment_quality,
            number_of_units,
            joist_spacing,
            terrain_surface_roughness,
        ],
        'MLRI': [
            shutters,
            masonry_reinforcing,
            roof_deck_age,
            roof_metal_deck_attachment_quality,
            terrain_surface_roughness,
        ],
        'MERB': [
            roof_cover,
            shutters,
            wind_debris_environment,
            roof_metal_deck_attachment_quality,
            window_area,
            terrain_surface_roughness,
        ],
        'MECB': [
            roof_cover,
            shutters,
            wind_debris_environment,
            roof_metal_deck_attachment_quality,
            window_area,
            terrain_surface_roughness,
        ],
        'CERB': [
            roof_cover,
            shutters,
            wind_debris_environment,
            window_area,
            terrain_surface_roughness,
        ],
        'CECB': [
            roof_cover,
            shutters,
            wind_debris_environment,
            window_area,
            terrain_surface_roughness,
        ],
        'SPMB': [
            shutters,
            roof_deck_age,
            roof_metal_deck_attachment_quality,
            terrain_surface_roughness,
        ],
        'SERB': [
            roof_cover,
            shutters,
            wind_debris_environment,
            roof_metal_deck_attachment_quality,
            window_area,
            terrain_surface_roughness,
        ],
        'SECB': [
            roof_cover,
            shutters,
            wind_debris_environment,
            roof_metal_deck_attachment_quality,
            window_area,
            terrain_surface_roughness,
        ],
        'MH': [shutters, tie_downs, terrain_surface_roughness],
        'HUEFH': [
            roof_cover,
            wind_debris_environment,
            roof_metal_deck_attachment_quality,
            shutters,
            terrain_surface_roughness,
        ],
        'HUEFS': [
            roof_cover,
            shutters,
            wind_debris_environment,
            roof_deck_age,
            roof_metal_deck_attachment_quality,
            terrain_surface_roughness,
        ],
        'HUEFEO': [
            roof_cover,
            shutters,
            wind_debris_environment,
            roof_metal_deck_attachment_quality,
            window_area,
            terrain_surface_roughness,
        ],
        'HUEFFS': [
            roof_cover,
            shutters,
            wind_debris_environment,
            roof_deck_age,
            roof_metal_deck_attachment_quality,
            terrain_surface_roughness,
        ],
        'HUEFPS': [
            roof_cover,
            shutters,
            wind_debris_environment,
            roof_metal_deck_attachment_quality,
            window_area,
            terrain_surface_roughness,
        ],
    }

    #
    # (5) Decode IDs and extend metadata with the individual records
    #

    # Create damage metadata

    fragility_data = pd.read_csv(source_file)

    with open(meta_file, encoding='utf-8') as f:  # noqa: PTH123
        meta_dict = json.load(f)

    # retrieve damage state descriptions and remove that part from
    # `hazus_hu_metadata`
    damage_state_classes = meta_dict.pop('DamageStateClasses')
    damage_state_descriptions = meta_dict.pop('DamageStateDescriptions')

    for fragility_id in fragility_data['ID'].to_list():
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # TODO(JVM): This is a temporary fix until we resolve the
        # presence of NaN values in the ID column of the fragility
        # library file.
        if pd.isna(fragility_id):
            continue
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        class_type = find_class_type(fragility_id)
        assert class_type is not None

        class_type_human_readable = class_types[class_type]

        general_class = general_classes[class_type]
        dictionaries = dictionaries_of_interest[general_class]
        remaining_chunks = fragility_id.replace(f'{class_type}.', '').split('.')
        assert len(remaining_chunks) == len(dictionaries)
        human_description = [class_type_human_readable]
        for chunk, dictionary in zip(remaining_chunks, dictionaries):
            human_description.append(dictionary[chunk])
        human_description_str = ' '.join(human_description)

        damage_state_class = damage_state_classes[class_type]
        damage_state_description = damage_state_descriptions[damage_state_class]

        limit_states = {}
        for damage_state, description in damage_state_description.items():
            limit_state = damage_state.replace('DS', 'LS')
            limit_states[limit_state] = {damage_state: {'Description': description}}

        record = {
            'Description': human_description_str,
            'SuggestedComponentBlockSize': '1 EA',
            'RoundUpToIntegerQuantity': 'True',
            'LimitStates': limit_states,
        }

        meta_dict[fragility_id] = record

    # save the metadata
    with open(target_meta_file_damage, 'w', encoding='utf-8') as f:  # noqa: PTH123
        json.dump(meta_dict, f, indent=2)

    # the unmodified damage state metadata are the same as those for
    # the fitted damage states.
    shutil.copy(target_meta_file_damage, target_meta_file_damage_original)

    # Create loss metadata & original loss function metadata

    with open(meta_file, encoding='utf-8') as f:  # noqa: PTH123
        meta_dict = json.load(f)

    # retrieve damage state descriptions and remove that part from
    # `hazus_hu_metadata`
    damage_state_classes = meta_dict.pop('DamageStateClasses')
    damage_state_descriptions = meta_dict.pop('DamageStateDescriptions')

    meta_dict_original = deepcopy(meta_dict)

    for fragility_id in fragility_data['ID'].to_list():
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # TODO(JVM): This is a temporary fix until we resolve the
        # presence of NaN values in the ID column of the fragility
        # library file.
        if pd.isna(fragility_id):
            continue
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        class_type = find_class_type(fragility_id)
        assert class_type is not None

        class_type_human_readable = class_types[class_type]

        general_class = general_classes[class_type]
        dictionaries = dictionaries_of_interest[general_class]
        remaining_chunks = fragility_id.replace(f'{class_type}.', '').split('.')
        assert len(remaining_chunks) == len(dictionaries)
        human_description = [class_type_human_readable]
        for chunk, dictionary in zip(remaining_chunks, dictionaries):
            human_description.append(dictionary[chunk])
        human_description_str = ' '.join(human_description)

        damage_state_class = damage_state_classes[class_type]
        damage_state_description = damage_state_descriptions[damage_state_class]

        damage_states = {}
        for damage_state, description in damage_state_description.items():
            damage_states[damage_state] = {'Description': description}

        record = {
            'Description': human_description_str,
            'SuggestedComponentBlockSize': '1 EA',
            'RoundUpToIntegerQuantity': 'True',
            'DamageStates': damage_states,
        }

        record_no_ds = {
            'Description': human_description_str,
            'SuggestedComponentBlockSize': '1 EA',
            'RoundUpToIntegerQuantity': 'True',
        }

        meta_dict[fragility_id] = record
        meta_dict_original[fragility_id] = record_no_ds

    # save the metadata
    with open(target_meta_file_loss, 'w', encoding='utf-8') as f:  # noqa: PTH123
        json.dump(meta_dict, f, indent=2)
    with open(target_meta_file_loss_original, 'w', encoding='utf-8') as f:  # noqa: PTH123
        json.dump(meta_dict_original, f, indent=2)


def main():
    """Generate Hazus Hurricane damage and loss database files."""
    create_hazus_hurricane_damage_loss_files()
    create_hazus_hurricane_metadata_files()


if __name__ == '__main__':
    main()
