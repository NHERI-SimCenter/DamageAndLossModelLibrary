"""
Generates Hazus Hurricane damage and loss database files.

"""

import os
import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import pandas as pd

warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", FutureWarning)


def parse_description(descr, parsed_data):
    """
    Parses the descr string and stores params in row Series

    Parameters
    ----------
    descr: string
        Hurricane archetype description from Hazus's raw database.
    parsed_data: pd.Series
        A pandas Series to collect parsed data.

    """

    # Roof Shape
    if "rsflt" in descr:
        parsed_data["roof_shape"] = "flt"
        descr = descr.replace("rsflt", "")
    elif "rsgab" in descr:
        parsed_data["roof_shape"] = "gab"
        descr = descr.replace("rsgab", "")
    elif "rship" in descr:
        parsed_data["roof_shape"] = "hip"
        descr = descr.replace("rship", "")

    # Secondary Water Resistance
    if "swrys" in descr:
        parsed_data["sec_water_res"] = True
        descr = descr.replace("swrys", "")
    elif "swrno" in descr:
        parsed_data["sec_water_res"] = False
        descr = descr.replace("swrno", "")

    # Roof Deck Attachment
    if "rda6d" in descr:
        parsed_data["roof_deck_attch"] = "6d"
        descr = descr.replace("rda6d", "")
    elif "rda8d" in descr:
        parsed_data["roof_deck_attch"] = "8d"
        descr = descr.replace("rda8d", "")
    elif "rda6s" in descr:
        parsed_data["roof_deck_attch"] = "6s"
        descr = descr.replace("rda6s", "")
    elif "rda8s" in descr:
        parsed_data["roof_deck_attch"] = "8s"
        descr = descr.replace("rda8s", "")

    # Roof Deck Attachment - Alternative Description
    if "rdast" in descr:
        parsed_data["roof_deck_attch"] = "st"  # standard
        descr = descr.replace("rdast", "")
    elif "rdasu" in descr:
        parsed_data["roof_deck_attch"] = "su"  # superior
        descr = descr.replace("rdasu", "")

    # Roof-Wall Connection
    if "tnail" in descr:
        parsed_data["roof_wall_conn"] = "tnail"
        descr = descr.replace("tnail", "")
    elif "strap" in descr:
        parsed_data["roof_wall_conn"] = "strap"
        descr = descr.replace("strap", "")

    # Garage
    if "gdnod" in descr:
        parsed_data["garage"] = "no"
        descr = descr.replace("gdnod", "")
    elif "gdno2" in descr:
        parsed_data["garage"] = "no"
        descr = descr.replace("gdno2", "")
    elif "gdstd" in descr:
        parsed_data["garage"] = "std"
        descr = descr.replace("gdstd", "")
    elif "gdwkd" in descr:
        parsed_data["garage"] = "wkd"
        descr = descr.replace("gdwkd", "")
    elif "gdsup" in descr:
        parsed_data["garage"] = "sup"
        descr = descr.replace("gdsup", "")

    # Shutters
    if "shtys" in descr:
        parsed_data["shutters"] = True
        descr = descr.replace("shtys", "")
    elif "shtno" in descr:
        parsed_data["shutters"] = False
        descr = descr.replace("shtno", "")

    # Roof Cover
    if "rcbur" in descr:
        parsed_data["roof_cover"] = "bur"
        descr = descr.replace("rcbur", "")
    elif "rcspm" in descr:
        parsed_data["roof_cover"] = "spm"
        descr = descr.replace("rcspm", "")

    # Roof Cover - Alternative Description
    if "rcshl" in descr:
        parsed_data["roof_cover"] = "cshl"  # cover, shingle
        descr = descr.replace("rcshl", "")
    elif "rsmtl" in descr:
        parsed_data["roof_cover"] = "smtl"  # sheet metal
        descr = descr.replace("rsmtl", "")

    # Roof Quality
    if "rqgod" in descr:
        parsed_data["roof_quality"] = "god"
        descr = descr.replace("rqgod", "")
    elif "rqpor" in descr:
        parsed_data["roof_quality"] = "por"
        descr = descr.replace("rqpor", "")

    # Masonry Reinforcing
    if "rmfys" in descr:
        parsed_data["masonry_reinforcing"] = True
        descr = descr.replace("rmfys", "")
    elif "rmfno" in descr:
        parsed_data["masonry_reinforcing"] = False
        descr = descr.replace("rmfno", "")

    # Roof Frame Type
    if "rftrs" in descr:
        parsed_data["roof_frame_type"] = "trs"  # wood truss
        descr = descr.replace("rftrs", "")
    elif "rfows" in descr:
        parsed_data["roof_frame_type"] = "ows"  # OWSJ
        descr = descr.replace("rfows", "")

    # Wind Debris Environment
    if "widdA" in descr:
        parsed_data["wind_debris"] = "A"  # res/comm.
        descr = descr.replace("widdA", "")
    elif "widdB" in descr:
        parsed_data["wind_debris"] = "B"  # varies by direction
        descr = descr.replace("widdB", "")
    elif "widdC" in descr:
        parsed_data["wind_debris"] = "C"  # residential
        descr = descr.replace("widdC", "")
    elif "widdD" in descr:
        parsed_data["wind_debris"] = "D"  # none
        descr = descr.replace("widdD", "")

    # Roof Deck Age
    if "dqgod" in descr:
        parsed_data["roof_deck_age"] = "god"  # new or average
        descr = descr.replace("dqgod", "")
    elif "dqpor" in descr:
        parsed_data["roof_deck_age"] = "por"  # old
        descr = descr.replace("dqpor", "")

    # Metal Roof Deck Attachment
    if "rd100" in descr:
        parsed_data["metal_rda"] = "std"  # standard
        descr = descr.replace("rd100", "")
    elif "rd110" in descr:
        parsed_data["metal_rda"] = "sup"  # superior
        descr = descr.replace("rd110", "")

    # Number of Units
    if "nusgl" in descr:
        parsed_data["num_of_units"] = "sgl"
        descr = descr.replace("nusgl", "")
    elif "numlt" in descr:
        parsed_data["num_of_units"] = "mlt"
        descr = descr.replace("numlt", "")

    # Joist Spacing
    if "jspa4" in descr:
        parsed_data["joist_spacing"] = "4"
        descr = descr.replace("jspa4", "")
    elif "jspa6" in descr:
        parsed_data["joist_spacing"] = "6"
        descr = descr.replace("jspa6", "")

    # Window Area
    if "walow" in descr:
        parsed_data["window_area"] = "low"
        descr = descr.replace("walow", "")
    elif "wamed" in descr:
        parsed_data["window_area"] = "med"
        descr = descr.replace("wamed", "")
    elif "wahig" in descr:
        parsed_data["window_area"] = "hig"
        descr = descr.replace("wahig", "")

    # ----- unknown attributes ---------

    if "uprys" in descr:
        parsed_data["upgrade_??"] = True
        descr = descr.replace("uprys", "")
    elif "uprno" in descr:
        parsed_data["upgrade_??"] = False
        descr = descr.replace("uprno", "")

    if "wcdbl" in descr:
        parsed_data["wall_cover_??"] = "dbl"
        descr = descr.replace("wcdbl", "")
    elif "wcsgl" in descr:
        parsed_data["wall_cover_??"] = "sgl"
        descr = descr.replace("wcsgl", "")

    if "tspa2" in descr:
        parsed_data["tspa_??"] = "2"
        descr = descr.replace("tspa2", "")
    elif "tspa4" in descr:
        parsed_data["tspa_??"] = "4"
        descr = descr.replace("tspa4", "")

    if "mtdys" in descr:
        parsed_data["tie_downs"] = True
        descr = descr.replace("mtdys", "")
    elif "mtdno" in descr:
        parsed_data["tie_downs"] = False
        descr = descr.replace("mtdno", "")

    return descr


def main():

    # Load RAW Hazus data

    raw_data_path = "input_files/"

    # read bldg data

    bldg_df_ST = pd.read_excel(
        raw_data_path + "huListOfWindBldgTypes.xlsx", index_col=0
    )
    bldg_df_EF = pd.read_excel(
        raw_data_path + "huListOfWindBldgTypesEF.xlsx", index_col=0
    )

    # make sure the column headers are in sync
    bldg_df_EF.columns = ["sbtName", *bldg_df_EF.columns[1:]]

    # offset the EF building IDs to ensure each archetype has a unique ID
    bldg_df_EF.index = max(bldg_df_ST.index) + bldg_df_EF.index
    bldg_df_EF.sort_index(inplace=True)

    bldg_df = pd.concat([bldg_df_ST, bldg_df_EF], axis=0)

    # read fragility data

    frag_df_ST = pd.read_excel(raw_data_path + "huDamLossFun.xlsx")

    frag_df_EF = pd.read_excel(raw_data_path + "huDamLossFunEF.xlsx")
    frag_df_EF['wbID'] += max(bldg_df_ST.index)

    frag_df = pd.concat([frag_df_ST, frag_df_EF], axis=0, ignore_index=True)

    frag_df.sort_values(['wbID', 'TERRAINID', 'DamLossDescID'], inplace=True)

    frag_df.reset_index(drop=True, inplace=True)

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
    wind_speeds_str = [c for c in frag_df.columns if "WS" in c]

    # also get a list of floats based on the above labels
    wind_speeds = np.array([float(ws[2:]) for ws in wind_speeds_str])

    # set the max wind speed of interest
    max_speed = 200
    max_speed_id = max(np.where(wind_speeds <= max_speed)[0]) + 1

    DS_data = [
        frag_df[frag_df['DamLossDescID'] == ds].loc[:, wind_speeds_str]
        for ds in range(1, 5)
    ]

    # the problem affects DS4 probabilities
    archetypes = (DS_data[2] - DS_data[3].values < -0.02).max(axis=1)
    # go through each affected archetype and fix the problem
    for frag_id in archetypes[archetypes == True].index:  # noqa
        # get the wbID and terrain_id
        wbID, terrain_id = frag_df.loc[frag_id, ['wbID', 'TERRAINID']]

        # load the fragility info for the archetype
        frag_df_arch = frag_df.loc[
            (frag_df['wbID'] == wbID) & (frag_df['TERRAINID'] == terrain_id)
        ]

        # check which DS is stored as DS4
        # we do this by looking at the median capacities at each DS
        # through simple interpolation
        median_capacities = [
            np.interp(
                0.5, frag_df_arch[wind_speeds_str].iloc[ds].values, wind_speeds
            )
            for ds in range(4)
        ]

        # then check where to store the values at DS4 to maintain
        # ascending exceedance probabilities
        target_DS = np.where(np.argsort(median_capacities) == 3)[0][0]

        # since this is always DS1 in the current database,
        # the script below works with that assumption and checks for exceptions
        if target_DS == 0:
            # first, extract the probabilities stored at DS4
            DS4_probs = frag_df_arch[wind_speeds_str].iloc[3].values

            # then offset the probabilities of DS1-3 by one level
            for ds in [3, 2, 1]:
                source_DS_index = frag_df_arch.index[ds - 1]
                target_DS_index = frag_df_arch.index[ds]

                frag_df.loc[target_DS_index, wind_speeds_str] = frag_df.loc[
                    source_DS_index, wind_speeds_str
                ].values

            # finally store the DS4 probs at the DS1 cells
            target_DS_index = frag_df_arch.index[0]

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
    DS_data = frag_df[frag_df['DamLossDescID'].isin([1, 2, 3, 4])].loc[
        :, wind_speeds_str
    ]

    # # check for invalid values
    # print(f'Global minimum value: {np.min(DS_data.values)}')
    # print(f'Global maximum value: {np.max(DS_data.values)}')

    # sum up the probabilities of exceeding each DS at various wind speeds
    DS_zero = DS_data.sum(axis=1)

    # and look for the lines where the sum is zero - i.e., all values are zero
    no_DS_info = frag_df.loc[DS_zero[DS_zero == 0].index]

    def overwrite_ds4_data():
        # now go through the building types in no_DS_info
        for wbID in no_DS_info['wbID'].unique():
            # and each terrain type that is affected
            for terrain_id in no_DS_info.loc[
                no_DS_info['wbID'] == wbID, 'TERRAINID'
            ].values:
                # get the fragility data for each archetype
                frag_df_arch = frag_df.loc[
                    (frag_df['wbID'] == wbID) & (frag_df['TERRAINID'] == terrain_id)
                ]

                # extract the DS3 information
                DS3_data = frag_df_arch.loc[
                    frag_df['DamLossDescID'] == 3, wind_speeds_str
                ].values

                # and overwrite the DS4 values in the original dataset
                DS4_index = frag_df_arch.loc[frag_df['DamLossDescID'] == 4].index
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
        "bldg_type",
        "roof_shape",
        "roof_cover",
        "roof_quality",
        "sec_water_res",
        "roof_deck_attch",
        "roof_wall_conn",
        "garage",
        "shutters",
        "terr_rough",
        "upgrade_??",
        "wall_cover_??",
        "tspa_??",
        "masonry_reinforcing",
        "roof_frame_type",
        "wind_debris",
        "roof_deck_age",
        "metal_rda",
        "num_of_units",
        "joist_spacing",
        "window_area",
        "tie_downs",
        "DS1_dist",
        "DS1_mu",
        "DS1_sig",
        "DS1_fit",
        "DS1_meps",
        "DS2_dist",
        "DS2_mu",
        "DS2_sig",
        "DS2_fit",
        "DS2_meps",
        "DS3_dist",
        "DS3_mu",
        "DS3_sig",
        "DS3_fit",
        "DS3_meps",
        "DS4_dist",
        "DS4_mu",
        "DS4_sig",
        "DS4_fit",
        "DS4_meps",
        "L1",
        "L2",
        "L3",
        "L4",
        "L_fit",
        "L_meps",
        "DS1_original",
        "DS2_original",
        "DS3_original",
        "DS4_original",
        "L_original",
    ]

    # resulting dataframe
    new_df = pd.DataFrame(
        None, columns=column_names, index=np.arange(len(flt_bldg_df.index) * 5)
    )

    rows = []

    # calculation
    for index, row in flt_bldg_df.iterrows():
        if index % 100 == 0:
            print(index)

        # initialize the row for the archetype
        new_row = pd.Series(index=new_df.columns, dtype=np.float64)

        # store building type
        new_row['bldg_type'] = row['sbtName']

        # then parse the description and store the recognized parameter values
        descr = parse_description(row['charDescription'].strip(), new_row)

        # check if any part of the description remained unparsed
        if descr != "":
            print('WARNING', index, descr)

        # filter only those parts of the frag_df that correspond to
        # this archetype
        frag_df_arch = frag_df[frag_df["wbID"] == index]

        # cycle through the five terrain types in Hazus
        for terrain_id, roughness in enumerate([0.03, 0.15, 0.35, 0.7, 1.0]):
            # Hazus array indexing is 1-based
            terrain_id += 1

            new_row_terrain = new_row.copy()

            # store the roughness length
            new_row_terrain["terr_rough"] = roughness

            # filter only those parts of the frag_df_arch that correspond
            # to this terrain type
            frag_df_arch_terrain = frag_df_arch[
                frag_df_arch["TERRAINID"] == terrain_id
            ]

            mu_min = 0

            # for each damage state
            for DS in [1, 2, 3, 4]:
                # get the exceedence probabilities for this DS of this
                # archetype
                P_exc = np.asarray(
                    frag_df_arch_terrain.loc[
                        frag_df_arch_terrain["DamLossDescID"] == DS, wind_speeds_str
                    ].values[0]
                )
                multilinear_CDF_parameters = (
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
                def MSE_normal(params, mu_min, res_type='MSE'):
                    # unpack the parameters
                    mu, sig = params

                    # penalize invalid params
                    if (np.round(mu, decimals=1) <= mu_min) or (sig <= 0):
                        return 1e10

                    eps = (norm.cdf(wind_speeds, loc=mu, scale=sig) - P_exc)[
                        :max_speed_id_mod
                    ]

                    if res_type == 'MSE':
                        MSE = sum(eps**2.0)

                        return MSE

                    elif res_type == 'max abs eps':
                        return max(abs(eps))

                    elif res_type == 'eps':
                        return eps

                # assuming Lognormal distribution for building capacity
                def MSE_lognormal(params, mu_min, res_type='MSE'):
                    # unpack the parameters
                    mu, beta = params

                    # penalize invalid params
                    if (np.round(mu, decimals=1) <= mu_min) or (beta <= 0):
                        return 1e10

                    eps = (
                        norm.cdf(np.log(wind_speeds), loc=np.log(mu), scale=beta)
                        - P_exc
                    )[:max_speed_id_mod]

                    if res_type == 'MSE':
                        MSE = sum(eps**2.0)

                        return MSE

                    elif res_type == 'max abs eps':
                        return max(abs(eps))

                    elif res_type == 'eps':
                        return eps

                # minimize MSE assuming Normal distribution
                res_normal = minimize(
                    MSE_normal,
                    [mu_0, sig_0],
                    args=(mu_min),
                    method='BFGS',
                    options=dict(maxiter=50),
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
                    options=dict(maxiter=50),
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
                        print(
                            f"WARNING: Error in CDF fitting "
                            f"for {index}, {terrain_id}, {DS}"
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

                else:
                    if (np.log(res_normal.fun / res_lognormal.fun) < 0.1) and (
                        np.log(res_lognormal.maxcv / res_normal.maxcv) > 0.1
                    ):
                        dist_type = 'normal'
                        res = res_normal

                    else:
                        dist_type = 'lognormal'
                        res = res_lognormal

                # store the parameters
                new_row_terrain[f"DS{DS}_dist"] = dist_type
                new_row_terrain[f"DS{DS}_mu"] = res.x[0]
                new_row_terrain[f"DS{DS}_sig"] = res.x[1]
                new_row_terrain[f"DS{DS}_fit"] = res.fun
                new_row_terrain[f"DS{DS}_meps"] = res.maxcv
                new_row_terrain[f"DS{DS}_original"] = multilinear_CDF_parameters

                # consecutive damage states should have increasing capacities
                mu_min = res.x[0]

            # Now we have the damages, continue with Losses

            # Focus on "Building losses" first
            L_ref = np.asarray(
                frag_df_arch_terrain.loc[
                    frag_df_arch_terrain["DamLossDescID"] == 5, wind_speeds_str
                ].values[0]
            )

            multilinear_CDF_parameters = (
                ','.join([str(x) for x in L_ref])
                + '|'
                + ','.join([str(x) for x in wind_speeds])
            )

            # We'll need the probability of each Damage State across the
            # pre-defined wind speeds
            DS_probs = np.zeros((4, len(wind_speeds)))

            for DS_id, DS in enumerate([1, 2, 3, 4]):
                if new_row_terrain[f"DS{DS}_dist"] == "normal":
                    DS_probs[DS_id] = norm.cdf(
                        wind_speeds,
                        loc=new_row_terrain[f"DS{DS}_mu"],
                        scale=new_row_terrain[f"DS{DS}_sig"],
                    )
                else:
                    DS_probs[DS_id] = norm.cdf(
                        np.log(wind_speeds),
                        loc=np.log(new_row_terrain[f"DS{DS}_mu"]),
                        scale=new_row_terrain[f"DS{DS}_sig"],
                    )

            # The losses for DS4 are calculated based on outcomes at the
            # highest wind speeds
            L_max = frag_df_arch_terrain.loc[
                frag_df_arch_terrain["DamLossDescID"] == 5, "WS250"
            ].values[0]
            DS4_max = DS_probs[3][-1]

            L4 = np.round(min(L_max / DS4_max, 1.0), decimals=3)

            # if L4 < 0.75:
            #    print(index, terrain_id, L_max, DS4_max, L4)

            for i in range(3):
                DS_probs[i] = DS_probs[i] - DS_probs[i + 1]

            # define the loss error measures to be minimized

            #
            def SSE_loss(params, res_type='SSE'):
                loss_ratios = params.copy()

                # assume 1.0 for DS4
                loss_ratios = np.append(loss_ratios, L4)

                L_est = np.sum(loss_ratios * DS_probs.T, axis=1)

                # the error is the difference between reference and estimated losses
                eps = (L_est - L_ref)[:max_speed_id]

                if res_type == 'SSE':
                    # calculate the sum of squared errors across wind speeds
                    SSE = sum(eps**2.0)

                elif res_type == 'max abs eps':
                    return max(abs(eps))

                return SSE

            cons = (
                {'type': 'ineq', 'fun': lambda x: x[1] - x[0] - 0.002},
                {'type': 'ineq', 'fun': lambda x: x[2] - x[1] - 0.002},
                {'type': 'ineq', 'fun': lambda x: L4 - x[2] - 0.002},
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
            new_row_terrain["L1"] = res.x[0]
            new_row_terrain["L2"] = res.x[1]
            new_row_terrain["L3"] = res.x[2]
            new_row_terrain["L4"] = L4
            new_row_terrain["L_original"] = multilinear_CDF_parameters
            new_row_terrain["L_fit"] = res.fun
            new_row_terrain["L_meps"] = res.maxcv

            # display(new_row.to_frame().T)

            rows.append(new_row_terrain.to_frame().T)

    main_df = pd.concat(rows, axis=0, ignore_index=True)

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
    out_df['ID'] = ""

    # some general formatting to make file name generation easier
    out_df['shutters'] = out_df['shutters'].astype(int)
    out_df['terr_rough'] = (out_df['terr_rough'] * 100.0).astype(int)

    for index, row in out_df.iterrows():
        if index % 1000 == 0:
            print(index)

        # define the name of the building damage and loss configuration
        bldg_type = row["bldg_type"]
        critical_cols = []

        if bldg_type[:3] == "WSF":
            cols_of_interest = [
                "bldg_type",
                "roof_shape",
                "sec_water_res",
                "roof_deck_attch",
                "roof_wall_conn",
                "garage",
                "shutters",
                "terr_rough",
            ]
            critical_cols = ["roof_deck_attch", "garage"]

        elif bldg_type[:4] == "WMUH":
            cols_of_interest = [
                "bldg_type",
                "roof_shape",
                "roof_cover",
                "roof_quality",
                "sec_water_res",
                "roof_deck_attch",
                "roof_wall_conn",
                "shutters",
                "terr_rough",
            ]

        elif bldg_type[:3] == "MSF":
            cols_of_interest = [
                "bldg_type",
                "roof_shape",
                "roof_wall_conn",
                "roof_frame_type",
                "roof_deck_attch",
                "shutters",
                "sec_water_res",
                "garage",
                "masonry_reinforcing",
                "roof_cover",
                "terr_rough",
            ]

        elif bldg_type[:4] == "MMUH":
            cols_of_interest = [
                "bldg_type",
                "roof_shape",
                "sec_water_res",
                "roof_cover",
                "roof_quality",
                "roof_deck_attch",
                "roof_wall_conn",
                "shutters",
                "masonry_reinforcing",
                "terr_rough",
            ]

        elif bldg_type[:5] == "MLRM1":
            cols_of_interest = [
                "bldg_type",
                "roof_cover",
                "shutters",
                "masonry_reinforcing",
                "wind_debris",
                "roof_frame_type",
                "roof_deck_attch",
                "roof_wall_conn",
                "roof_deck_age",
                "metal_rda",
                "terr_rough",
            ]

        elif bldg_type[:5] == "MLRM2":
            cols_of_interest = [
                "bldg_type",
                "roof_cover",
                "shutters",
                "masonry_reinforcing",
                "wind_debris",
                "roof_frame_type",
                "roof_deck_attch",
                "roof_wall_conn",
                "roof_deck_age",
                "metal_rda",
                "num_of_units",
                "joist_spacing",
                "terr_rough",
            ]

        elif bldg_type[:4] == "MLRI":
            cols_of_interest = [
                "bldg_type",
                "shutters",
                "masonry_reinforcing",
                "roof_deck_age",
                "metal_rda",
                "terr_rough",
            ]

        elif bldg_type[:4] == "MERB":
            cols_of_interest = [
                "bldg_type",
                "roof_cover",
                "shutters",
                "wind_debris",
                "metal_rda",
                "window_area",
                "terr_rough",
            ]

        elif bldg_type[:4] == "MECB":
            cols_of_interest = [
                "bldg_type",
                "roof_cover",
                "shutters",
                "wind_debris",
                "metal_rda",
                "window_area",
                "terr_rough",
            ]

        elif bldg_type[:4] == "CERB":
            cols_of_interest = [
                "bldg_type",
                "roof_cover",
                "shutters",
                "wind_debris",
                "window_area",
                "terr_rough",
            ]

        elif bldg_type[:4] == "CECB":
            cols_of_interest = [
                "bldg_type",
                "roof_cover",
                "shutters",
                "wind_debris",
                "window_area",
                "terr_rough",
            ]

        elif bldg_type[:4] == "SERB":
            cols_of_interest = [
                "bldg_type",
                "roof_cover",
                "shutters",
                "wind_debris",
                "metal_rda",
                "window_area",
                "terr_rough",
            ]

        elif bldg_type[:4] == "SECB":
            cols_of_interest = [
                "bldg_type",
                "roof_cover",
                "shutters",
                "wind_debris",
                "metal_rda",
                "window_area",
                "terr_rough",
            ]

        elif bldg_type[:4] == "SPMB":
            cols_of_interest = [
                "bldg_type",
                "shutters",
                "roof_deck_age",
                "metal_rda",
                "terr_rough",
            ]

        elif bldg_type[:2] == "MH":
            cols_of_interest = ["bldg_type", "shutters", "tie_downs", "terr_rough"]

            critical_cols = [
                "tie_downs",
            ]

        elif bldg_type[:6] == "HUEFFS":
            cols_of_interest = [
                "bldg_type",
                "roof_cover",
                "shutters",
                "wind_debris",
                "roof_deck_age",
                "metal_rda",
                "terr_rough",
            ]

        elif bldg_type[:6] == "HUEFPS":
            cols_of_interest = [
                "bldg_type",
                "roof_cover",
                "shutters",
                "wind_debris",
                "metal_rda",
                "window_area",
                "terr_rough",
            ]

        elif bldg_type[:6] == "HUEFEO":
            cols_of_interest = [
                "bldg_type",
                "roof_cover",
                "shutters",
                "wind_debris",
                "metal_rda",
                "window_area",
                "terr_rough",
            ]

        elif bldg_type[:5] == "HUEFH":
            cols_of_interest = [
                "bldg_type",
                "roof_cover",
                "wind_debris",
                "metal_rda",
                "shutters",
                "terr_rough",
            ]

        elif bldg_type[:5] == "HUEFS":
            cols_of_interest = [
                "bldg_type",
                "roof_cover",
                "shutters",
                "wind_debris",
                "roof_deck_age",
                "metal_rda",
                "terr_rough",
            ]

        else:
            continue

        bldg_chars = row[cols_of_interest]

        # If a critical feature is underfined, we consider the archetype
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
        if bldg_type.startswith("MSF"):
            if bldg_chars["roof_cover"] in ['smtl', 'cshl']:
                bldg_chars['roof_frame_type'] = 'ows'
            else:
                bldg_chars['roof_frame_type'] = 'trs'

        bldg_chars["bldg_type"] = bldg_type_map[bldg_chars["bldg_type"]]

        out_df.loc[index, 'ID'] = ".".join(bldg_chars.astype(str))

    # the out_df we have in the end is the direct input to generate the
    # damage and loss data
    # -> so that we don't have to go through fitting everything every time
    # -> we need to re-generate the data for some reason
    out_df.head(10)

    # ## Export to CSV file

    # import dill
    # dill.load_session('session.dill')

    output_directory = 'output'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # initialize the fragility table
    df_db_fit = pd.DataFrame(
        columns=[
            "ID",
            "Incomplete",
            "Demand-Type",
            "Demand-Unit",
            "Demand-Offset",
            "Demand-Directional",
            "LS1-Family",
            "LS1-Theta_0",
            "LS1-Theta_1",
            "LS2-Family",
            "LS2-Theta_0",
            "LS2-Theta_1",
            "LS3-Family",
            "LS3-Theta_0",
            "LS3-Theta_1",
            "LS4-Family",
            "LS4-Theta_0",
            "LS4-Theta_1",
        ],
        index=out_df.index,
        dtype=float,
    )

    df_db_original = pd.DataFrame(
        columns=[
            "ID",
            "Incomplete",
            "Demand-Type",
            "Demand-Unit",
            "Demand-Offset",
            "Demand-Directional",
            "LS1-Family",
            "LS1-Theta_0",
            "LS2-Family",
            "LS2-Theta_0",
            "LS3-Family",
            "LS3-Theta_0",
            "LS4-Family",
            "LS4-Theta_0",
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

    for LS_i in range(1, 5):

        df_db_original[f'LS{LS_i}-Family'] = 'multilinear_CDF'
        df_db_original[f'LS{LS_i}-Theta_0'] = out_df[f'DS{LS_i}_original']

        df_db_fit[f'LS{LS_i}-Family'] = out_df[f'DS{LS_i}_dist']
        df_db_fit[f'LS{LS_i}-Theta_0'] = out_df[f'DS{LS_i}_mu']
        df_db_fit[f'LS{LS_i}-Theta_1'] = out_df[f'DS{LS_i}_sig']

    for df_db in (df_db_original, df_db_fit):
        df_db = df_db.loc[df_db['ID'] != '']
        df_db = df_db.set_index("ID").sort_index().convert_dtypes()

    df_db_fit.to_csv(
        f'{output_directory}/damage_DB_SimCenter_Hazus_HU_bldg_fitted.csv'
    )
    df_db_original.to_csv(
        f'{output_directory}/damage_DB_SimCenter_Hazus_HU_bldg_original.csv'
    )

    # initialize the output loss table
    # define the columns
    out_cols = [
        "ID",
        "Incomplete",
        "Demand-Type",
        "Demand-Unit",
        "Demand-Offset",
        "Demand-Directional",
        "DV-Unit",
        "LossFunction-Theta_0",
    ]
    df_db_original = pd.DataFrame(columns=out_cols, index=out_df.index, dtype=float)
    df_db_original['ID'] = [f'{id}-Cost' for id in out_df['ID']]
    df_db_original['Incomplete'] = 0
    df_db_original['Demand-Type'] = 'Peak Gust Wind Speed'
    df_db_original['Demand-Unit'] = 'mph'
    df_db_original['Demand-Offset'] = 0
    df_db_original['Demand-Directional'] = 0
    df_db_original['DV-Unit'] = 'loss_ratio'
    df_db_original['LossFunction-Theta_0'] = out_df['L_original']
    df_db_original = df_db_original.loc[df_db_original['ID'] != '-Cost']
    df_db_original = df_db_original.set_index("ID").sort_index().convert_dtypes()

    out_cols = [
        "Incomplete",
        "Quantity-Unit",
        "DV-Unit",
    ]
    for DS_i in range(1, 5):
        out_cols += [f"DS{DS_i}-Theta_0"]
    df_db_fit = pd.DataFrame(columns=out_cols, index=out_df.index, dtype=float)
    df_db_fit['ID'] = [f'{id}-Cost' for id in out_df['ID']]
    df_db_fit['Incomplete'] = 0
    df_db_fit['Quantity-Unit'] = '1 EA'
    df_db_fit['DV-Unit'] = 'loss_ratio'
    for LS_i in range(1, 5):
        df_db_fit[f'DS{LS_i}-Theta_0'] = out_df[f'L{LS_i}']
    df_db_fit = df_db_fit.loc[df_db_fit['ID'] != '-Cost']
    df_db_fit = df_db_fit.set_index("ID").sort_index().convert_dtypes()

    df_db_fit.to_csv(
        f'{output_directory}/loss_repair_DB_SimCenter_Hazus_HU_bldg_fitted.csv'
    )
    df_db_original.to_csv(
        f'{output_directory}/loss_repair_DB_SimCenter_Hazus_HU_bldg_original.csv'
    )


if __name__ == '__main__':
    main()
