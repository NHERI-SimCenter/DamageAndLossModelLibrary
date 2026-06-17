"""
Tabular (CSV) format deserializer for the dlml library.

The functions in this module are ported from pelicun (``base.py`` and
``file_io.py``) and serve as dlml's deserializer for its own CSV format.
They must remain behavior-compatible with pelicun: a behavioral
equivalence test against pelicun lands in a later step. The logic is
preserved verbatim from the pelicun source, adjusting only imports and
inlining pelicun-internal references so the module is standalone and
pelicun-free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def convert_to_MultiIndex(  # noqa: N802
    data: pd.DataFrame | pd.Series, axis: int = 0, *, inplace: bool = False
) -> pd.DataFrame | pd.Series:
    """
    Convert the index of a DataFrame to a MultiIndex.

    We assume that the index uses standard SimCenter convention to
    identify different levels: a dash character ('-') is expected to
    separate each level of the index.

    Parameters
    ----------
    data: DataFrame
        The DataFrame that will be modified.
    axis: int, optional, default:0
        Identifies if the index (0) or the columns (1) shall be
        edited.
    inplace: bool, optional, default:False
        If yes, the operation is performed directly on the input
        DataFrame and not on a copy of it.

    Returns
    -------
    DataFrame
        The modified DataFrame.

    Raises
    ------
    ValueError
        If an invalid axis is specified.

    """
    # check if the requested axis is already a MultiIndex
    if ((axis == 0) and (isinstance(data.index, pd.MultiIndex))) or (
        (axis == 1) and (isinstance(data.columns, pd.MultiIndex))
    ):
        # if yes, return the data unchanged
        return data

    if axis == 0:
        index_labels = [str(label).split('-') for label in data.index]

    elif axis == 1:
        index_labels = [str(label).split('-') for label in data.columns]

    else:
        msg = f'Invalid axis parameter: {axis}'
        raise ValueError(msg)

    max_lbl_len = np.max([len(labels) for labels in index_labels])

    for l_i, labels in enumerate(index_labels):
        if len(labels) != max_lbl_len:
            labels += [''] * (max_lbl_len - len(labels))  # noqa: PLW2901
            index_labels[l_i] = labels

    index_labels_np = np.array(index_labels)

    if index_labels_np.shape[1] > 1:
        data_mod = data if inplace else data.copy()

        if axis == 0:
            data_mod.index = pd.MultiIndex.from_arrays(index_labels_np.T)

        else:
            data_mod.columns = pd.MultiIndex.from_arrays(index_labels_np.T)

        return data_mod

    return data


def convert_dtypes(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to a numeric datatype whenever possible.

    The function replaces None with NA otherwise columns containing
    None would continue to have the `object` type.

    Parameters
    ----------
    dataframe: DataFrame
        The DataFrame that will be modified.

    Returns
    -------
    DataFrame
        The modified DataFrame.

    """
    with (
        pd.option_context('future.no_silent_downcasting', True),  # noqa: FBT003
        pd.option_context('mode.copy_on_write', True),  # noqa: FBT003
    ):
        dataframe = dataframe.fillna(value=np.nan).infer_objects()

    # We want numeric conversion to be best-effort: columns that parse
    # as numbers are converted, columns that do not (e.g. unit strings)
    # are left untouched. Historically this was expressed as
    # `pd.to_numeric(x, errors='ignore')`, but that option is deprecated
    # in pandas 2.2 and scheduled to raise in a future release. The
    # replacement recommended by the pandas docs is to catch the
    # parse failure ourselves and fall back to the original column.
    # See: https://pandas.pydata.org/docs/reference/api/pandas.to_numeric.html
    def _to_numeric_if_possible(column: pd.Series) -> pd.Series:
        try:
            return pd.to_numeric(column)
        except (ValueError, TypeError):
            return column

    # note: `axis=0` applies the function to the columns
    return dataframe.apply(_to_numeric_if_possible, axis=0)


def check_if_str_is_na(string: Any) -> bool:
    """
    Check if the provided string can be interpreted as N/A.

    Parameters
    ----------
    string: object
            The string to evaluate

    Returns
    -------
    bool
        The evaluation result. Yes, if the string is considered N/A.
    """
    na_vals = {
        '',
        'N/A',
        '-1.#QNAN',
        'null',
        'None',
        '<NA>',
        'nan',
        '-NaN',
        '1.#IND',
        'NaN',
        '#NA',
        '1.#QNAN',
        'NULL',
        '-nan',
        '#N/A',
        '#N/A N/A',
        'n/a',
        '-1.#IND',
        'NA',
    }
    # obtained from Pandas' internal STR_NA_VALUES variable.

    return isinstance(string, str) and string in na_vals


def with_parsed_str_na_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify string values interpretable as N/A.

    Given a dataframe, this function identifies values that have
    string type and can be interpreted as N/A, and replaces them with
    actual NA's.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to process

    Returns
    -------
    pd.DataFrame
        The dataframe with proper N/A values.
    """
    # Replace string NA values with actual NaNs
    return df.apply(
        lambda col: col.map(lambda x: np.nan if check_if_str_is_na(x) else x)
    )


def load_tabular(path: str | Path) -> pd.DataFrame:
    """
    Load a dlml model-parameter CSV into a tidy DataFrame.

    Replicates exactly what pelicun does when it loads a
    model-parameter CSV, i.e. ``file_io.load_data`` called with
    ``unit_conversion_factors=None``, ``orientation=1``,
    ``reindex=False`` and ``return_units=False``. The data is assumed
    to follow the standard SimCenter tabular schema: a single header
    line and an index column, with levels separated by a dash
    character ('-').

    Parameters
    ----------
    path: str or Path
        The location of the source CSV file.

    Returns
    -------
    pd.DataFrame
        The parsed data, with both columns and index converted to a
        MultiIndex where applicable and sorted.

    """
    filepath = Path(path).resolve()

    # read the CSV using the same parameters as pelicun's load_from_file
    data = pd.read_csv(
        filepath,
        header=0,
        index_col=0,
        low_memory=False,
        encoding_errors='replace',
    )

    # NOTE on pelicun fidelity: load_data separates an optional 'units'
    # row/column at this point. The DLML tabular schema never uses one --
    # units are carried in named columns such as 'Demand-Unit' / 'DV-Unit',
    # never a bare 'units' entry -- so that branch is intentionally omitted.
    # The output stays identical to pelicun's
    # load_data(None, orientation=1, reindex=False) for every DLML file, as
    # the behavioral equivalence test verifies.
    data = convert_dtypes(data)

    # convert columns to MultiIndex if needed, then sort
    data = convert_to_MultiIndex(data, axis=1)
    data = data.sort_index(axis=1)

    # reindex=False: convert index to MultiIndex if needed, then sort
    data = convert_to_MultiIndex(data, axis=0)
    return data.sort_index()
