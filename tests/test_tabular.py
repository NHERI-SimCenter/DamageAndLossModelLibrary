"""Standalone unit tests for :mod:`dlml._tabular`.

These tests are pelicun-free and do not access the network. They cover
the ported deserializer helpers and the public ``load_tabular`` entry
point against the real library CSV files.
"""

from importlib import resources

import numpy as np
import pandas as pd
import pytest

from dlml._tabular import (
    check_if_str_is_na,
    convert_dtypes,
    convert_to_MultiIndex,
    load_tabular,
    with_parsed_str_na_values,
)


def _data_path(*parts):
    """Return a path to a file under the dlml ``data`` directory."""
    path = resources.files('dlml') / 'data'
    for part in parts:
        path = path / part
    return path


# ---------------------------------------------------------------------------
# convert_to_MultiIndex
# ---------------------------------------------------------------------------


def test_convert_to_multiindex_splits_hyphenated_columns():
    frame = pd.DataFrame(
        [[1, 2, 3]],
        columns=['LS1-Theta_0', 'LS1-Theta_1', 'Demand-Type'],
    )
    result = convert_to_MultiIndex(frame, axis=1)
    assert isinstance(result.columns, pd.MultiIndex)
    assert result.columns.nlevels == 2
    assert ('LS1', 'Theta_0') in result.columns
    assert ('Demand', 'Type') in result.columns


def test_convert_to_multiindex_idempotent_on_multiindex():
    cols = pd.MultiIndex.from_tuples([('LS1', 'Theta_0'), ('LS1', 'Theta_1')])
    frame = pd.DataFrame([[1, 2]], columns=cols)
    result = convert_to_MultiIndex(frame, axis=1)
    # already a MultiIndex: returned unchanged
    assert result is frame
    assert result.columns.equals(cols)


def test_convert_to_multiindex_pads_uneven_depths():
    frame = pd.DataFrame(
        [[1, 2]],
        columns=['LS1-Theta_0', 'Incomplete'],
    )
    result = convert_to_MultiIndex(frame, axis=1)
    assert isinstance(result.columns, pd.MultiIndex)
    # the shorter label is padded with an empty string
    assert ('LS1', 'Theta_0') in result.columns
    assert ('Incomplete', '') in result.columns


def test_convert_to_multiindex_leaves_single_level_flat():
    frame = pd.DataFrame([[1, 2]], columns=['Incomplete', 'Demand'])
    result = convert_to_MultiIndex(frame, axis=1)
    # no hyphens: a single-level (flat) index is left unchanged
    assert not isinstance(result.columns, pd.MultiIndex)
    assert list(result.columns) == ['Incomplete', 'Demand']


def test_convert_to_multiindex_invalid_axis_raises():
    frame = pd.DataFrame([[1]], columns=['a-b'])
    with pytest.raises(ValueError, match='Invalid axis'):
        convert_to_MultiIndex(frame, axis=2)


def test_convert_to_multiindex_axis0_splits_dashed_index():
    # axis=0 operates on the index; a dashed index becomes a MultiIndex.
    frame = pd.DataFrame({'x': [1, 2, 3]}, index=['A-1', 'A-2', 'B-1'])
    result = convert_to_MultiIndex(frame, axis=0)
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.nlevels == 2
    assert ('A', '1') in result.index
    assert ('B', '1') in result.index


# ---------------------------------------------------------------------------
# convert_dtypes
# ---------------------------------------------------------------------------


def test_convert_dtypes_numeric_strings_become_numeric():
    frame = pd.DataFrame({'a': ['1', '2', '3'], 'b': ['x', 'y', 'z']})
    result = convert_dtypes(frame)
    assert pd.api.types.is_numeric_dtype(result['a'])
    # non-numeric column is left untouched
    assert not pd.api.types.is_numeric_dtype(result['b'])
    assert list(result['a']) == [1, 2, 3]


def test_convert_dtypes_none_becomes_na():
    frame = pd.DataFrame({'a': [1.0, None, 3.0]})
    result = convert_dtypes(frame)
    assert pd.isna(result['a'].iloc[1])
    assert pd.api.types.is_numeric_dtype(result['a'])


# ---------------------------------------------------------------------------
# check_if_str_is_na / with_parsed_str_na_values
# ---------------------------------------------------------------------------


def test_check_if_str_is_na_tokens():
    for token in ('', 'N/A', 'NaN', 'null', 'None', 'NA', '<NA>'):
        assert check_if_str_is_na(token) is True
    assert check_if_str_is_na('hello') is False
    # non-string input is never N/A
    assert check_if_str_is_na(0) is False
    assert check_if_str_is_na(np.nan) is False


def test_with_parsed_str_na_values_maps_to_nan():
    frame = pd.DataFrame({'a': ['N/A', 'value', 'NaN'], 'b': ['null', 'x', 'NA']})
    result = with_parsed_str_na_values(frame)
    assert pd.isna(result['a'].iloc[0])
    assert result['a'].iloc[1] == 'value'
    assert pd.isna(result['a'].iloc[2])
    assert pd.isna(result['b'].iloc[0])
    assert result['b'].iloc[1] == 'x'
    assert pd.isna(result['b'].iloc[2])


# ---------------------------------------------------------------------------
# load_tabular against real library files
# ---------------------------------------------------------------------------


def test_load_tabular_fema_component_fragility():
    path = _data_path(
        'seismic',
        'building',
        'component',
        'FEMA P-58 2nd Edition',
        'fragility.csv',
    )
    frame = load_tabular(path)

    # columns are a MultiIndex with expected level-0 keys
    assert isinstance(frame.columns, pd.MultiIndex)
    level0 = set(frame.columns.get_level_values(0))
    assert 'Incomplete' in level0
    assert 'LS1' in level0
    assert 'Demand' in level0

    # the index holds the component IDs, not a default 0..N RangeIndex
    assert not isinstance(frame.index, pd.RangeIndex)
    assert 'B.10.31.001' in frame.index

    # a Theta_0 column is numeric
    assert ('LS1', 'Theta_0') in frame.columns
    assert pd.api.types.is_numeric_dtype(frame[('LS1', 'Theta_0')])


def test_load_tabular_portfolio_fragility_columns_sorted():
    path = _data_path(
        'seismic',
        'building',
        'portfolio',
        'Hazus v6.1',
        'fragility.csv',
    )
    frame = load_tabular(path)
    cols = list(frame.columns)
    assert list(cols) == sorted(cols)


def test_load_tabular_accepts_str_and_path():
    path = _data_path(
        'seismic',
        'building',
        'portfolio',
        'Hazus v6.1',
        'fragility.csv',
    )
    df_from_path = load_tabular(path)
    df_from_str = load_tabular(str(path))
    assert df_from_path.equals(df_from_str)
