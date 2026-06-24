"""
Canonical controlled vocabularies for the model parameters.

These enumerations are the authoritative definition of the controlled
vocabularies that appear in the packaged model parameters: the demand types a
fragility or loss model may be conditioned on, and the statistical
distribution families a model parameter may use. They are owned here, in the
data library, so a single source of truth governs what the data may contain.

The values mirror pelicun's current definitions -- ``EDP_to_demand_type`` in
``pelicun.base`` and the ``rv_class_map`` registry in ``pelicun.uq`` -- and
are intended to be imported from here by pelicun rather than redefined there.
"""

from __future__ import annotations

#: Mapping from a demand's full descriptive name to its short demand-type
#: code (e.g. ``'Peak Floor Acceleration' -> 'PFA'``). The keys are the
#: demand-type names a model's ``Demand``/``Type`` field may use; a spectral
#: name may additionally carry a ``|<period>`` qualifier in the data, such as
#: ``'Spectral Acceleration|1.0'``.
EDP_to_demand_type: dict[str, str] = {  # pelicun-compatible name
    # Drifts
    'Story Drift Ratio': 'PID',
    'Peak Interstory Drift Ratio': 'PID',
    'Roof Drift Ratio': 'PRD',
    'Peak Roof Drift Ratio': 'PRD',
    'Damageable Wall Drift': 'DWD',
    'Racking Drift Ratio': 'RDR',
    'Mega Drift Ratio': 'PMD',
    'Residual Drift Ratio': 'RID',
    'Residual Interstory Drift Ratio': 'RID',
    'Peak Effective Drift Ratio': 'EDR',
    # Floor response
    'Peak Floor Acceleration': 'PFA',
    'Peak Floor Velocity': 'PFV',
    'Peak Floor Displacement': 'PFD',
    # Component response
    'Peak Link Rotation Angle': 'LR',
    'Peak Link Beam Chord Rotation': 'LBR',
    # Wind Intensity
    'Peak Gust Wind Speed': 'PWS',
    # Wind Demands
    'Peak Wind Force': 'PWF',
    'Peak Internal Force': 'PIF',
    'Peak Line Force': 'PLF',
    'Peak Wind Pressure': 'PWP',
    # Inundation Intensity
    'Peak Inundation Height': 'PIH',
    # Shaking Intensity
    'Peak Ground Acceleration': 'PGA',
    'Peak Ground Velocity': 'PGV',
    'Spectral Acceleration': 'SA',
    'Spectral Velocity': 'SV',
    'Spectral Displacement': 'SD',
    'Peak Spectral Acceleration': 'SA',
    'Peak Spectral Velocity': 'SV',
    'Peak Spectral Displacement': 'SD',
    'Permanent Ground Deformation': 'PGD',
    # Placeholder for advanced calculations
    'One': 'ONE',
}

#: The valid demand-type names (the keys of :data:`EDP_to_demand_type`). A
#: data value may append a ``|<period>`` qualifier to one of these names.
DEMAND_TYPES: frozenset[str] = frozenset(EDP_to_demand_type)

#: The statistical distribution families a model parameter may declare, from
#: pelicun's random-variable registry (``pelicun.uq.rv_class_map``). An absent
#: or empty family is treated as ``'deterministic'``.
DISTRIBUTION_FAMILIES: frozenset[str] = frozenset(
    {
        'normal',
        'normal_std',
        'normal_cov',
        'lognormal',
        'uniform',
        'weibull',
        'multilinear_CDF',
        'empirical',
        'multinomial',
        'coupled_empirical',
        'deterministic',
    }
)
