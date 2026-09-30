"""
Canonical controlled vocabularies for the model parameters.

These enumerations are the authoritative definition of the controlled
vocabularies that appear in the packaged model parameters: the demand types a
fragility or loss model may be conditioned on, the unit type each demand is
measured in, and the statistical distribution families a model parameter may
use. They are owned here, in the data library, so a single source of truth
governs what the data may contain.

pelicun imports ``EDP_TYPES`` and ``UNIT_TYPES`` from this module instead of
defining them in ``pelicun.base``, so an entry added here, such as the bridge
component demands, becomes valid in pelicun without a pelicun change.
``DISTRIBUTION_FAMILIES`` mirrors the family names of pelicun's
random-variable registry (``rv_class_map`` in ``pelicun.uq``).
"""

from __future__ import annotations

#: The valid unit types a demand may be measured in. A demand type's unit
#: type identifies the physical quantity its values represent (so a consumer
#: can pick the appropriate unit for it); ``'unitless'`` marks dimensionless
#: demands, and ``'rotation'`` marks radian-measured demands that undergo no
#: unit conversion.
UNIT_TYPES: frozenset[str] = frozenset(
    {
        'acceleration',
        'speed',
        'displacement',
        'unitless',
        'rotation',
        'force',
        'force_per_length',
        'pressure',
    }
)

#: Mapping from a demand's full descriptive name to its properties: the short
#: demand-type code under ``'Acronym'`` (e.g. ``'Peak Floor Acceleration' ->
#: 'PFA'``) and the physical quantity it is measured in under ``'UnitType'``
#: (one of :data:`UNIT_TYPES`). The keys are the demand-type names a model's
#: ``Demand``/``Type`` field may use. Some demands carry an additional
#: qualifier, such as the period for spectral accelerations as ``|<period>``:
#: ``'Spectral Acceleration|1.0'``.
EDP_TYPES: dict[str, dict[str, str]] = {
    # Drifts
    'Story Drift Ratio': {'Acronym': 'PID', 'UnitType': 'unitless'},
    'Peak Interstory Drift Ratio': {'Acronym': 'PID', 'UnitType': 'unitless'},
    'Roof Drift Ratio': {'Acronym': 'PRD', 'UnitType': 'unitless'},
    'Peak Roof Drift Ratio': {'Acronym': 'PRD', 'UnitType': 'unitless'},
    'Damageable Wall Drift': {'Acronym': 'DWD', 'UnitType': 'unitless'},
    'Racking Drift Ratio': {'Acronym': 'RDR', 'UnitType': 'unitless'},
    'Mega Drift Ratio': {'Acronym': 'PMD', 'UnitType': 'unitless'},
    'Residual Drift Ratio': {'Acronym': 'RID', 'UnitType': 'unitless'},
    'Residual Interstory Drift Ratio': {'Acronym': 'RID', 'UnitType': 'unitless'},
    'Peak Effective Drift Ratio': {'Acronym': 'EDR', 'UnitType': 'unitless'},
    # Floor response
    'Peak Floor Acceleration': {'Acronym': 'PFA', 'UnitType': 'acceleration'},
    'Peak Floor Velocity': {'Acronym': 'PFV', 'UnitType': 'speed'},
    'Peak Floor Displacement': {'Acronym': 'PFD', 'UnitType': 'displacement'},
    # Component response
    'Peak Link Rotation Angle': {'Acronym': 'LR', 'UnitType': 'rotation'},
    'Peak Link Beam Chord Rotation': {'Acronym': 'LBR', 'UnitType': 'rotation'},
    # Bridge component response
    'Peak Column Curvature Ductility': {'Acronym': 'CCD', 'UnitType': 'unitless'},
    'Peak Column Drift Ratio': {'Acronym': 'CDR', 'UnitType': 'unitless'},
    'Peak Joint Opening': {'Acronym': 'JOP', 'UnitType': 'displacement'},
    'Peak Abutment Active Displacement': {
        'Acronym': 'ABA',
        'UnitType': 'displacement',
    },
    'Peak Abutment Passive Displacement': {
        'Acronym': 'ABP',
        'UnitType': 'displacement',
    },
    'Peak Abutment Transverse Displacement': {
        'Acronym': 'ABT',
        'UnitType': 'displacement',
    },
    'Peak Bearing Deformation': {'Acronym': 'BRD', 'UnitType': 'displacement'},
    'Peak Deck Displacement': {'Acronym': 'DEC', 'UnitType': 'displacement'},
    'Peak Foundation Translation': {'Acronym': 'FNT', 'UnitType': 'displacement'},
    'Peak Foundation Rotation': {'Acronym': 'FNR', 'UnitType': 'rotation'},
    'Peak Shear Key Deformation': {'Acronym': 'KEY', 'UnitType': 'displacement'},
    'Permanent Approach Settlement': {'Acronym': 'PAS', 'UnitType': 'displacement'},
    # Wind Intensity
    'Peak Gust Wind Speed': {'Acronym': 'PWS', 'UnitType': 'speed'},
    # Wind Demands
    'Peak Wind Force': {'Acronym': 'PWF', 'UnitType': 'force'},
    'Peak Internal Force': {'Acronym': 'PIF', 'UnitType': 'force'},
    'Peak Line Force': {'Acronym': 'PLF', 'UnitType': 'force_per_length'},
    'Peak Wind Pressure': {'Acronym': 'PWP', 'UnitType': 'pressure'},
    # Inundation Intensity
    'Peak Inundation Height': {'Acronym': 'PIH', 'UnitType': 'displacement'},
    # Shaking Intensity
    'Peak Ground Acceleration': {'Acronym': 'PGA', 'UnitType': 'acceleration'},
    'Peak Ground Velocity': {'Acronym': 'PGV', 'UnitType': 'speed'},
    'Spectral Acceleration': {'Acronym': 'SA', 'UnitType': 'acceleration'},
    'Spectral Velocity': {'Acronym': 'SV', 'UnitType': 'speed'},
    'Spectral Displacement': {'Acronym': 'SD', 'UnitType': 'displacement'},
    'Peak Spectral Acceleration': {'Acronym': 'SA', 'UnitType': 'acceleration'},
    'Peak Spectral Velocity': {'Acronym': 'SV', 'UnitType': 'speed'},
    'Peak Spectral Displacement': {'Acronym': 'SD', 'UnitType': 'displacement'},
    'Permanent Ground Deformation': {'Acronym': 'PGD', 'UnitType': 'displacement'},
    # Placeholder for advanced calculations
    'One': {'Acronym': 'ONE', 'UnitType': 'unitless'},
}

#: Mapping from a demand's full descriptive name to its short demand-type
#: code, derived from :data:`EDP_TYPES` (e.g.
#: ``'Peak Floor Acceleration' -> 'PFA'``).
EDP_to_demand_type: dict[str, str] = {  # pelicun-compatible name
    name: info['Acronym'] for name, info in EDP_TYPES.items()
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
