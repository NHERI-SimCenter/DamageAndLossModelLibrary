"""Generates SimCenter Hurricane Wind Component Library files."""

import json

import pandas as pd


def create_fragility_files():
    """Create the standard CSV and JSON files for the library."""
    # open the raw input data file
    input_data = pd.read_csv('fragility_source.csv')

    # start by creating the fragility CSV
    fragility_cols = [
        'ID',
        'Incomplete',
        'Demand-Type',
        'Demand-Unit',
        'Demand-Offset',
        'Demand-Directional',
        'LS1-Family',
        'LS1-Theta_0',
        'LS1-Theta_1',
    ]

    fragility_data = (
        input_data[fragility_cols].set_index('ID').sort_index().convert_dtypes()
    )

    fragility_data.to_csv('fragility.csv')

    # now open the metadata header file
    with open('fragility_header.json') as f:  # noqa: PTH123
        meta = json.load(f)

    # create the component-specific metadata
    meta_plus = {}
    for __, row in input_data.iterrows():
        component_meta = {
            'Description': row['Meta-Description'],
            'Comments': row.get('Meta-Comments'),
            'SuggestedComponentBlockSize': row['Meta-SuggestedComponentBlockSize'],
            'RoudUpToIntegerQuantity': row['Meta-RoundUpToIntegerQuantity'],
            'Reference': [ref.strip() for ref in row['Meta-Reference'].split(',')],
            'LimitStates': {
                'LS1': {
                    'DS1': {
                        'Description': row['Meta-LimitStates-LS1-DS1-Description']
                    }
                }
            },
        }

        if pd.isna(row['Meta-Comments']):
            del component_meta['Comments']

        # make sure the references are included in the header
        for ref in component_meta['Reference']:
            if ref not in meta['References']:
                print(f'Error, missing reference: {ref}')  # noqa: T201

        meta_plus[row['ID']] = component_meta

    meta.update(meta_plus)

    with open('fragility.json', 'w') as f:  # noqa: PTH123
        json.dump(meta, f, indent=2)


def main():
    """Generate SimCenter Hurricane Wind Component Library Files."""
    create_fragility_files()


if __name__ == '__main__':
    main()
