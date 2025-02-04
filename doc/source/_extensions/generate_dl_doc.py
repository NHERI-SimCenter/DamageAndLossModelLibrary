"""Generate the DLML documentation pages."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from textwrap import dedent
from zipfile import ZipFile

import numpy as np
from tqdm import tqdm
from visuals import plot_fragility, plot_repair

os.chdir('../')


def generate_md5(file_path):
    """
    Generate an MD5 hash of a file.

    Parameters
    ----------
    file_path : str
        The path to the file for which to generate the MD5 hash.

    Returns
    -------
    str
        The MD5 hash of the file.
    """
    md5 = hashlib.md5()  # noqa: S324
    with Path(file_path).open('rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5.update(chunk)
    return md5.hexdigest()


def combine_md5_hashes(md5_list):
    """
    Combine a list of MD5 hashes and generate a new MD5 hash.

    Parameters
    ----------
    md5_list : list of str
        A list of MD5 hashes.

    Returns
    -------
    str
        A new MD5 hash based on the combination of the given hashes.
    """
    combined_md5 = hashlib.md5()  # noqa: S324
    for md5_hash in md5_list:
        combined_md5.update(md5_hash.encode('utf-8'))
    return combined_md5.hexdigest()


def get_dlml_tag(dlml):
    """Get the damage and loss model tag."""
    return '-'.join(str(dlml.parent).split('/')).replace(' ', '_')


def create_component_group_directory(cmp_groups, root, dlml_tag):
    """Create a component group directory."""
    member_ids = []

    if isinstance(cmp_groups, dict):
        for grp_name, grp_members in cmp_groups.items():
            grp_id = f"{grp_name.split('-')[0].strip()}"

            # create the first-level dirs
            grp_dir = Path(root) / grp_id
            grp_dir.mkdir(parents=True, exist_ok=True)

            # call this function again to add subdirs
            subgrp_ids = create_component_group_directory(
                grp_members, grp_dir, dlml_tag=dlml_tag
            )

            grp_index_contents = dedent(
                f"""

            {"*" * len(grp_name)}
            {grp_name}
            {"*" * len(grp_name)}

            The following models are available:

            .. toctree::
               :maxdepth: 1

            """
            )

            for member_id in subgrp_ids:
                grp_index_contents += f'   {member_id}/index\n'

            grp_index_path = grp_dir / 'index.rst'
            with grp_index_path.open('w', encoding='utf-8') as f:
                f.write(grp_index_contents)

            member_ids.append(grp_id)

    else:
        for grp_name in cmp_groups:
            grp_id = f"{grp_name.split('-')[0].strip()}"

            grp_dir = Path(root) / grp_id

            # create group dirs
            grp_dir.mkdir(parents=True, exist_ok=True)

            grp_index_contents = dedent(
                f"""

            {"*" * len(grp_name)}
            {grp_name}
            {"*" * len(grp_name)}

            The following models are available:

            """
            )

            grp_index_path = grp_dir / 'index.rst'
            with grp_index_path.open('w', encoding='utf-8') as f:
                f.write(grp_index_contents)

            member_ids.append(grp_id)

    return member_ids


def generate_damage_docs(doc_folder: Path, cache_folder: Path):  # noqa: C901
    """Generate damage parameter documentation."""
    doc_folder = doc_folder / 'damage'

    damage_dlmls = []

    for hazard_name in ['seismic', 'hurricane', 'flood']:
        resource_folder = Path(f'./{hazard_name}')

        # get all the available damage dlmls
        damage_dlmls.extend(list(resource_folder.rglob('fragility.csv')))

    # create the main index file
    damage_index_contents = dedent(
        """\

    *************
    Damage Models
    *************

    The following collections are available in our Damage and Loss Model Library:

    .. toctree::
       :maxdepth: 1

    """
    )

    # for each database
    for dlml in (pbar := tqdm(damage_dlmls)):
        pbar.set_postfix({'File': f'{str(dlml)[:80]:<80}'})

        # blacklist
        if ignore_file(dlml):
            continue

        # add dlml to main damage index file
        damage_index_contents += f'   {dlml.parent}/index\n'

        # create a folder
        (doc_folder / dlml.parent).mkdir(parents=True, exist_ok=True)

        zip_hash = generate_md5(dlml)
        zip_filepath = ((cache_folder) / zip_hash).with_suffix('.zip')

        # if it doesn't exist in the cache, create it.
        # otherwise it exists, obviously.
        if not zip_filepath.is_file():
            plot_fragility(
                str(dlml),
                str(zip_filepath),
                create_zip='1',
            )

        # check if there are metadata available
        dlml_json = dlml.with_suffix('.json')
        if dlml_json.is_file():
            with dlml_json.open('r', encoding='utf-8') as f:
                dlml_meta = json.load(f)
        else:
            dlml_meta = None

        if dlml_meta is not None:
            dlml_general = dlml_meta.get('_GeneralInformation', {})

            # create the top of the dlml index file
            dlml_short_name = dlml_general.get('ShortName', dlml)

            dlml_description = dlml_general.get(
                'Description', f'The following models are available in {dlml}:'
            )

            dlml_index_contents = dedent(
                f"""

            {"*" * len(dlml_short_name)}
            {dlml_short_name}
            {"*" * len(dlml_short_name)}

            {dlml_description}

            """
            )

            # check if there are component groups defined
            dlml_cmp_groups = dlml_general.get('ComponentGroups', None)

            # if yes, create the corresponding directory structure and index files
            if dlml_cmp_groups is not None:
                dlml_index_contents += dedent(
                    """
                .. toctree::
                   :maxdepth: 1

                """
                )

                # create the directory structure and index files
                dlml_tag = '-'.join(str(dlml.parent).split('/')).replace(' ', '_')
                grp_ids = create_component_group_directory(
                    dlml_cmp_groups,
                    root=(doc_folder / dlml.parent),
                    dlml_tag=dlml_tag,
                )

                for member_id in grp_ids:
                    dlml_index_contents += f'   {member_id}/index\n'

        else:
            print(f'No metadata available for {dlml}')  # noqa: T201

            # create the top of the dlml index file
            dlml_index_contents = dedent(
                f"""\

            {"*" * len(dlml)}
            {dlml}
            {"*" * len(dlml)}

            The following models are available in {dlml}:

            """
            )

        dlml_index_path = doc_folder / dlml.parent / 'index.rst'
        with dlml_index_path.open('w', encoding='utf-8') as f:
            f.write(dlml_index_contents)

        # now open the zip file
        with ZipFile(zip_filepath, 'r') as zipObj:  # noqa: N806
            # for each component
            for comp in sorted(zipObj.namelist()):
                if comp == 'fragility':
                    continue
                comp = Path(comp).stem.removesuffix('.html')  # noqa: PLW2901

                # check where the component belongs
                comp_labels = comp.split('.')
                comp_path = doc_folder / dlml.parent
                new_path = deepcopy(comp_path)

                c_i = 0
                while new_path.is_dir():
                    comp_path = new_path

                    if c_i > len(comp_labels):
                        break

                    new_path = comp_path / f"{'.'.join(comp_labels[:c_i])}"

                    c_i += 1

                grp_index_path = comp_path / 'index.rst'

                comp_meta = None
                if dlml_meta is not None:
                    comp_meta = dlml_meta.get(comp, None)

                with grp_index_path.open('a', encoding='utf-8') as f:
                    # add the component info to the docs

                    if comp_meta is None:
                        comp_contents = dedent(
                            f"""
                        {comp}
                        {"*" * len(comp)}

                        .. raw:: html
                           :file: {comp}.html


                        .. raw:: html

                           <hr>

                        """
                        )

                    else:
                        comp_contents = dedent(
                            f"""
                        .. raw:: html

                           <p class="dl_comp_name"><b>{comp}</b> | {comp_meta.get("Description", "")}</p>
                           <div>

                        """
                        )

                        comp_comments = comp_meta.get('Comments', '').split('\n')

                        for comment_line in comp_comments:
                            if comment_line != '':
                                comp_contents += f'| {comment_line}\n'

                        if 'SuggestedComponentBlockSize' in comp_meta:
                            roundup = comp_meta.get(
                                'RoundUpToIntegerQuantity', 'False'
                            )
                            if roundup == 'True':
                                roundup_text = '(round up to integer quantity)'
                            else:
                                roundup_text = ''

                            comp_contents += dedent(
                                f"""

                            Suggested Block Size: {comp_meta['SuggestedComponentBlockSize']} {roundup_text}

                            """
                            )

                        comp_contents += dedent(
                            f"""

                        .. raw:: html
                           :file: {comp}.html

                        """
                        )

                        if 'Reference' in comp_meta:
                            comp_refs = [
                                dlml_meta['References'][ref]
                                for ref in comp_meta['Reference']
                            ]
                            comp_refs_str = '|\n'

                            for ref in comp_refs:
                                comp_refs_str += f'| {ref}\n'

                            comp_contents += comp_refs_str

                        comp_contents += dedent(
                            """

                        .. raw:: html

                           <hr>
                        """
                        )

                    f.write(comp_contents)

                # copy the file from the zip to the dlml folder
                zipObj.extract(f'{comp}.html', path=comp_path)

    damage_index_path = doc_folder / 'index.rst'
    with damage_index_path.open('w', encoding='utf-8') as f:
        f.write(damage_index_contents)


def generate_repair_docs(doc_folder: Path, cache_folder: Path):  # noqa: C901
    """Generate repair parameter documentation."""
    resource_folder = Path()

    doc_folder = doc_folder / 'repair'

    repair_dlmls = []

    for hazard_name in ['seismic', 'hurricane', 'flood']:
        resource_folder = Path(f'./{hazard_name}')

        # get all the available consequence repair dlmls
        repair_dlmls.extend(list(resource_folder.rglob('consequence_repair.csv')))

    # create the main index file
    repair_index_contents = dedent(
        """\

    *************************
    Repair Consequence Models
    *************************

    The following collections are available in our Damage and Loss Model Library:

    .. toctree::
       :maxdepth: 1

    """
    )

    # for each database
    for dlml in (pbar := tqdm(repair_dlmls)):
        pbar.set_postfix({'File': f'{str(dlml)[:80]:<80}'})

        # blacklist
        if ignore_file(dlml):
            continue

        # add dlml to main repair index file
        repair_index_contents += f'   {dlml.parent}/index\n'

        # create a folder
        (doc_folder / dlml.parent).mkdir(parents=True, exist_ok=True)

        zip_hash = generate_md5(dlml)
        zip_filepath = ((cache_folder) / zip_hash).with_suffix('.zip')

        # if it doesn't exist in the cache, create it.
        # otherwise it exists, obviously.
        if not zip_filepath.is_file():
            plot_repair(
                str(dlml),
                str(zip_filepath),
                create_zip='1',
            )

        # check if there is metadata available
        dlml_json = dlml.with_suffix('.json')
        if dlml_json.is_file():
            with dlml_json.open('r', encoding='utf-8') as f:
                dlml_meta = json.load(f)
        else:
            dlml_meta = None

        if dlml_meta is not None:
            dlml_general = dlml_meta.get('_GeneralInformation', {})

            # create the top of the dlml index file
            dlml_short_name = dlml_general.get('ShortName', dlml)

            dlml_description = dlml_general.get(
                'Description', f'The following models are available in {dlml}:'
            )

            dlml_index_contents = dedent(
                f"""

            {"*" * len(dlml_short_name)}
            {dlml_short_name}
            {"*" * len(dlml_short_name)}

            {dlml_description}

            """
            )

            # check if there are component groups defined
            dlml_cmp_groups = dlml_general.get('ComponentGroups', None)

            # if yes, create the corresponding directory structure and index files
            if dlml_cmp_groups is not None:
                dlml_index_contents += dedent(
                    """
                .. toctree::
                   :maxdepth: 1

                """
                )

                # create the directory structure and index files
                dlml_tag = get_dlml_tag(dlml)
                grp_ids = create_component_group_directory(
                    dlml_cmp_groups,
                    root=(doc_folder / dlml.parent),
                    dlml_tag=dlml_tag,
                )

                for member_id in grp_ids:
                    dlml_index_contents += f'   {member_id}/index\n'

        else:
            print(f'No metadata available for {dlml}')  # noqa: T201

            # create the top of the dlml index file
            dlml_index_contents = dedent(
                f"""\

            {"*" * len(dlml)}
            {dlml}
            {"*" * len(dlml)}

            The following models are available in {dlml}:

            """
            )

        dlml_index_path = doc_folder / dlml.parent / 'index.rst'
        with dlml_index_path.open('w', encoding='utf-8') as f:
            f.write(dlml_index_contents)

        # now open the zip file
        with ZipFile(zip_filepath, 'r') as zipObj:  # noqa: N806
            html_files = [
                Path(filepath).stem for filepath in sorted(zipObj.namelist())
            ]

            comp_ids = np.unique([c_id.split('-')[0] for c_id in html_files])

            dv_types = np.unique([c_id.split('-')[1] for c_id in html_files])

            # for each component
            for comp in comp_ids:
                comp_files = []
                for dv_i in dv_types:
                    filename = f'{comp}-{dv_i}'
                    if filename in html_files:
                        comp_files.append(filename)

                # check where the component belongs
                comp_labels = comp.split('.')
                comp_path = doc_folder / dlml.parent
                new_path = deepcopy(comp_path)

                c_i = 0
                while new_path.is_dir():
                    comp_path = new_path

                    if c_i > len(comp_labels):
                        break

                    new_path = comp_path / f"{'.'.join(comp_labels[:c_i])}"

                    c_i += 1

                grp_index_path = comp_path / 'index.rst'

                comp_meta = None
                if dlml_meta is not None:
                    comp_meta = dlml_meta.get(comp, None)

                with grp_index_path.open('a', encoding='utf-8') as f:
                    # add the component info to the docs

                    if comp_meta is None:
                        comp_contents = dedent(
                            f"""
                        {comp}
                        {"*" * len(comp)}

                        """
                        )

                    else:
                        comp_contents = dedent(
                            f"""
                        .. raw:: html

                           <p class="dl_comp_name"><b>{comp}</b> | {comp_meta.get("Description", "")}</p>
                           <div>

                        """
                        )

                        comp_comments = comp_meta.get('Comments', '').split('\n')

                        for comment_line in comp_comments:
                            if comment_line != '':
                                comp_contents += f'| {comment_line}\n'

                        if 'SuggestedComponentBlockSize' in comp_meta:
                            roundup = comp_meta.get(
                                'RoundUpToIntegerQuantity', 'False'
                            )
                            if roundup == 'True':
                                roundup_text = '(round up to integer quantity)'
                            else:
                                roundup_text = ''

                            comp_contents += dedent(
                                f"""

                            Suggested Block Size: {comp_meta['SuggestedComponentBlockSize']} {roundup_text}

                            """
                            )

                    comp_contents += dedent(
                        """

                    The following repair consequences are available for this model:

                    """
                    )

                    for comp_file in comp_files:
                        dv_type = comp_file.split('-')[1]

                        comp_contents += dedent(
                            f"""

                        **{dv_type}**

                        .. raw:: html
                           :file: {comp_file}.html

                        """
                        )

                        # copy the file from the zip to the dlml folder
                        zipObj.extract(f'{comp_file}.html', path=comp_path)

                    comp_contents += dedent(
                        """

                    .. raw:: html

                       <hr>

                    """
                    )

                    f.write(comp_contents)

    repair_index_path = doc_folder / 'index.rst'
    with repair_index_path.open('w', encoding='utf-8') as f:
        f.write(repair_index_contents)


def ignore_file(dlml):
    """Ignore certain paths due to lack of support. To remove."""
    return str(dlml.parent) in {
        'hurricane/building/portfolio/Hazus v5.1 original',
        'hurricane/building/portfolio/Hazus v5.1 coupled',
        'seismic/water_network/portfolio/Hazus v6.1',
        'seismic/building/subassembly/Hazus v5.1',
        'flood/building/portfolio/Hazus v6.1',
    }


def main():
    """Run the code."""
    cache_folder = Path('doc/cache')

    doc_folder = Path('doc/source/dl_doc')
    if Path(doc_folder).exists():
        shutil.rmtree(doc_folder)
    doc_folder.mkdir(parents=True, exist_ok=True)

    generate_damage_docs(doc_folder, cache_folder)
    generate_repair_docs(doc_folder, cache_folder)


if __name__ == '__main__':
    main()
