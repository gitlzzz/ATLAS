"""Module for testing the kpoint generation for DFT calculations."""

import itertools as it

import numpy as np
import pytest
from aiida import load_profile
from pymatgen.core import Lattice, Structure
from pymatgen.core.surface import SlabGenerator

import MatDBForge.core.surfaces as mdb_surf
from MatDBForge.workflows import aiida_utils as mdb_aut


@pytest.mark.xfail
def test_both_kpoint_style_given():
    _ = mdb_aut.generate_kpoints_data(
        structure=...,
        calc_type=...,
        kspacing=0.1,
        kspacing_vec=[1, 1, 1],
    )


def test_kpoints_bulk(): ...


@pytest.mark.xfail
def test_kpoints_surface():
    # Loading aiida profile
    load_profile('')

    # Defining calculation type
    calc_type = mdb_aut.CalcType.single_point_surface

    # Defining kspacing
    kspacing = 0.133203528512207

    # Generating sample structure
    bcc_cu = Structure.from_spacegroup(
        'Pm-3m', Lattice.cubic(3.6258), ['Cu'], [[0, 0, 0]]
    )

    for miller in list(it.product(list(range(3 + 1)), repeat=3))[1:]:
        slabgen = SlabGenerator(
            bcc_cu,
            miller_index=miller,
            min_slab_size=8,
            min_vacuum_size=15,
            center_slab=True,
        )

        slabs = slabgen.get_slabs()

        for slab in slabs:
            slab = mdb_surf.slab_to_bottom(slab)

            kspacing_vec = mdb_aut.kpoint_mesh_from_density(slab, kspacing=kspacing)

            if np.all(np.equal(kspacing_vec, 1)):
                kspacing_vec = mdb_aut.kpoint_mesh_from_density(
                    slab, 0.0994491889005363
                )

            kpoints_data = mdb_aut.generate_kpoints_data(
                structure=slab,
                calc_type=calc_type,
                kspacing=kspacing,
            )
            assert np.any(np.greater(kpoints_data.get_kpoints_mesh()[0], [1, 1, 1]))


def test_kpoints_cluster(): ...
