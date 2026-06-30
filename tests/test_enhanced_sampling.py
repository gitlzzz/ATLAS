"""Tests for CV-free enhanced-sampling drivers (using the cheap EMT calculator)."""

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from atlas.active_learning.md.enhanced_sampling import (
    BoostCalculator,
    make_enhanced_calculator,
    replica_temperature_ladder,
    run_minima_hopping,
    run_replica_exchange_md,
)


def _cu():
    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
    atoms.calc = EMT()
    return atoms


class TestBoostCalculatorGaMD:
    """GaMD boost contract and force scaling."""

    def test_no_boost_above_threshold(self):
        # E below the true energy -> V >= E -> no boost, energy/forces unchanged.
        atoms = _cu()
        v_true = atoms.get_potential_energy()
        f_true = atoms.get_forces()
        atoms.calc = BoostCalculator(
            EMT(), method='gamd', boost_energy=v_true - 10.0, boost_k=0.1
        )
        assert np.isclose(atoms.get_potential_energy(), v_true)
        assert np.allclose(atoms.get_forces(), f_true)

    def test_boost_below_threshold_raises_energy_scales_forces(self):
        atoms = _cu()
        base = EMT()
        probe = atoms.copy()
        probe.calc = base
        v_true = probe.get_potential_energy()
        f_true = probe.get_forces()

        e = v_true + 5.0  # threshold above true energy -> boost active
        k = 0.02
        atoms.calc = BoostCalculator(EMT(), method='gamd', boost_energy=e, boost_k=k)

        u = e - v_true
        expected_dv = 0.5 * k * u * u
        expected_scale = max(1.0 - k * u, 0.0)

        assert np.isclose(atoms.get_potential_energy(), v_true + expected_dv)
        assert np.allclose(atoms.get_forces(), expected_scale * f_true)

    def test_auto_tuning_sets_params_after_equilibration(self):
        atoms = _cu()
        calc = BoostCalculator(EMT(), method='gamd', equilibration_steps=1)
        assert calc._params_ready is False
        atoms.calc = calc
        atoms.get_potential_energy()  # one call -> finalize
        assert calc._params_ready is True
        assert calc.boost_energy is not None
        assert calc.boost_k is not None


class TestBoostCalculatorAMD:
    def test_amd_force_factor(self):
        atoms = _cu()
        probe = atoms.copy()
        probe.calc = EMT()
        v_true = probe.get_potential_energy()
        f_true = probe.get_forces()

        e = v_true + 5.0
        alpha = 2.0
        atoms.calc = BoostCalculator(
            EMT(), method='amd', boost_energy=e, amd_alpha=alpha
        )
        u = e - v_true
        expected_dv = u * u / (alpha + u)
        expected_scale = max(1.0 - u * (2 * alpha + u) / (alpha + u) ** 2, 0.0)
        assert np.isclose(atoms.get_potential_energy(), v_true + expected_dv)
        assert np.allclose(atoms.get_forces(), expected_scale * f_true)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            BoostCalculator(EMT(), method='nonsense')


class TestMakeEnhancedCalculator:
    def test_none_returns_base(self):
        base = EMT()
        assert make_enhanced_calculator(base, {'method': 'none'}) is base
        assert make_enhanced_calculator(base, None) is base

    def test_gamd_wraps(self):
        wrapped = make_enhanced_calculator(EMT(), {'method': 'gamd'})
        assert isinstance(wrapped, BoostCalculator)
        assert wrapped.method == 'gamd'


class TestReplicaTemperatureLadder:
    def test_single_replica(self):
        assert replica_temperature_ladder(300.0, 1) == [300.0]

    def test_geometric_ladder(self):
        ladder = replica_temperature_ladder(300.0, 4, max_multiplier=2.0)
        assert len(ladder) == 4
        assert np.isclose(ladder[0], 300.0)
        assert np.isclose(ladder[-1], 600.0)
        # Monotonically increasing.
        assert all(b > a for a, b in zip(ladder, ladder[1:]))


class TestReplicaExchange:
    def test_produces_frames_with_refs(self):
        atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
        frames = run_replica_exchange_md(
            atoms=atoms,
            calculator_factory=EMT,
            temperatures=[300.0, 450.0],
            n_steps=20,
            swap_interval=10,
            timestep_fs=1.0,
            friction_ps=50.0,
            rng=np.random.default_rng(0),
        )
        # 2 replicas * (20/10) blocks = 4 frames.
        assert len(frames) == 4
        for frame in frames:
            assert 'REF_energy' in frame.info
            assert frame.arrays['REF_forces'].shape == (len(atoms), 3)


class TestMinimaHopping:
    def test_returns_minima(self):
        # Small distorted cluster; a couple of hops with EMT.
        atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
        minima = run_minima_hopping(
            atoms, EMT(), total_steps=2, temperature=500.0
        )
        assert isinstance(minima, list)
        assert len(minima) >= 1
