"""Tests for the Phase and PhaseDiagram classes."""

import pytest
from pymatgen.core import Element
from pymatgen.core import Structure as pmg_struct

# Import atlas first to resolve circular imports
import atlas  # noqa: F401
import atlas.core.phase_diagram as atl_pd
from atlas.core import exceptions as exc


class TestPhase:
    """Tests for the Phase class."""

    def _make_phase(self, **kwargs):
        """Helper to create a Phase with reasonable defaults."""
        defaults = {
            'name': 'test-phase',
            'element_list': ['Cu', 'Zn'],
            'composition': {
                'Cu': {'min': 0.5, 'max': 0.8},
                'Zn': {'min': 0.2, 'max': 0.5},
            },
            'prototype': 'mp-123',
            'offset': 0.0,
            'base_elem': 'Cu',
        }
        defaults.update(kwargs)
        return atl_pd.Phase(**defaults)

    def test_phase_creation(self):
        p = self._make_phase()
        assert p.name == 'test-phase'
        assert len(p.element_list) == 2
        assert p.prototype == 'mp-123'
        assert p.offset == 0.0

    def test_phase_slugifies_name(self):
        p = self._make_phase(name='My Phase Name!')
        assert p.name == 'my-phase-name'

    def test_phase_with_base_elem_in_list(self):
        """Phase accepts base_elem that is in element_list."""
        p = self._make_phase(base_elem='Cu')
        assert p.base_elem.symbol == 'Cu'

    def test_phase_with_base_elem_from_diagram(self):
        """Phase without base_elem gets it from phase_diagram."""
        pd_obj = atl_pd.BinaryPhaseDiagram(
            material='CuZn',
            element_list=['Cu', 'Zn'],
            base_elem='Cu',
        )
        p = atl_pd.Phase(
            name='alpha',
            element_list=['Cu', 'Zn'],
            composition={'Cu': {'min': 0, 'max': 1}, 'Zn': {'min': 0, 'max': 1}},
            prototype='mp-1',
            phase_diagram=pd_obj,
        )
        assert p.base_elem.symbol == 'Cu'

    def test_phase_base_elem_not_in_list_raises(self):
        """Phase raises MissingElementError if base_elem not in element_list."""
        with pytest.raises(exc.MissingElementError):
            self._make_phase(base_elem='Fe')

    def test_perc_in_phase_within_range(self):
        p = self._make_phase()
        # Cu base elem has min=0.5, max=0.8
        result = p.perc_in_phase(0.65)
        assert result is True

    def test_perc_in_phase_below_range(self):
        p = self._make_phase()
        result = p.perc_in_phase(0.3)
        assert result is False

    def test_perc_in_phase_above_range(self):
        p = self._make_phase()
        result = p.perc_in_phase(0.9)
        assert result is False

    def test_perc_in_phase_with_percentage_input(self):
        """perc_in_phase accepts values > 1 as percentages."""
        p = self._make_phase()
        # 65% = 0.65, should be within range
        result = p.perc_in_phase(65)
        assert result is True

    def test_get_base_elem_perc(self):
        """get_base_elem_perc returns the base element fraction from a structure."""
        p = self._make_phase()
        # Create a simple structure with known composition
        # CuZn structure: 50% Cu, 50% Zn
        lattice = [[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]]
        coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
        struct = pmg_struct(lattice, ['Cu', 'Zn'], coords)
        result = p.get_base_elem_perc(struct)
        assert abs(result - 0.5) < 0.01

    def test_phase_with_replace_dict(self):
        p = self._make_phase(replace_dict={'Cu': 'Zn'})
        assert p.replace_dict == {'Cu': 'Zn'}

    def test_phase_default_replace_dict(self):
        p = self._make_phase()
        assert p.replace_dict == {}

    def test_phase_with_offset(self):
        p = self._make_phase(offset=0.1)
        assert p.offset == 0.1

    def test_phase_str_representation(self):
        p = self._make_phase()
        s = str(p)
        assert 'test-phase' in s

    def test_phase_repr(self):
        p = self._make_phase()
        assert repr(p) == "phase: 'test-phase'"

    def test_phase_composition_validation(self):
        """Phase raises error if composition keys don't match element_list."""
        with pytest.raises(exc.CompositionNotMatchingElementListError):
            atl_pd.Phase(
                name='test',
                element_list=['Cu', 'Zn'],
                composition={'Cu': {'min': 0, 'max': 1}},  # Missing Zn
                prototype='mp-1',
                base_elem='Cu',
            )

    def test_phase_cluster_elem(self):
        p = self._make_phase(cluster_elem='Zn')
        assert p.cluster_elem.symbol == 'Zn'

    def test_phase_default_cluster_elem(self):
        p = self._make_phase()
        assert p.cluster_elem.symbol == 'Cu'  # defaults to base_elem


class TestSinglePhaseDiagram:
    """Tests for SinglePhaseDiagram."""

    def test_single_phase_diagram_creation(self):
        spd = atl_pd.SinglePhaseDiagram(
            material='Cu',
            element_list=['Cu'],
            base_elem='Cu',
        )
        assert spd.material == 'Cu'
        assert len(spd.element_list) == 1
        assert spd.base_elem.symbol == 'Cu'

    def test_single_phase_diagram_empty_phases(self):
        """New phase diagram has no phases until added."""
        spd = atl_pd.SinglePhaseDiagram(
            material='Cu',
            element_list=['Cu'],
            base_elem='Cu',
        )
        assert len(spd.phases) == 0
        assert spd.phase_names == []

    def test_single_phase_diagram_add_phase(self):
        spd = atl_pd.SinglePhaseDiagram(
            material='Cu',
            element_list=['Cu'],
            base_elem='Cu',
        )
        phase = atl_pd.Phase(
            name='alpha',
            element_list=['Cu'],
            composition={'Cu': {'min': 0, 'max': 1}},
            prototype='mp-1',
            base_elem='Cu',
        )
        spd.add_phase(phase)
        assert len(spd.phases) == 1
        assert 'alpha' in spd.phase_names

    def test_single_phase_diagram_get_phase(self):
        spd = atl_pd.SinglePhaseDiagram(
            material='Cu',
            element_list=['Cu'],
            base_elem='Cu',
        )
        phase = atl_pd.Phase(
            name='alpha',
            element_list=['Cu'],
            composition={'Cu': {'min': 0, 'max': 1}},
            prototype='mp-1',
            base_elem='Cu',
        )
        spd.add_phase(phase)
        retrieved = spd.get_phase('alpha')
        assert retrieved is not None
        assert retrieved.name == 'alpha'

    def test_single_phase_diagram_get_phase_by_object(self):
        spd = atl_pd.SinglePhaseDiagram(
            material='Cu',
            element_list=['Cu'],
            base_elem='Cu',
        )
        phase = atl_pd.Phase(
            name='alpha',
            element_list=['Cu'],
            composition={'Cu': {'min': 0, 'max': 1}},
            prototype='mp-1',
            base_elem='Cu',
        )
        spd.add_phase(phase)
        retrieved = spd.get_phase(phase)
        assert retrieved is phase

    def test_single_phase_diagram_alloy_set(self):
        spd = atl_pd.SinglePhaseDiagram(
            material='Cu',
            element_list=['Cu'],
            base_elem='Cu',
        )
        assert spd.alloy_set == {Element('Cu')}

    def test_single_phase_diagram_with_phases_in_init(self):
        phase = atl_pd.Phase(
            name='alpha',
            element_list=['Cu'],
            composition={'Cu': {'min': 0, 'max': 1}},
            prototype='mp-1',
            base_elem='Cu',
        )
        spd = atl_pd.SinglePhaseDiagram(
            'Cu',
            ['Cu'],
            'Cu',
            phase,
        )
        assert len(spd.phases) == 1


class TestBinaryPhaseDiagram:
    """Tests for BinaryPhaseDiagram."""

    def test_binary_phase_diagram_creation(self):
        bpd = atl_pd.BinaryPhaseDiagram(
            material='CuZn',
            element_list=['Cu', 'Zn'],
            base_elem='Cu',
        )
        assert bpd.material == 'CuZn'
        assert len(bpd.element_list) == 2
        assert bpd.base_elem.symbol == 'Cu'

    def test_binary_phase_diagram_add_phase(self):
        bpd = atl_pd.BinaryPhaseDiagram(
            material='CuZn',
            element_list=['Cu', 'Zn'],
            base_elem='Cu',
        )
        phase = atl_pd.Phase(
            name='alpha',
            element_list=['Cu', 'Zn'],
            composition={
                'Cu': {'min': 0.5, 'max': 0.8},
                'Zn': {'min': 0.2, 'max': 0.5},
            },
            prototype='mp-1',
            base_elem='Cu',
        )
        bpd.add_phase(phase)
        assert len(bpd.phases) == 1
        assert 'alpha' in bpd.phase_names

    def test_binary_phase_diagram_get_phase(self):
        bpd = atl_pd.BinaryPhaseDiagram(
            material='CuZn',
            element_list=['Cu', 'Zn'],
            base_elem='Cu',
        )
        phase = atl_pd.Phase(
            name='alpha',
            element_list=['Cu', 'Zn'],
            composition={
                'Cu': {'min': 0.5, 'max': 0.8},
                'Zn': {'min': 0.2, 'max': 0.5},
            },
            prototype='mp-1',
            base_elem='Cu',
        )
        bpd.add_phase(phase)
        retrieved = bpd.get_phase('alpha')
        assert retrieved is not None
        assert retrieved.name == 'alpha'

    def test_binary_phase_diagram_alloy_set(self):
        bpd = atl_pd.BinaryPhaseDiagram(
            material='CuZn',
            element_list=['Cu', 'Zn'],
            base_elem='Cu',
        )
        assert bpd.alloy_set == {Element('Cu'), Element('Zn')}

    def test_binary_phase_diagram_add_phase_wrong_elements_raises(self):
        """Adding phase with wrong element list raises error."""
        bpd = atl_pd.BinaryPhaseDiagram(
            material='CuZn',
            element_list=['Cu', 'Zn'],
            base_elem='Cu',
        )
        phase = atl_pd.Phase(
            name='beta',
            element_list=['Cu', 'Ni'],  # Wrong elements
            composition={'Cu': {'min': 0, 'max': 1}, 'Ni': {'min': 0, 'max': 1}},
            prototype='mp-1',
            base_elem='Cu',
        )
        with pytest.raises(exc.MissingElementError):
            bpd.add_phase(phase)

    def test_binary_phase_diagram_repr(self):
        bpd = atl_pd.BinaryPhaseDiagram(
            material='CuZn',
            element_list=['Cu', 'Zn'],
            base_elem='Cu',
        )
        r = repr(bpd)
        assert 'BinaryPhaseDiagram' in r
        assert 'CuZn' in r


class TestTernaryPhaseDiagram:
    """Tests for TernaryPhaseDiagram."""

    def test_ternary_phase_diagram_creation(self):
        tpd = atl_pd.TernaryPhaseDiagram(
            material='CuZnNi',
            element_list=['Cu', 'Zn', 'Ni'],
            base_elem='Cu',
        )
        assert tpd.material == 'CuZnNi'
        assert len(tpd.element_list) == 3
        assert tpd.base_elem.symbol == 'Cu'

    def test_ternary_phase_diagram_add_phase(self):
        tpd = atl_pd.TernaryPhaseDiagram(
            material='CuZnNi',
            element_list=['Cu', 'Zn', 'Ni'],
            base_elem='Cu',
        )
        phase = atl_pd.Phase(
            name='gamma',
            element_list=['Cu', 'Zn', 'Ni'],
            composition={
                'Cu': {'min': 0.3, 'max': 0.6},
                'Zn': {'min': 0.2, 'max': 0.4},
                'Ni': {'min': 0.1, 'max': 0.3},
            },
            prototype='mp-1',
            base_elem='Cu',
        )
        tpd.add_phase(phase)
        assert len(tpd.phases) == 1

    def test_ternary_phase_diagram_alloy_set(self):
        tpd = atl_pd.TernaryPhaseDiagram(
            material='CuZnNi',
            element_list=['Cu', 'Zn', 'Ni'],
            base_elem='Cu',
        )
        assert tpd.alloy_set == {Element('Cu'), Element('Zn'), Element('Ni')}


class TestPhaseDiagramFactory:
    """Tests for the PhaseDiagram factory function."""

    def test_single_element(self):
        pd_obj = atl_pd.PhaseDiagram(
            material='Cu',
            element_list=['Cu'],
            base_elem='Cu',
        )
        assert isinstance(pd_obj, atl_pd.SinglePhaseDiagram)

    def test_two_elements(self):
        pd_obj = atl_pd.PhaseDiagram(
            material='CuZn',
            element_list=['Cu', 'Zn'],
            base_elem='Cu',
        )
        assert isinstance(pd_obj, atl_pd.BinaryPhaseDiagram)

    def test_three_elements(self):
        pd_obj = atl_pd.PhaseDiagram(
            material='CuZnNi',
            element_list=['Cu', 'Zn', 'Ni'],
            base_elem='Cu',
        )
        assert isinstance(pd_obj, atl_pd.TernaryPhaseDiagram)

    def test_quaternary_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match='binary or ternary'):
            atl_pd.PhaseDiagram(
                material='CuZnNiFe',
                element_list=['Cu', 'Zn', 'Ni', 'Fe'],
                base_elem='Cu',
            )


class TestBasePhaseDiagram:
    """Tests for BasePhaseDiagram functionality."""

    def test_phase_dict_updated_on_add(self):
        bpd = atl_pd.BinaryPhaseDiagram(
            material='CuZn',
            element_list=['Cu', 'Zn'],
            base_elem='Cu',
        )
        phase = atl_pd.Phase(
            name='alpha',
            element_list=['Cu', 'Zn'],
            composition={
                'Cu': {'min': 0.5, 'max': 0.8},
                'Zn': {'min': 0.2, 'max': 0.5},
            },
            prototype='mp-1',
            base_elem='Cu',
        )
        bpd.add_phase(phase)
        assert 'alpha' in bpd.phase_dict
        assert bpd.phase_dict['alpha'] is phase

    def test_get_phase_nonexistent(self):
        bpd = atl_pd.BinaryPhaseDiagram(
            material='CuZn',
            element_list=['Cu', 'Zn'],
            base_elem='Cu',
        )
        result = bpd.get_phase('nonexistent')
        assert result is None

    def test_get_phase_invalid_type_raises(self):
        bpd = atl_pd.BinaryPhaseDiagram(
            material='CuZn',
            element_list=['Cu', 'Zn'],
            base_elem='Cu',
        )
        with pytest.raises(TypeError, match='not a Phase-like object'):
            bpd.get_phase(123)

    def test_phase_diagram_phase_assigned_on_add(self):
        """Phase gets phase_diagram reference when added."""
        bpd = atl_pd.BinaryPhaseDiagram(
            material='CuZn',
            element_list=['Cu', 'Zn'],
            base_elem='Cu',
        )
        phase = atl_pd.Phase(
            name='alpha',
            element_list=['Cu', 'Zn'],
            composition={
                'Cu': {'min': 0.5, 'max': 0.8},
                'Zn': {'min': 0.2, 'max': 0.5},
            },
            prototype='mp-1',
            base_elem='Cu',
        )
        assert phase.phase_diagram is None
        bpd.add_phase(phase)
        assert phase.phase_diagram is bpd

    def test_base_elem_element_object(self):
        """base_elem is stored as pymatgen Element object."""
        bpd = atl_pd.BinaryPhaseDiagram(
            material='CuZn',
            element_list=['Cu', 'Zn'],
            base_elem='Cu',
        )
        assert hasattr(bpd.base_elem, 'symbol')
        assert bpd.base_elem.symbol == 'Cu'

    def test_element_list_element_objects(self):
        """element_list contains Element objects, not strings."""
        bpd = atl_pd.BinaryPhaseDiagram(
            material='CuZn',
            element_list=['Cu', 'Zn'],
            base_elem='Cu',
        )
        for ele in bpd.element_list:
            assert hasattr(ele, 'symbol')
