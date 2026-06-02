"""Tests for the custom exceptions module."""

import pytest

from atlas.core import exceptions as exc


class TestBaseStructureNotFound:
    """Tests for BaseStructureNotFound exception."""

    def test_can_be_raised(self):
        with pytest.raises(exc.BaseStructureNotFound):
            raise exc.BaseStructureNotFound

    def test_is_exception(self):
        assert issubclass(exc.BaseStructureNotFound, Exception)


class TestEmptyDataBase:
    """Tests for EmptyDataBase exception."""

    def test_str_representation(self):
        e = exc.EmptyDataBase()
        assert str(e) == 'A database could not be read from the given path.'

    def test_can_be_raised(self):
        with pytest.raises(exc.EmptyDataBase):
            raise exc.EmptyDataBase


class TestIncompatibleDataBase:
    """Tests for IncompatibleDataBase exception."""

    def test_str_representation(self):
        e = exc.IncompatibleDataBase()
        assert str(e) == "The format of the structure can't be understood."

    def test_can_be_raised(self):
        with pytest.raises(exc.IncompatibleDataBase):
            raise exc.IncompatibleDataBase


class TestFilterError:
    """Tests for FilterError exception."""

    def test_can_be_raised(self):
        with pytest.raises(exc.FilterError):
            raise exc.FilterError

    def test_is_exception(self):
        assert issubclass(exc.FilterError, Exception)


class TestMissingElementError:
    """Tests for MissingElementError exception."""

    def test_attributes(self):
        e = exc.MissingElementError('Zn', ['Cu', 'Zn'], 'alpha')
        assert e.element == 'Zn'
        assert e.element_list == ['Cu', 'Zn']
        assert e.name == 'alpha'

    def test_str_representation(self):
        e = exc.MissingElementError('Fe', ['Cu', 'Zn'], 'alpha')
        msg = str(e)
        assert "'Fe'" in msg
        assert "'alpha'" in msg
        assert "['Cu', 'Zn']" in msg

    def test_default_name(self):
        e = exc.MissingElementError('Fe', ['Cu', 'Zn'])
        assert e.name == 'unknown'


class TestIncompatiblePhaseError:
    """Tests for IncompatiblePhaseError exception."""

    def test_attributes(self):
        mock_phase = type('Phase', (), {'name': 'beta', 'element_list': ['Cu', 'Al']})()
        e = exc.IncompatiblePhaseError(['Cu', 'Zn'], mock_phase)
        assert e.phase_diagram_ele_list == ['Cu', 'Zn']
        assert e.phase.name == 'beta'

    def test_str_representation(self):
        mock_phase = type('Phase', (), {'name': 'beta', 'element_list': ['Cu', 'Al']})()
        e = exc.IncompatiblePhaseError(['Cu', 'Zn'], mock_phase)
        msg = str(e)
        assert 'beta' in msg
        assert "'Cu', 'Al'" in msg


class TestCompositionNotMatchingElementListError:
    """Tests for CompositionNotMatchingElementListError exception."""

    def test_attributes(self):
        e = exc.CompositionNotMatchingElementListError(
            {'Cu': 0.5, 'Zn': 0.5}, ['Cu', 'Zn'], 'alpha'
        )
        assert e.composition == {'Cu': 0.5, 'Zn': 0.5}
        assert e.element_list == ['Cu', 'Zn']
        assert e.name == 'alpha'

    def test_default_name(self):
        e = exc.CompositionNotMatchingElementListError({'Cu': 1.0}, ['Cu'])
        assert e.name == 'unknown'

    def test_str_representation(self):
        e = exc.CompositionNotMatchingElementListError(
            {'Fe': 1.0}, ['Cu', 'Zn'], 'alpha'
        )
        msg = str(e)
        assert 'alpha' in msg
        assert "'Fe'" in msg


class TestPhaseDiagramEmpty:
    """Tests for PhaseDiagramEmpty exception."""

    def test_str_representation(self):
        e = exc.PhaseDiagramEmpty()
        assert str(e) == 'The phase diagram is empty.'

    def test_can_be_raised(self):
        with pytest.raises(exc.PhaseDiagramEmpty):
            raise exc.PhaseDiagramEmpty


class TestPhaseNotFound:
    """Tests for PhaseNotFound exception."""

    def test_attributes(self):
        mock_pd = type('PD', (), {'phase_names': ['alpha', 'beta']})()
        e = exc.PhaseNotFound(mock_pd, 'gamma')
        assert e.phases == ['alpha', 'beta']
        assert e.given_phase == 'gamma'

    def test_str_representation(self):
        mock_pd = type('PD', (), {'phase_names': ['alpha', 'beta']})()
        e = exc.PhaseNotFound(mock_pd, 'gamma')
        msg = str(e)
        assert 'gamma' in msg
        assert 'alpha' in msg


class TestAtomNotFoundForCluster:
    """Tests for AtomNotFoundForCluster exception."""

    def test_str_representation(self):
        e = exc.AtomNotFoundForCluster()
        assert str(e) == 'The given atom type has no geometry description for clusters.'

    def test_can_be_raised(self):
        with pytest.raises(exc.AtomNotFoundForCluster):
            raise exc.AtomNotFoundForCluster


class TestMissingMandatoryParameterError:
    """Tests for MissingMandatoryParameterError exception."""

    def test_can_be_raised(self):
        with pytest.raises(exc.MissingMandatoryParameterError):
            raise exc.MissingMandatoryParameterError

    def test_is_exception(self):
        assert issubclass(exc.MissingMandatoryParameterError, Exception)
