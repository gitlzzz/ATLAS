"""Custom exceptions for the MatDBForge package."""


class BaseStructureNotFound(Exception):
    """Raised when the base structure is not found in the database."""

    pass


class EmptyDataBase(Exception):
    """Raised when the database is empty."""

    def __str__(self):
        return 'A database could not be read from the given path.'


class FilterError(Exception):
    """Raised when the filter is not valid."""

    pass


class MissingElementError(Exception):
    """Raised when the element is not found in the element_list."""

    def __init__(self, element, element_list):
        self.element = element
        self.element_list = element_list

    def __str__(self):
        return (
            f"The element '{self.element}' is not found in the "
            f'current phase element list, '
            f'which has the following elements: {self.element_list}.'
            f'\nCheck your `base_element` or `cluster_element` options.'
        )


class IncompatiblePhaseError(Exception):
    """Raised when the phase has elements not expected by the phase diagram."""

    def __init__(self, phase_diagram_ele_list, phase):
        self.phase_diagram_ele_list = phase_diagram_ele_list
        self.phase = phase

    def __str__(self):
        return (
            f"The phase '{self.phase.name}' has an element list incompatible with "
            f'the phase diagram. Check the elements in the phase:'
            f'\nPhase elements: {self.phase.element_list}'
            f'\nPhase diagram elements: {self.phase_diagram_ele_list}'
        )


class CompositionNotMatchingElementListError(Exception):
    """Raised when the composition does not match the element list."""

    def __init__(self, composition, element_list):
        self.composition = composition
        self.element_list = element_list

    def __str__(self):
        return (
            f"The composition '{self.composition}' for the current phase"
            ' does not match the element list.'
            f'\nElement list: {self.element_list}'
        )


class PhaseDiagramEmpty(Exception):
    """Raised when the phase diagram is empty."""

    def __str__(self):
        return 'The phase diagram is empty.'


class PhaseNotFound(Exception):
    """Raised when the phase is not found in the phase diagram."""

    def __init__(self, phase_diagram, given_phase):
        self.phases = phase_diagram.phase_names
        self.given_phase = given_phase

    def __str__(self):
        return (
            f"The given phase '{self.given_phase}' str is not found in the phase"
            f' diagram, which has the following phases:\n {self.phases}'
        )


class AtomNotFoundForCluster(Exception):
    """Raised when the atom type is not found in the cluster."""

    def __str__(self):
        return 'The given atom type has no geometry description for clusters.'


class MissingMandatoryParameterError(Exception):
    """Raised when a mandatory parameter is missing in the toml dictionary."""

    pass
