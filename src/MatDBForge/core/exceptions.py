"""Custom exceptions for the MatDBForge package."""


class BaseStructureNotFound(Exception):
    """Raised when the base structure is not found in the database."""

    pass


class EmptyDataBase(Exception):
    """Raised when the database is empty."""

    def __str__(self):
        return "A database could not be read from the given path."


class FilterError(Exception):
    """Raised when the filter is not valid."""

    pass


class PhaseDiagramEmpty(Exception):
    """Raised when the phase diagram is empty."""

    def __str__(self):
        return "The phase diagram is empty."


class PhaseNotFound(Exception):
    """Raised when the phase is not found in the phase diagram."""

    def __init__(self, phase_diagram, given_phase):
        self.phases = phase_diagram.phase_names
        self.given_phase = given_phase

    def __str__(self):
        return (
            f"The given phase '{self.given_phase}' str is not found in the phase"
            f" diagram, which has the following phases:\n {self.phases}"
        )


class AtomNotFoundForCluster(Exception):
    """Raised when the atom type is not found in the cluster."""

    def __str__(self):
        return "The given atom type has no geometry description for clusters."


class MissingMandatoryParameterError(Exception):
    """Raised when a mandatory parameter is missing in the toml dictionary."""

    pass
