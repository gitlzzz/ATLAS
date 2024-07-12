class BaseStructureNotFound(Exception):
    pass


class EmptyDataBase(Exception):
    def __str__(self):
        return "A database could not be read from the given path."


class FilterError(Exception):
    pass

class PhaseDiagramEmpty(Exception):
    def __str__(self):
        return "The phase diagram is empty."


class PhaseNotFound(Exception):
    def __init__(self, phase_diagram, given_phase):
        self.phases = phase_diagram.phase_names
        self.given_phase = given_phase

    def __str__(self):
        return (
            f"The given phase '{self.given_phase}' str is not found in the phase"
            f" diagram, which has the following phases:\n {self.phases}"
        )


class AtomNotFoundForCluster(Exception):
    def __str__(self):
        return "The given atom type has no geometry description for clusters."
