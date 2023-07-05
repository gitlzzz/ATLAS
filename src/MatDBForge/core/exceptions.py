class BaseStructureNotFound(Exception):
    pass


class PhaseNotFound(Exception):
    def __init__(self, phase_diagram, given_phase):
        self.phases = phase_diagram.phase_names
        self.given_phase = given_phase

    def __str__(self):

        return (
            f"The given phase '{self.given_phase}' str is not found in the phase diagram, "
            f"which has the following phases:\n {self.phases}"
        )
