"""General classes for representing phase diagrams of materials."""

from pymatgen.core.periodic_table import Element

from MatDBForge.core import exceptions as mdb_exc


class BinaryPhaseDiagram:
    """
    Class representing a binary phase diagram of materials.

    Parameters
    ----------
        material (str): The name of the material.
        *phases (Phase): Variable number of Phase objects representing the
                         phases in the diagram.

    Attributes
    ----------
        phases (list): List of Phase objects representing the phases in the diagram.
        material (str): The name of the material.

    """

    def __init__(self, material: str, *phases: "Phase"):
        self.phases = []
        self.phase_names = []
        self.material = material
        self.phase_dict = {}

        for phase in phases:
            self.add_phase(phase)

        for phase in self.phases:
            self.phase_names.append(phase.name)

        self.alloy_set = self.get_alloy_set()

    def add_phase(self, phase):
        """Add a phase to the phase diagram.

        Parameters
        ----------
            phase (Phase): The Phase object to add.

        """
        self.phases.append(phase)
        phase.phasediagram = self.__class__
        self.phase_dict[phase.name] = phase

    def get_phase(self, phase):
        """
        Gets a Phase object from a BinaryPhaseDiagram using either
        the phase name (as a string) or a Phase object.

        Parameters
        ----------
        phase : str | Phase
            Phase or name of the desired phase.

        Returns
        -------
        Phase
            Phase object corresponding to the given phase

        Raises
        ------
        TypeError
            _description_
        TypeError
            _description_
        """
        if isinstance(phase, Phase):
            return self.phase_dict[phase.name]
        if isinstance(phase, str):
            phase_str = phase
            phase = self.phase_dict.get(phase_str, None)
            if phase:
                return phase
            else:
                raise mdb_exc.PhaseNotFound(self, phase_str)
        else:
            raise TypeError("The given phase object is not a Phase-like object.")

    def __repr__(self):
        """
        Override attribute lookup.
        Allows accessing phases by their names as attributes.

        Parameters
        ----------
            name (str): The name of the attribute.

        Returns
        -------
            Phase: The Phase object with the specified name.

        Raises
        ------
            AttributeError: If the attribute is not found.

        """
        repr_str = (
            f"{self.material} phase diagram with phases:"
            f" {[phase.name for phase in self.phases]}"
        )

        return repr_str

    def __getattr__(self, name):
        if name in self.phase_dict:
            return self.phase_dict[name]
        else:
            raise AttributeError(f"'PhaseDiagram' object has no attribute '{name}'")

    def get_phase_for_structure(self, structure):
        print(structure.formula)

    def get_alloy_set(self):
        element_list = []

        if len(self.phases) == 0:
            raise mdb_exc.PhaseDiagramEmpty()
        for phase in self.phases:
            element_list.append(Element(phase.base_elem))
            element_list.append(Element(phase.cluster_elem))
        element_set = set(element_list)
        return element_set


# TODO: Imlpement this
class TernaryPhaseDiagram:
    """Class representing a ternary phase diagram of materials."""

    ...


class Phase:
    """Class representing a phase in a phase diagram.

    Parameters
    ----------
    name: str
        The name of the phase.
    base_elem:
        The base element of the phase.
    base_elem_comp_max: float
        The maximum composition of the base element in the phase.
    base_elem_comp_min: float
        The minimum composition of the base element in the phase.
    prototype: str
        The prototype of the phase.
    offset: float
        The offset value of the phase.
    phase_diagram: PhaseDiagram
        The parent PhaseDiagram object that the phase belongs to.

    """

    def __init__(
        self,
        name: str,
        base_elem,
        cluster_elem,
        base_elem_comp_max: float,
        base_elem_comp_min: float,
        prototype: str,
        offset: float = 0,
        phase_diagram: "BinaryPhaseDiagram" = None,
    ):
        self.name = name
        self.base_elem = Element(base_elem)
        self.cluster_elem = Element(cluster_elem)
        self.base_elem_comp_max = float(base_elem_comp_max)
        self.base_elem_comp_min = float(base_elem_comp_min)
        self.prototype = prototype
        self.offset = float(offset)
        self.phase_diagram = phase_diagram

        # if phase_diagram is not None:
        # phase_diagram.add_phase(self)
        # self.phase_diagram = phase_diagram.__name__

    def __str__(self):
        """Return a string representation of the phase.

        Returns
        -------
            str: The string representation of the phase.

        """
        repr_string = (
            f"Phase '{self.name}', {self.base_elem_comp_min*100:.1f}%"
            f" {self.base_elem} - {self.base_elem_comp_max*100:.1f}%"
            f" {self.base_elem} (± {self.offset*100:.1f}%)"
        )

        if self.phase_diagram is not None:
            # repr_string += f" (belongs to {self.phase_diagram.material})"
            repr_string += f" (belongs to {self.phase_diagram})"

        return repr_string

    def __repr__(self):
        """Return a string representation of the phase.

        Returns
        -------
            str: The string representation of the phase.

        """
        repr_string = (
            f"Phase '{self.name}', {self.base_elem_comp_min*100:.1f}%"
            f" {self.base_elem} - {self.base_elem_comp_max*100:.1f}%"
            f" {self.base_elem} (± {self.offset*100:.1f}%)"
        )

        if self.phase_diagram is not None:
            # repr_string += f" (belongs to {self.phase_diagram.material})"
            repr_string += f" (belongs to {self.phase_diagram})"

        return repr_string

    def __key(self):
        return (
            self.name,
            self.base_elem,
            self.base_elem_comp_max,
            self.base_elem_comp_min,
            self.prototype,
            self.offset,
            self.phase_diagram,
        )

    def __eq__(self, other):
        if not isinstance(other, Phase):
            # Do not try to compare against different types
            return NotImplemented
        return self.__key() == other.__key()

    def __hash__(self):
        return hash(self.__key())

    def perc_in_phase(self, perc: float, offset: bool = True) -> bool:
        if perc > 1:
            perc /= 100

        offset = self.offset if offset else 0

        inPhase = (
            (self.base_elem_comp_min - offset)
            < perc
            < (self.base_elem_comp_max + offset)
        )

        return bool(inPhase)

    def get_base_elem_perc(self, structure) -> float:
        comp_dict = structure.composition.fractional_composition.as_dict()
        comp_base = comp_dict.get(self.base_elem.symbol, 0.0)
        return comp_base
