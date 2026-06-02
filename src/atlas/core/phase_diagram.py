"""General classes for representing phase diagrams of materials."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from pymatgen.core.periodic_table import Element
from slugify import slugify

from atlas.core import exceptions as atl_exc


def PhaseDiagram(material: str, element_list: list, base_elem: str, *phases: 'Phase'):
    """
    Factory method to return a BinaryPhaseDiagram or TernaryPhaseDiagram
    depending on the number of elements in the element_list.

    Parameters
    ----------
    material : str
        The name of the material.
    element_list : list
        List of elements in the phase diagram.
    base_elem: str
        Symbol of the base element.
    *phases : Phase
        Variable number of Phase objects representing the phases in the diagram.

    Returns
    -------
    BinaryPhaseDiagram | TernaryPhaseDiagram
    """
    if len(element_list) == 1:
        return SinglePhaseDiagram(material, element_list, base_elem, *phases)
    elif len(element_list) == 2:
        return BinaryPhaseDiagram(material, element_list, base_elem, *phases)
    elif len(element_list) == 3:
        return TernaryPhaseDiagram(material, element_list, base_elem, *phases)
    else:
        raise NotImplementedError(
            'PhaseDiagram only supports binary or ternary diagrams.'
        )


class BasePhaseDiagram:
    """
    Base class representing a phase diagram of materials.

    Parameters
    ----------
    material : str
        The name of the material.
    base_elem : str
        The base element of the phase diagram.
    *phases : Phase
        Variable number of Phase objects representing the phases in the diagram.

    Attributes
    ----------
        phases : list
            List of Phase objects representing the phases in the diagram.
        material : str
            The name of the material.
    """

    def __init__(
        self, material: str, element_list: list, base_elem: str, *phases: 'Phase'
    ):
        self.phases = []
        self.phase_names = []
        self.material = material
        self.phase_dict = {}

        # Convert elements to Element objects and store
        self.element_list = [Element(ele) for ele in sorted(element_list)]

        # Parsing base element
        if Element(base_elem) in self.element_list:
            self.base_elem = Element(base_elem)
        else:
            raise atl_exc.MissingElementError(base_elem, element_list)

        # Add the provided phases
        if len(phases) > 0:
            for phase in phases:
                self.add_phase(phase)

        for phase in self.phases:
            self.phase_names.append(phase.name)

        self.alloy_set = set(element_list)

    def add_phase(self, phase):
        """Add a phase to the phase diagram."""
        if self.element_list != phase.element_list:
            raise atl_exc.MissingElementError(phase.base_elem, self.element_list)

        self.phases.append(phase)

        # If unassigned, assign the phase diagram to the current phase
        if phase.phase_diagram is None:
            phase.phase_diagram = self

        self.phase_dict[phase.name] = phase
        if phase.name not in self.phase_names:
            self.phase_names.append(phase.name)

    def get_phase(self, phase) -> 'Phase':
        """Get a phase object from the phase diagram."""
        if isinstance(phase, Phase):
            return self.phase_dict[phase.name]
        if isinstance(phase, str):
            phase = self.phase_dict.get(slugify(phase), None)
            if phase:
                return phase
            else:
                return None
        else:
            raise TypeError('The given phase object is not a Phase-like object.')

    def _calculate_centroid(self, vertices):
        """Calculte the centroid of a polygon.

        https://en.wikipedia.org/wiki/Centroid#Of_a_polygon

        Parameters
        ----------
        vertices : (n, 2) np.ndarray
            Vertices of a polygon.

        Returns
        -------
        centroid : (2, ) np.ndarray
            Centroid of the polygon.
        """
        roll0 = np.roll(vertices, 0, axis=0)
        roll1 = np.roll(vertices, 1, axis=0)
        cross = np.cross(roll0, roll1)
        area = 0.5 * np.sum(cross)
        return np.sum((roll0 + roll1) * cross[:, None], axis=0) / (6.0 * area)

    def __repr__(self):
        description = f"{self.__class__.__name__} named '{self.material}', for elements"
        description += f' {[str(ele) for ele in self.element_list]} and '
        description += 'containing '

        if len(self.phases) == 0:
            description += 'no phases (empty).'
        else:
            phase_list = [phase.name for phase in self.phases]
            description += f'{len(phase_list)} phases: {phase_list}'

        return description


class SinglePhaseDiagram(BasePhaseDiagram):
    """Binary phase diagram class for one element."""

    def __init__(
        self, material: str, element_list: list, base_elem: str, *phases: 'Phase'
    ):
        if len(element_list) > 1:
            raise ValueError(
                f'{self.__class__.__name__} requires exactly 1 element. '
                f'However, this does not match the current element list: {element_list}'
            )
        super().__init__(material, element_list, base_elem, *phases)

    def plot_diagram(self, **kwargs):
        # pd = pmg_phasediagram(struct_comp_base_elem)
        # PDPlotter(pd).get_plot()
        pass


class BinaryPhaseDiagram(BasePhaseDiagram):
    """Binary phase diagram class for two elements."""

    def __init__(
        self, material: str, element_list: list, base_elem: str, *phases: 'Phase'
    ):
        if len(element_list) != 2:
            raise ValueError(
                f'{self.__class__.__name__} requires exactly 2 elements. '
                f'However, this does not match the current element list: {element_list}'
            )
        super().__init__(material, element_list, base_elem, *phases)

    def plot_diagram(
        self,
        max_temp_K: float = 1000,
        min_temp_K: float = 300,
        rc_params=None,
        show_plot=True,
        ax=None,
    ) -> plt.Axes:
        """
        Plot a basic binary phase diagram of the material with temperature
        and composition axes.

        Parameters
        ----------
        max_temp_K : float, optional
            The maximum temperature in Kelvin for the y-axis (default is 1000 K).
        min_temp_K : float, optional
            The minimum temperature in Kelvin for the y-axis (default is 300 K).
        rc_params : dict, optional
            Dictionary of matplotlib rcParams to override default plotting
            parameters (default is None).

        Returns
        -------
        plt.Axes
            The matplotlib Axes object containing the plot.

        Notes
        -----
        - The function generates a plot with the x-axis representing the composition
          (in at. %) of the base element in the material, and the y-axis representing
          the temperature in Kelvin.

        - Each phase is represented by a filled patch in the diagram, and the phases
          are labeled  at their centroid positions.

        - The function uses the `viridis` colormap to assign colors to phases.

        Examples
        --------
        >>> diagram.plot_diagram(
        >>>    max_temp_K=1200,
        >>>    min_temp_K=400,
        >>>    rc_params={'figure.figsize': (10, 6)}
        >>> )
        """
        # Update the rcParams if provided
        if rc_params:
            rcParams.update(rc_params)

        # Create a new figure if ax is not provided
        if not ax:
            ax = plt.subplot()

        # Get the number of phases and create n colors from the viridis colormap
        color_list = plt.cm.viridis(np.linspace(0, 1, len(self.phases)))
        ini_zorder = 2.1 + (len(self.phases) * 0.1)

        # Getting list of equidistant heights for phase labels,
        # in order to avoid overlapping.
        phase_label_heights = np.linspace(
            start=min_temp_K + 0.05 * min_temp_K,
            stop=max_temp_K - 0.05 * max_temp_K,
            num=len(self.phases),
        )

        # Plot the phases
        for idx, phase in enumerate(self.phases):
            comp_min = phase.composition[str(self.base_elem)]['min'] * 100
            comp_max = phase.composition[str(self.base_elem)]['max'] * 100
            arr = np.array(
                [
                    (comp_min, max_temp_K),
                    (comp_min, min_temp_K),
                    (comp_max, min_temp_K),
                    (comp_max, max_temp_K),
                ]
            )
            patch = ax.fill(
                arr[:, 0],
                arr[:, 1],
                ec='k',
                fc=color_list[idx],
                alpha=0.1,
                zorder=ini_zorder - (idx * 0.1),
            )

            # Write name
            label = phase.name

            # Getting center of phase to place text
            centroid = self._calculate_centroid(patch[0].get_xy())

            ax.text(
                centroid[0],
                phase_label_heights[idx],
                label,
                ha='center',
                va='center',
                transform=ax.transData,
                rotation=25,
                weight='bold',
            )

        # Update the chart layout
        ax.grid(which='both')

        # Add complimentar elements to the x-axis
        secax = ax.twiny()
        secax.set_xlabel(
            f'Composition at. % {list(self.alloy_set - {str(self.base_elem)})[0]}'
        )
        secax.set_xlim(105, -5)

        ax.set_xlabel(f'Composition at. % {self.base_elem}')
        ax.set_ylabel('T [K]')
        ax.set_xlim(-5, 105)
        ax.set_ylim(min_temp_K, max_temp_K)
        ax.set_title(f"'{self.material}' Binary Phase Diagram")

        # Plot the phase diagram if show_plot is True
        if show_plot:
            plt.show()

        return ax


class TernaryPhaseDiagram(BasePhaseDiagram):
    """Ternary phase diagram class for three elements."""

    def __init__(
        self, material: str, element_list: list, base_elem: str, *phases: 'Phase'
    ):
        if len(element_list) != 3:
            raise ValueError('TernaryPhaseDiagram requires exactly 3 elements.')
        super().__init__(material, element_list, base_elem, *phases)

    def plot_diagram(self):
        ax = plt.subplot(projection='ternary', ternary_sum=100.0)

        # Get the number of phases and create n colors from the viridis colormap
        color_list = plt.cm.viridis(np.linspace(0, 1, len(self.phases)))

        # for (key, value), color in zip(self.phases.items(), _Set3_data):
        for idx, curr_phase in enumerate(self.phases):
            tn0, tn1, tn2 = np.array(curr_phase).T
            patch = ax.fill(
                tn0, tn1, tn2, ec='k', fc=color_list[idx], alpha=0.1, zorder=2.1
            )
            centroid = self._calculate_centroid(patch[0].get_xy())

            # last space replaced with line break
            label = curr_phase.name
            ax.text(
                centroid[0],
                centroid[1],
                label,
                ha='center',
                va='center',
                transform=ax.transData,
            )

        ax.taxis.set_major_locator(MultipleLocator(10.0))
        ax.laxis.set_major_locator(MultipleLocator(10.0))
        ax.raxis.set_major_locator(MultipleLocator(10.0))

        ax.taxis.set_minor_locator(AutoMinorLocator(2))
        ax.laxis.set_minor_locator(AutoMinorLocator(2))
        ax.raxis.set_minor_locator(AutoMinorLocator(2))

        ax.grid(which='both')

        ax.set_tlabel('Clay (%)')
        ax.set_llabel('Sand (%)')
        ax.set_rlabel('Silt (%)')

        ax.taxis.set_ticks_position('tick2')
        ax.laxis.set_ticks_position('tick2')
        ax.raxis.set_ticks_position('tick2')


class Phase:
    """Class representing a phase in a phase diagram.

    Parameters
    ----------
    name: str
        The name of the phase.
    base_elem:
        The base element of the phase.
    prototype: str | list
        The prototype(s) for the current phase. Either a string
        representing an ID from the Materials Project database
        or a list of IDs.
    offset: float
        The offset value of the phase.
    phase_diagram: PhaseDiagram
        The parent PhaseDiagram object that the phase belongs to.
    replace_dict: dict
        A dictionary of replacements for the prototype structure.
    spacegroup: str
        The spacegroup of the phase.
    symbol: str
        The symbol of the phase.
    """

    def __init__(
        self,
        name: str,
        element_list: list,
        composition: dict,
        prototype: str | list,
        offset: float = 0,
        phase_diagram: PhaseDiagram = None,
        cluster_elem: str = None,
        replace_dict: dict = None,
        base_elem: str = None,
        allow_modifications: bool = True,
        use_cache: bool = False,
        spacegroup: str = None,
        symbol: str = None,
    ):
        self.name = slugify(name)
        self.spacegroup = spacegroup
        self.symbol = symbol
        self.original_name = name
        self.phase_diagram = phase_diagram
        self.element_list = [Element(ele) for ele in sorted(element_list)]

        if not base_elem:
            self.base_elem = self.phase_diagram.base_elem
        elif Element(base_elem) in self.element_list:
            self.base_elem = Element(base_elem)
        else:
            raise atl_exc.MissingElementError(base_elem, element_list, self.name)

        # cluster_elem should be the base_base element if not specified
        if cluster_elem is None:
            self.cluster_elem = self.base_elem
        elif Element(cluster_elem) in self.element_list:
            self.cluster_elem = Element(cluster_elem)
        else:
            raise atl_exc.MissingElementError(cluster_elem, element_list, self.name)

        if set(composition.keys()) != set(element_list):
            raise atl_exc.CompositionNotMatchingElementListError(
                str(list(composition.keys())), str(element_list), self.name
            )
        self.composition = composition

        self.base_elem_comp_max = float(self.composition[str(self.base_elem)]['max'])
        self.base_elem_comp_min = float(self.composition[str(self.base_elem)]['min'])

        self.prototype = prototype

        if replace_dict:
            self.replace_dict = replace_dict
        else:
            self.replace_dict = {}

        self.offset = float(offset)
        self.allow_modifications = allow_modifications

        self.use_cache = use_cache

    def __str__(self):
        """Return a string representation of the phase.

        Returns
        -------
            str: The string representation of the phase.

        """
        repr_string = f"Phase: '{self.name}' |"

        for ele in self.composition:
            repr_string += (
                f' {ele}: {self.composition[ele]["min"] * 100:.1f}% -'
                f' {self.composition[ele]["max"] * 100:.1f}%,'
            )

        repr_string += ' |'
        if self.phase_diagram is not None:
            repr_string += (
                f' Belongs to {self.phase_diagram.__class__.__name__}, '
                f"'{self.phase_diagram.material}'"
            )

        return repr_string

    def __repr__(self):
        """Return a string representation of the phase.

        Returns
        -------
            str: The string representation of the phase.

        """
        return f"phase: '{self.name}'"

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
        """Check if a given composition percentage is within the phase."""
        if perc > 1:
            perc /= 100

        offset = self.offset if offset else 0

        min_range = max(self.base_elem_comp_min - offset, 0)
        max_range = min(self.base_elem_comp_max + offset, 1)
        inPhase = min_range < perc < max_range

        return bool(inPhase)

    def get_base_elem_perc(self, structure) -> float:
        """Get the percentage of the base element in a structure."""
        comp_dict = structure.composition.fractional_composition.as_dict()
        comp_base = comp_dict.get(self.base_elem.symbol, 0.0)
        return comp_base
