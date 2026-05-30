"""Custom ASE calculators for ATL active learning."""

import numpy as np
from ase import Atoms
from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)

import atlas.core.code_utils as atl_cut


# Define a custom exception for unphysical energies/forces
class UnphysicalStateError(Exception):
    """Custom exception raised when unphysical energies or forces are detected."""

    pass


class ATLSafeCalculatorWrapper(Calculator):
    """
    A wrapper around an ASE calculator that checks for unphysical energies
    (too high, NaN, or Inf) or forces (NaN or Inf) and raises
    UnphysicalStateError to allow for graceful termination of MD.
    """

    implemented_properties = None  # Will be set from the wrapped calculator

    def __init__(self, calculator, max_energy_threshold_per_atom, atoms=None, **kwargs):
        """
        Initialize the wrapper.

        Parameters
        ----------
        calculator : ase.calculators.calculator.Calculator
            The ASE calculator to wrap (e.g., a MACE calculator).
        max_energy_threshold_per_atom : float
            The maximum allowed potential energy per atom (in eV).
            If the total potential energy divided by the number of atoms
            exceeds this, an UnphysicalStateError is raised.
        atoms : ase.Atoms, optional
            The Atoms object to be used with the calculator.
        **kwargs
            Keyword arguments for the base Calculator class.
        """
        super().__init__(atoms=atoms, **kwargs)
        self.calculator = calculator
        self.max_energy_threshold_per_atom = max_energy_threshold_per_atom
        self._num_atoms = None  # Will be set on first calculate

        # Inherit implemented_properties and parameters from the wrapped calculator
        if hasattr(calculator, 'implemented_properties'):
            self.implemented_properties = list(calculator.implemented_properties)
        else:
            # Fallback if the wrapped calculator doesn't explicitly list them
            # This is a basic set, real calculators might have more (e.g., stress)
            self.implemented_properties = ['energy', 'forces']
            atl_cut.custom_print(
                'Warning: Wrapped calculator does not explicitly state '
                "implemented_properties. Defaulting to ['energy', 'forces'].",
                'warn',
            )

        if hasattr(calculator, 'parameters'):
            self.parameters = calculator.parameters.copy()
        else:
            self.parameters = {}  # Ensure parameters attribute exists

        # Ensure 'energy' and 'forces' are in implemented_properties if we
        # are to check them
        if 'energy' not in self.implemented_properties:
            self.implemented_properties.append('energy')
        if 'forces' not in self.implemented_properties:
            self.implemented_properties.append('forces')

    def _update_num_atoms(self, atoms_obj):
        if self._num_atoms is None or self._num_atoms != len(atoms_obj):
            self._num_atoms = len(atoms_obj)
            if self._num_atoms == 0:
                raise ValueError('Cannot calculate with zero atoms.')
            self.max_total_energy_threshold = (
                self.max_energy_threshold_per_atom * self._num_atoms
            )
            atl_cut.custom_print(
                f'SafeCalculatorWrapper: System has {self._num_atoms} atoms. '
                'Max total energy threshold set to: '
                f'{self.max_total_energy_threshold:.4f} eV'
            )

    def calculate(self, atoms: Atoms, properties, system_changes):
        """
        Perform the calculation.

        This method calls the wrapped calculator and then checks the results.
        """
        if atoms is not None:
            self._update_num_atoms(atoms)
        elif self.atoms is not None:
            self._update_num_atoms(self.atoms)
        else:
            # This case should ideally not happen if atoms are passed correctly
            # during MD or optimization.
            raise ValueError(
                'Atoms object not available to SafeCalculatorWrapper for determining '
                'number of atoms.'
            )

        # Call the original calculator
        try:
            self.calculator.calculate(
                atoms=atoms, properties=properties, system_changes=system_changes
            )
        except PropertyNotImplementedError:  # Passthrough this specific ASE error
            raise
        except Exception as e:
            # If the underlying calculator itself crashes (e.g., internal MACE error)
            raise UnphysicalStateError(f'Wrapped calculator failed: {e}') from e

        # Get results from the original calculator
        self.results = self.calculator.results.copy()

        # Check for NaN or Inf in energy
        if 'energy' in self.results:
            energy = self.results['energy']
            if not np.isfinite(energy):
                msg = f'Unphysical energy detected: {energy} eV. Stopping.'
                # To ensure the dynamics stops, we can set properties to something
                # that won't cause further numerical errors if an exception
                # isn't caught perfectly.
                # However, raising is the primary mechanism.
                self.results['energy'] = (
                    self.max_total_energy_threshold * 2
                )  # Force high
                if 'forces' in self.results and atoms is not None:
                    self.results['forces'] = np.zeros((len(atoms), 3))
                raise UnphysicalStateError(msg)

            # Check against the threshold
            if energy > self.max_total_energy_threshold:
                msg = (
                    f'Potential energy {energy:.4f} eV exceeded threshold '
                    f'{self.max_total_energy_threshold:.4f} eV. Stopping.'
                )
                raise UnphysicalStateError(msg)

        # Check for NaN or Inf in forces
        if 'forces' in self.results:
            forces = self.results['forces']
            if not np.all(np.isfinite(forces)):
                msg = 'Unphysical forces detected (NaN or Inf). Stopping.'
                atl_cut.custom_print(f'SafeCalculatorWrapper: {msg}')
                # If energy was not calculated or was finite, set it high to help stop
                if 'energy' not in self.results or np.isfinite(
                    self.results.get('energy', 0)
                ):
                    self.results['energy'] = self.max_total_energy_threshold * 2
                if atoms is not None:
                    self.results['forces'] = np.zeros((len(atoms), 3))
                raise UnphysicalStateError(msg)

    def get_property(self, name, atoms=None, allow_calculation=True):
        """Get a property from the calculator.

        This ensures that even direct calls like `atoms.get_potential_energy()`
        go through the check if a calculation is triggered.
        """
        if name not in self.implemented_properties:
            raise PropertyNotImplementedError(f'{name} property not implemented')

        if atoms is None:
            atoms = self.atoms

        if self.calculation_required(atoms, [name]):
            if not allow_calculation:
                return None
            self.calculate(atoms=atoms, properties=[name], system_changes=all_changes)

        if name not in self.results:
            # This can happen if calculate() was called with a subset of properties
            # and the requested property wasn't among them. Recalculate.
            if not allow_calculation:
                return None
            self.calculate(atoms=atoms, properties=[name], system_changes=all_changes)

        if name not in self.results:
            # Should not happen if calculate ran correctly for the property
            raise RuntimeError(
                f"Property '{name}' not found in results after calculation."
            )

        return self.results[name]
