import numpy as np
import pennylane as qml


class ChemistryEnvironment:
    """
    Encapsulates all physics needed to run VQE on a molecule.

    Parameters
    ----------
    symbols : list of str
        Atomic symbols, e.g. ["H", "H"].
    coordinates : ndarray
        Flat array of nuclear positions in Bohr:
        [x0, y0, z0, x1, y1, z1, ...].
    charge : int
        Net molecular charge.
    mult : int
        Spin multiplicity (2S + 1).
    basis : str
        Basis set name for the electronic structure calculation.
    """

    def __init__(self, symbols, coordinates, charge=0, mult=1, basis="sto-3g"):
        self.symbols = symbols
        self.coordinates = np.array(coordinates, dtype=float)
        self.charge = charge
        self.mult = mult
        self.basis = basis

        self.hamiltonian = None
        self.n_qubits = None
        self.n_electrons = None
        self.hf_state = None

        self.build()

    def build(self):
        """Construct the molecular Hamiltonian and Hartree-Fock reference state."""
        self.hamiltonian, self.n_qubits = qml.qchem.molecular_hamiltonian(
            self.symbols,
            self.coordinates,
            charge=self.charge,
            mult=self.mult,
            basis=self.basis,
        )
        self.n_electrons = sum(
            self._atomic_number(s) for s in self.symbols
        ) - self.charge
        self.hf_state = qml.qchem.hf_state(self.n_electrons, self.n_qubits)

    def prepare_state(self, wires):
        """Apply the Hartree-Fock basis state to the circuit (replaces embedding)."""
        qml.BasisState(self.hf_state, wires=wires)


    @classmethod
    def hydrogen(cls, bond_length=1.4):
        """
        Build an H2 molecule along the z-axis.

        Parameters
        ----------
        bond_length : float
            Inter-atomic distance in Bohr (default 1.4 ~ 0.74 Angstrom,
            the equilibrium geometry).
        """
        symbols = ["H", "H"]
        coordinates = np.array([0.0, 0.0, -bond_length / 2,
                                0.0, 0.0,  bond_length / 2])
        return cls(symbols, coordinates)


    @staticmethod
    def _atomic_number(symbol):
        table = {
            "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6,
            "N": 7, "O": 8, "F": 9, "Ne": 10,
        }
        return table[symbol]

    def __repr__(self):
        formula = "".join(self.symbols)
        return (
            f"ChemistryEnvironment({formula}, "
            f"qubits={self.n_qubits}, electrons={self.n_electrons})"
        )
