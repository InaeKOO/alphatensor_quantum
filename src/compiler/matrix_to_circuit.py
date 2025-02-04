import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator

def matrix_to_circuit(matrix: np.ndarray) -> QuantumCircuit:
    """Converts a unitary matrix to a quantum circuit.

    Args:
        matrix: A unitary matrix.

    Returns:
        A QuantumCircuit object representing the matrix.
    """
    # Validate the matrix is unitary
    if not np.allclose(matrix @ matrix.conj().T, np.eye(matrix.shape[0])):
        raise ValueError("Input matrix is not unitary.")

    # Create a quantum circuit from the matrix
    num_qubits = int(np.log2(matrix.shape[0]))
    circuit = QuantumCircuit(num_qubits)
    operator = Operator(matrix)
    circuit.unitary(operator, range(num_qubits))

    # Optionally, transpile the circuit to optimize it
    circuit = transpile(circuit, basis_gates=['u3', 'cx'])

    return circuit