from qiskit import QuantumCircuit, transpile, passmanager
from qiskit.quantum_info import Operator, random_unitary
from qiskit.circuit.library import UnitaryGate
from qiskit.transpiler.passes.synthesis import UnitarySynthesis
import numpy as np

def qsd_recursive(qc, unitary, qubits):
    """Recursively applies Quantum Shannon Decomposition (QSD) to break a unitary into smaller blocks."""
    num_qubits = len(qubits)

    if num_qubits == 1:
        # Base case: Apply a single-qubit unitary directly
        qc.unitary(unitary, [qubits[0]])
        return

    # Perform Cosine-Sine Decomposition (CSD)
    U = Operator(unitary).data
    half_size = 2**(num_qubits // 2)

    A, B, C, D = U[:half_size, :half_size], U[:half_size, half_size:], \
                 U[half_size:, :half_size], U[half_size:, half_size:]

    # Apply first unitary blocks
    qc.unitary(A, qubits[:num_qubits//2])
    qc.unitary(D, qubits[num_qubits//2:])

    # Apply an entangling gate (e.g., CNOT as a placeholder for the cosine-sine block)
    qc.cx(qubits[num_qubits//2 - 1], qubits[num_qubits//2])
    
    # Recursively apply QSD to sub-blocks
    qsd_recursive(qc, A, qubits[:num_qubits//2])
    qsd_recursive(qc, D, qubits[num_qubits//2:])

def matrix_to_circuit(matrix: np.matrix) -> QuantumCircuit:
    """Converts a unitary matrix to a quantum circuit using Clifford+T gates.

    Args:
        matrix: A unitary matrix.

    Returns:
        A QuantumCircuit object representing the matrix using Clifford+T gates.
    """
    # Validate the matrix is unitary
    if not np.allclose(matrix.getH() @ matrix, np.eye(matrix.shape[0])):
        raise ValueError("Input matrix is not unitary.")

    # Determine the number of qubits
    num_qubits = int(np.log2(matrix.shape[0]))

    # Step 1: Apply QSD decomposition before synthesis
    circuit = QuantumCircuit(num_qubits)
    qsd_recursive(circuit, matrix, list(range(num_qubits)))

    # Step 2: Convert to a unitary gate and decompose
    unitary_gate = UnitaryGate(matrix)
    circuit.append(unitary_gate, range(num_qubits))

    # Step 3: Define Clifford+T basis and synthesize
    clifford_t_basis = ['h', 's', 'sdg', 'x', 'y', 'z', 'cx', 't', 'tdg']
    unitary_synth = UnitarySynthesis(basis_gates=clifford_t_basis)
    pass_manager = passmanager(unitary_synth)
    synth_circuit = pass_manager.run(circuit)

    # Step 4: Decompose and transpile to Clifford+T gates
    decomposed_circuit = synth_circuit.decompose()
    clifford_t_circuit = transpile(decomposed_circuit, basis_gates=['h', 's', 'sdg', 'cx', 't'])

    return clifford_t_circuit