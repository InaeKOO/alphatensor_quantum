from qiskit import QuantumCircuit, transpile, passmanager
from qiskit.quantum_info import Operator, random_unitary
from qiskit.circuit.library import UnitaryGate
from qiskit.transpiler.passes.synthesis import UnitarySynthesis
from scipy.linalg import cossin
import matplotlib.pyplot as plt
import numpy as np

def zero_out_small_values(matrix, threshold=1e-10):
    """Set small values in a matrix to zero."""
    matrix_copy = np.real_if_close(matrix, tol=threshold)
    matrix_copy = matrix_copy * (np.abs(np.real(matrix_copy)) >= threshold)

    # Set values below the threshold to zero
    matrix_copy[np.abs(matrix_copy) < threshold] = 0

    return matrix_copy

def normalize_matrix(matrix):
    """Normalize a matrix to have a determinant of 1."""
    # Calculate the determinant
    det = np.linalg.det(matrix)
    
    # Scale the matrix to make the determinant 1
    scaling_factor = det**(-1/2)
    unitary_matrix = matrix * scaling_factor
    
    return unitary_matrix

def qsd_recursive(qc, U, qubits):
    """
    Recursively applies Quantum Shannon Decomposition (QSD) to break a unitary into smaller blocks.
    Uses Cosine-Sine Decomposition (CSD).
    """
    num_qubits = len(qubits)
    
    if num_qubits == 1:
        # Base case: Apply a single-qubit unitary directly
        qc.unitary(U, [qubits[0]])
        return

    N = U.shape[0]  # Full matrix size

    p = 2 ** (num_qubits // 2)
    q = N // p  # Ensuring q is also a power of 2

    # Perform Cosine-Sine Decomposition (CSD)
    A, D, theta = compute_csd(U, p, q)

    # Assign qubits correctly based on adjusted p and q
    num_qubits_A = int(np.log2(p))  # Number of qubits in A
    num_qubits_D = int(np.log2(q))  # Number of qubits in D

    qubits_A = qubits[:num_qubits_A]  # Select correct number of qubits for A
    qubits_D = qubits[num_qubits_A:]  # Remaining qubits for D

    # Apply first unitary blocks (A and D) to appropriate qubits
    qc.unitary(A, qubits_A, label="A")
    qc.unitary(D, qubits_D, label="D")

    # Apply controlled Y-rotations (entanglement)
    for i in range(len(theta)):
        if i < len(qubits_A) and i < len(qubits_D):
            control = qubits_A[-1 - i]
            target = qubits_D[i]
            qc.cry(2 * theta[i], control, target)

    # Recursively apply QSD to sub-blocks
    qsd_recursive(qc, A, qubits_A)
    qsd_recursive(qc, D, qubits_D)

def compute_csd(U, p, q):
    """
    Compute the Cosine-Sine Decomposition (CSD) of a unitary matrix U with p ≠ q.
    Ensures that A and D are always 2^n x 2^n unitary matrices.
    """
    U1, S, U2 = cossin(U, p=p, q=q)

    # Ensure A and D are proper unitary matrices with power-of-2 sizes
    A = U1[:p, :p]  # A should be (p x p)
    D = U2[:q, :q]  # D should be (q x q)

    # Extract theta values from the cosine-sine matrix S
    min_size = min(p, q)  # Use the smaller dimension to extract angles
    theta = np.arccos(np.diag(S[:min_size, :min_size]))

    return A, D, theta


def matrix_to_circuit(matrix: np.matrix) -> QuantumCircuit:
    """Converts a unitary matrix to a quantum circuit using Clifford+T gates.

    Args:
        matrix: A unitary matrix.

    Returns:
        A QuantumCircuit object representing the matrix using Clifford+T gates.
    """
    # Validate the matrix is unitary
    if not np.allclose(zero_out_small_values(matrix.getH() @ matrix), np.eye(matrix.shape[0])):
        raise ValueError("Input matrix is not unitary.")
    
    # Determine the number of qubits
    num_qubits = int(np.log2(matrix.shape[0]))

    # Step 1: Apply QSD decomposition before synthesis
    circuit = QuantumCircuit(num_qubits)
    qsd_recursive(circuit, matrix, list(range(num_qubits)))

    # Step 2: Define Clifford+T basis and synthesize
    clifford_t_basis = ['h', 's', 'u', 'cx', 't', 'cry']
    optimized_circuit = transpile(circuit, basis_gates=clifford_t_basis, optimization_level=2)

    return optimized_circuit

def process_fidelity(U, V):
    """Computes the process fidelity between two unitary matrices U and V."""
    d = U.shape[0]  # Get matrix dimension
    fidelity = np.abs(np.trace(np.dot(U.conj().T, V)))**2 / (d**2)
    return fidelity

def main():
    num_qubits = 8
    unitary_matrix = np.matrix(random_unitary(2 ** num_qubits).data)
    print("Unitary Matrix:")
    print(unitary_matrix)
    # Convert matrix to circuit
    qc = matrix_to_circuit(unitary_matrix)
    #print(circuit.draw(output='text'))
    fig = qc.draw(output='mpl')  # This returns a Matplotlib Figure
    plt.show()
    circuit_matrix = Operator(qc).data
    print("Circuit Matrix:")
    print(circuit_matrix)

    fidelity = process_fidelity(unitary_matrix, circuit_matrix)
    print(f"Process Fidelity: {fidelity:.6f}")

if __name__ == "__main__":
    main()