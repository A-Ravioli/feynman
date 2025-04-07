import numpy as np
import itertools
from typing import Dict, List, Any, Callable, Tuple, Optional
from scipy.sparse import diags, kron, identity, csr_matrix, eye
from scipy.sparse.linalg import eigs, expm_multiply # Keep expm_multiply for comparison or simpler cases if needed
from numpy.fft import fftn, ifftn, fftshift, ifftshift
import scipy.constants as const
import warnings

# Helper function for grid creation
def _create_grid(dims: int, points: List[int], ranges: List[Tuple[float, float]]) -> Tuple[List[np.ndarray], List[float], List[int]]:
    """Creates grid points and differentials for 1D, 2D, or 3D."""
    if not (1 <= dims <= 3):
        raise ValueError("Simulator currently supports 1, 2, or 3 dimensions.")
    if len(points) != dims or len(ranges) != dims:
        raise ValueError("Length of 'points' and 'ranges' must match 'dims'.")

    grid_coords = []
    deltas = []
    n_points = []
    for i in range(dims):
        coords, delta = np.linspace(ranges[i][0], ranges[i][1], points[i], retstep=True)
        grid_coords.append(coords)
        deltas.append(delta)
        n_points.append(points[i])
        
    # Create meshgrid for easy potential application
    # mesh = np.meshgrid(*grid_coords, indexing='ij') # 'ij' indexing is crucial

    return grid_coords, deltas, n_points

# Helper for kinetic operator (Finite Difference)
def _build_kinetic_operator_finite_diff(dims: int, n_points: List[int], deltas: List[float], mass: float, hbar: float) -> csr_matrix:
    """Builds the kinetic energy operator using finite differences."""
    laplacians = []
    total_points = np.prod(n_points)

    for i in range(dims):
        nx = n_points[i]
        dx = deltas[i]
        
        # 1D Laplacian
        diag_vals = np.ones(nx) * 2.0
        offdiag_vals = np.ones(nx - 1) * -1.0
        laplacian_1d = diags([offdiag_vals, diag_vals, offdiag_vals], [-1, 0, 1], shape=(nx, nx), format='csr')

        # Kronecker product to extend to N dimensions
        identities_before = [identity(n) for n in n_points[:i]]
        identities_after = [identity(n) for n in n_points[i+1:]]
        
        term_list = identities_before + [laplacian_1d] + identities_after
        
        laplacian_nd = term_list[0]
        for j in range(1, len(term_list)):
            laplacian_nd = kron(laplacian_nd, term_list[j], format='csr')

        laplacians.append((-hbar**2 / (2 * mass * dx**2)) * laplacian_nd)

    # Sum Laplacians for each dimension
    H_kinetic = sum(laplacians)
    return H_kinetic.asformat('csr')

# Helper for kinetic operator (Fourier Space) - Used in Split-Operator
def _build_kinetic_operator_fourier(dims: int, n_points: List[int], deltas: List[float], mass: float, hbar: float) -> np.ndarray:
    """Builds the kinetic energy operator in Fourier space."""
    momentum_grids = []
    for i in range(dims):
        nx = n_points[i]
        dx = deltas[i]
        # Frequencies for FFT:
        # From 0 to pi/dx, then -pi/dx back towards 0
        k_vals = 2 * np.pi * np.fft.fftfreq(nx, dx)
        momentum_grids.append(k_vals)

    # Create meshgrid of momentum components
    k_mesh = np.meshgrid(*momentum_grids, indexing='ij')

    # Calculate kinetic energy T = p^2 / 2m = (hbar*k)^2 / 2m
    k_squared = sum(k**2 for k in k_mesh)
    T_fourier = (hbar**2 * k_squared) / (2 * mass)
    return T_fourier # This is a diagonal operator in Fourier space (represented as an array)

class QuantumSimulator:
    """Simulator for quantum physics in 1D, 2D, or 3D."""
    
    def __init__(self):
        self.entities = {}
        self.interactions = [] # Interactions now mainly define potentials
        self.wavefunction_generators = {
            "gaussian": self._gaussian_wavepacket,
            # Add more as needed, making them dimension-aware
        }
        self.potential_functions = {
            "harmonic": self._harmonic_potential,
            "coulomb_reg": self._coulomb_potential_regularized, # Regularized
            "infinite_barrier": self._infinite_barrier,
            "double_slit_1d": self._double_slit_potential_1d, # Example 1D specific
            "uniform_field": self._uniform_field_potential,
            # Add more dimension-aware potentials
        }
        
        self.hbar = const.hbar
        self.e = const.e
        self.m_e = const.m_e # Default mass

    def simulate(self,
                 entities: Dict[str, Any],
                 interactions: List[Dict[str, Any]],
                 time_start: float, time_end: float, time_step: float,
                 simulation_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a quantum physics simulation.

        Args:
            entities: Dictionary of entities (name -> properties).
                      Must contain 'type': 'atom', 'properties': {...}.
                      Properties should include 'mass', 'initial_state',
                      potentially 'spin', 'relativistic_correction'.
                      'initial_state' must define 'wavefunction' (name or dict)
                      and provide info needed for grid setup ('dimensions', 'points', 'ranges').
            interactions: List of interactions, defining potentials acting on entities.
                          Interaction format: {'source': 'field_name', 'target': 'entity_name',
                                               'properties': {'potential': ...}}
            time_start: Start time.
            time_end: End time.
            time_step: Time step for evolution and output.
            simulation_params: Dictionary containing global simulation settings like
                               'solver_method' ('split_operator' or 'expm'),
                               'domain_settings': {'dimensions': int, 'points': list[int], 'ranges': list[tuple]}.
                               Optional: 'calculate_eigenstates': {'num_states': int}.

        Returns:
            Dictionary containing simulation results for each entity.
        """
        self.entities = entities # Store for potential lookups if needed
        self.interactions = interactions
        
        # Validate and extract global simulation parameters
        if 'domain_settings' not in simulation_params:
            raise ValueError("Missing 'domain_settings' in simulation_params.")
        domain = simulation_params['domain_settings']
        dims = domain.get('dimensions', None)
        points = domain.get('points', None)
        ranges = domain.get('ranges', None)
        
        if not all([dims, points, ranges]):
             raise ValueError("Missing 'dimensions', 'points', or 'ranges' in domain_settings.")
        if not (1 <= dims <= 3):
            raise ValueError("Simulator currently supports 1, 2, or 3 dimensions.")
        if len(points) != dims or len(ranges) != dims:
            raise ValueError("Length of 'points' and 'ranges' must match 'dimensions'.")

        solver_method = simulation_params.get('solver_method', 'split_operator').lower()
        if solver_method not in ['split_operator', 'expm']:
            warnings.warn(f"Unknown solver_method '{solver_method}', defaulting to 'split_operator'.")
            solver_method = 'split_operator'
        
        # Create Grid
        grid_coords, deltas, n_points = _create_grid(dims, points, ranges)
        grid_info = {'coords': grid_coords, 'deltas': deltas, 'n_points': n_points}
        total_points = np.prod(n_points)
        volume_element = np.prod(deltas)

        time_points = np.arange(time_start, time_end + time_step, time_step)
        num_steps = len(time_points)

        results = {"time_points": time_points, "entities": {}, "simulation_params": simulation_params}

        # --- Process Quantum Entities ---
        quantum_entities = {
            name: props for name, props in entities.items()
            if props.get("type") == "atom"
        }
        
        if not quantum_entities:
            warnings.warn("No quantum entities ('type': 'atom') found in the simulation.")
            return results

        # --- Simulate Each Quantum Entity (currently independent) ---
        # Future: Could handle interactions/entanglement here
        for name, entity in quantum_entities.items():
            print(f"Simulating quantum entity: {name}...")
            props = entity["properties"]
            mass = float(props.get("mass", self.m_e))
            spin = props.get("spin") # Placeholder for future use
            rel_correction = float(props.get("relativistic_correction", 1.0)) # Simple mass correction factor
            effective_mass = mass * rel_correction

            # --- Build Hamiltonian ---
            # Kinetic Part (Choose based on solver)
            if solver_method == 'split_operator':
                # Fourier space kinetic energy (diagonal array)
                T_op_fourier = _build_kinetic_operator_fourier(dims, n_points, deltas, effective_mass, self.hbar)
                # Real space kinetic operator (sparse matrix) - needed for eigenstates & energy calc
                T_op_real = _build_kinetic_operator_finite_diff(dims, n_points, deltas, effective_mass, self.hbar)
            else: # 'expm' uses real space only
                 T_op_real = _build_kinetic_operator_finite_diff(dims, n_points, deltas, effective_mass, self.hbar)

            # Potential Part (Real Space, Diagonal)
            V_potential = np.zeros(n_points) # shape depends on dims
            potential_applied = False
            for interaction in interactions:
                if interaction.get("target") == name and "potential" in interaction.get("properties", {}):
                    potential_info = interaction["properties"]["potential"]
                    V_potential += self._evaluate_potential(potential_info, grid_coords, dims)
                    potential_applied = True
            
            if not potential_applied:
                print(f"  Warning: No potential applied to entity '{name}'.")
                
            V_op_real = diags([V_potential.flatten()], [0], shape=(total_points, total_points), format='csr') # Sparse diagonal matrix

            # Full Hamiltonian (Real Space) - Needed for eigenstates & energy
            H_real = T_op_real + V_op_real

            # --- Initial State ---
            initial_state_info = props.get("initial_state")
            if not initial_state_info or "wavefunction" not in initial_state_info:
                 raise ValueError(f"Entity '{name}' missing 'initial_state' with 'wavefunction'.")
            
            psi_0_flat = self._generate_initial_wavefunction(
                initial_state_info["wavefunction"], grid_coords, dims
            ).flatten() # Flatten for sparse matrix operations

            # Normalize initial state
            norm_psi_0 = np.sqrt(np.sum(np.abs(psi_0_flat)**2) * volume_element)
            if norm_psi_0 < 1e-12:
                 raise ValueError(f"Initial wavefunction for '{name}' has near-zero norm.")
            psi_0_flat /= norm_psi_0
            
            psi_0 = psi_0_flat.reshape(n_points) # Reshape back for split-operator

            # --- Time Evolution ---
            print(f"  Evolving using {solver_method} method...")
            psi_t_flat = np.zeros((num_steps, total_points), dtype=complex)
            psi_t_flat[0, :] = psi_0_flat
            
            if solver_method == 'split_operator':
                # Pre-calculate evolution operators for efficiency
                exp_V = np.exp(-1j * V_potential * time_step / (2 * self.hbar)) # Position space V/2
                exp_T = np.exp(-1j * T_op_fourier * time_step / self.hbar)      # Momentum space T
                
                current_psi = psi_0.copy() # Work with multi-dimensional array
                for i in range(1, num_steps):
                    # Split-Operator Step: V/2 -> T -> V/2
                    current_psi *= exp_V                                  # Apply V/2
                    psi_k = fftshift(fftn(current_psi))                   # FFT to momentum space
                    psi_k *= exp_T                                        # Apply T
                    current_psi = ifftn(ifftshift(psi_k))                 # IFFT back to position space
                    current_psi *= exp_V                                  # Apply V/2
                    
                    # Optional normalization check/enforcement (numerical drift)
                    # norm = np.sqrt(np.sum(np.abs(current_psi)**2) * volume_element)
                    # current_psi /= norm
                    
                    psi_t_flat[i, :] = current_psi.flatten()

            elif solver_method == 'expm':
                # Use matrix exponential (can be slow for large grids)
                evolution_op_sparse = -1j * H_real * time_step / self.hbar
                current_psi_flat = psi_0_flat.copy()
                for i in range(1, num_steps):
                    # expm_multiply is efficient for sparse matrix * vector
                    current_psi_flat = expm_multiply(evolution_op_sparse, current_psi_flat)
                    # Optional normalization
                    # norm = np.linalg.norm(current_psi_flat) * np.sqrt(volume_element)
                    # current_psi_flat /= norm
                    psi_t_flat[i, :] = current_psi_flat

            # --- Calculate Observables ---
            print("  Calculating observables...")
            prob_density_flat = np.abs(psi_t_flat)**2
            
            # Reshape for easier calculation where needed
            # psi_t = psi_t_flat.reshape((num_steps,) + tuple(n_points))
            # prob_density = prob_density_flat.reshape((num_steps,) + tuple(n_points))

            # <Position>
            expected_position = np.zeros((num_steps, dims))
            for d in range(dims):
                 # Create a grid representing the d-th coordinate values broadcasted to the full grid shape
                 coord_grid = np.meshgrid(*grid_coords, indexing='ij')[d]
                 expected_position[:, d] = np.sum(coord_grid.flatten() * prob_density_flat, axis=1) * volume_element

            # <Momentum> (using Fourier transform method for simplicity)
            # p = -iħ∇
            # <p> = ∫ ψ* (-iħ∇) ψ dV
            expected_momentum = np.zeros((num_steps, dims))
            k_grids_shifted = [fftshift(2 * np.pi * np.fft.fftfreq(n, d)) for n, d in zip(n_points, deltas)]
            k_mesh_shifted = np.meshgrid(*k_grids_shifted, indexing='ij')

            for i in range(num_steps):
                 psi_current_flat = psi_t_flat[i, :]
                 psi_current = psi_current_flat.reshape(n_points)
                 psi_k = fftshift(fftn(psi_current)) # Get momentum space wavefunction
                 prob_density_k = np.abs(psi_k)**2
                 norm_k = np.sum(prob_density_k) # Normalization in k-space (Parseval's theorem)

                 if norm_k > 1e-12 :
                     for d in range(dims):
                         expected_momentum[i, d] = self.hbar * np.sum(k_mesh_shifted[d] * prob_density_k) / norm_k


            # <Energy> = <H> = <T> + <V>
            # <V> = ∫ V(r) |ψ(r)|² dV
            expected_potential_energy = np.sum(V_potential.flatten() * prob_density_flat, axis=1) * volume_element
            # <T> = ∫ ψ* (T_op) ψ dV
            # Applying sparse T_op_real to each time step's state vector
            expected_kinetic_energy = np.zeros(num_steps)
            for i in range(num_steps):
                psi_vector = psi_t_flat[i, :]
                T_psi = T_op_real.dot(psi_vector)
                expected_kinetic_energy[i] = np.real(np.vdot(psi_vector, T_psi)) * volume_element # ψ* T ψ
            
            expected_energy = expected_kinetic_energy + expected_potential_energy

            # --- Store Results for Entity ---
            results["entities"][name] = {
                "wavefunction_flat": psi_t_flat, # Store flat version for consistency
                "probability_density_flat": prob_density_flat,
                "expected_position": expected_position,
                "expected_momentum": expected_momentum,
                "expected_kinetic_energy": expected_kinetic_energy,
                "expected_potential_energy": expected_potential_energy,
                "expected_energy": expected_energy,
                "grid_info": grid_info,
                "dimensions": dims,
                "type": "atom",
                "mass": mass,
                "spin": spin, # Include placeholder info
            }

            # --- Optional: Calculate Eigenstates ---
            eigenstate_calc_info = simulation_params.get('calculate_eigenstates')
            if eigenstate_calc_info and isinstance(eigenstate_calc_info, dict):
                num_states = eigenstate_calc_info.get('num_states', 5)
                print(f"  Calculating {num_states} lowest eigenstates...")
                try:
                    # Use shift-invert mode for better convergence for lowest eigenvalues
                    # 'sigma=0' finds eigenvalues near 0. Adjust if needed (e.g., ground state energy estimate)
                    eigenvalues, eigenvectors_flat = eigs(H_real, k=num_states, which='SR', sigma=None) # 'SR' = Smallest Real part
                    
                    # Sort by real part of eigenvalue
                    sort_indices = np.argsort(np.real(eigenvalues))
                    eigenvalues = eigenvalues[sort_indices]
                    eigenvectors_flat = eigenvectors_flat[:, sort_indices]
                    
                    # Normalize eigenvectors
                    eigenvectors_normalized_flat = np.zeros_like(eigenvectors_flat)
                    for j in range(num_states):
                         vec = eigenvectors_flat[:, j]
                         norm = np.sqrt(np.sum(np.abs(vec)**2) * volume_element)
                         eigenvectors_normalized_flat[:, j] = vec / norm

                    results["entities"][name]["eigenstates"] = {
                        "eigenvalues": np.real(eigenvalues), # Usually real for Hermitian H
                        "eigenvectors_flat": eigenvectors_normalized_flat
                    }
                    print(f"  Calculated eigenvalues: {np.real(eigenvalues)}")
                except Exception as e:
                    warnings.warn(f"Eigenstate calculation failed for entity '{name}': {e}")

        
        return results
    
    # --- Potential Function Evaluation ---
    def _evaluate_potential(self, potential_info: Any, grid_coords: List[np.ndarray], dims: int) -> np.ndarray:
        """Evaluates a potential function on the grid."""
        V = np.zeros([len(gc) for gc in grid_coords]) # Initialize potential grid
        mesh = np.meshgrid(*grid_coords, indexing='ij') # Create meshgrid once

        if isinstance(potential_info, str):
            pot_name = potential_info
            pot_params = {}
        elif isinstance(potential_info, dict) and "function" in potential_info:
            pot_name = potential_info["function"]
            pot_params = self._parse_params(potential_info.get("args", []))
        else:
            warnings.warn(f"Invalid potential format: {potential_info}. Skipping.")
            return V

        if pot_name in self.potential_functions:
            pot_func = self.potential_functions[pot_name]
            try:
                # Pass meshgrid components directly
                V = pot_func(pos=mesh, dims=dims, **pot_params)
                if V.shape != tuple(len(gc) for gc in grid_coords):
                     raise ValueError(f"Potential function '{pot_name}' returned incorrect shape.")
            except Exception as e:
                warnings.warn(f"Error evaluating potential '{pot_name}' for {dims}D: {e}. Skipping.")
                return np.zeros([len(gc) for gc in grid_coords])
        else:
            warnings.warn(f"Potential function '{pot_name}' not found. Skipping.")
            
        return V

    # --- Initial Wavefunction Generation ---
    def _generate_initial_wavefunction(self, wf_info: Any, grid_coords: List[np.ndarray], dims: int) -> np.ndarray:
        """Generates the initial wavefunction on the grid."""
        psi_0 = np.zeros([len(gc) for gc in grid_coords], dtype=complex)
        mesh = np.meshgrid(*grid_coords, indexing='ij') # Create meshgrid once

        if isinstance(wf_info, str):
            wf_name = wf_info
            wf_params = {}
        elif isinstance(wf_info, dict) and "function" in wf_info:
            wf_name = wf_info["function"]
            wf_params = self._parse_params(wf_info.get("args", []))
        else:
            warnings.warn(f"Invalid wavefunction format: {wf_info}. Defaulting to Gaussian.")
            wf_name = "gaussian"
            wf_params = {} # Use defaults of gaussian

        if wf_name in self.wavefunction_generators:
            wf_gen = self.wavefunction_generators[wf_name]
            try:
                 psi_0 = wf_gen(pos=mesh, dims=dims, **wf_params)
                 if psi_0.shape != tuple(len(gc) for gc in grid_coords):
                     raise ValueError(f"Wavefunction generator '{wf_name}' returned incorrect shape.")
            except Exception as e:
                warnings.warn(f"Error generating wavefunction '{wf_name}' for {dims}D: {e}. Defaulting to Gaussian.")
                psi_0 = self._gaussian_wavepacket(pos=mesh, dims=dims) # Use defaults
        else:
             warnings.warn(f"Wavefunction generator '{wf_name}' not found. Defaulting to Gaussian.")
             psi_0 = self._gaussian_wavepacket(pos=mesh, dims=dims) # Use defaults
             
        return psi_0


    def _parse_params(self, args: List[Any]) -> Dict[str, Any]:
        """Helper to parse function arguments from DSL format."""
        params = {}
        positional_idx = 0
        for arg in args:
            if isinstance(arg, dict) and "param" in arg and "value" in arg:
                params[arg["param"]] = arg["value"]
            else:
                # Assume positional if not dict or missing keys
                params[f"_pos{positional_idx}"] = arg
                positional_idx += 1
        return params

    # --- Example Wavefunction Generators (Dimension-Aware) ---
    def _gaussian_wavepacket(self, pos: List[np.ndarray], dims: int, center: Optional[List[float]] = None, spread: Optional[List[float]] = None, k: Optional[List[float]] = None):
        """Gaussian wavepacket N-D: ψ(r) ∝ exp(-Σ(xi-x₀i)²/4σi²) exp(i k·r)"""
        if center is None: center = [0.0] * dims
        if spread is None: spread = [1.0] * dims
        if k is None: k = [0.0] * dims
        if len(center) != dims or len(spread) != dims or len(k) != dims:
             raise ValueError("Length of 'center', 'spread', and 'k' must match 'dims'.")

        exponent_gauss = sum(-((pos[d] - center[d])**2) / (4 * spread[d]**2) for d in range(dims))
        exponent_phase = sum(k[d] * pos[d] for d in range(dims))
        
        psi = np.exp(exponent_gauss + 1j * exponent_phase)
        
        # Normalization (analytical for Gaussian)
        norm_factor = np.prod([(2 * np.pi * s**2)**(-0.25) for s in spread])
        return norm_factor * psi

    # --- Example Potential Functions (Dimension-Aware) ---
    def _harmonic_potential(self, pos: List[np.ndarray], dims: int, k: Optional[List[float]] = None, center: Optional[List[float]] = None):
        """Harmonic potential N-D: V(r) = 1/2 Σ ki (xi-x₀i)²"""
        if k is None: k = [1.0] * dims
        if center is None: center = [0.0] * dims
        if len(k) != dims or len(center) != dims:
            raise ValueError("Length of 'k' and 'center' must match 'dims'.")
            
        return 0.5 * sum(k[d] * (pos[d] - center[d])**2 for d in range(dims))

    def _coulomb_potential_regularized(self, pos: List[np.ndarray], dims: int, Q=1.0, q=1.0, center: Optional[List[float]] = None, epsilon=1e-6):
        """Regularized Coulomb potential N-D: V(r) = k Q q / sqrt(|r-r₀|² + ε²)"""
        if center is None: center = [0.0] * dims
        if len(center) != dims:
             raise ValueError("Length of 'center' must match 'dims'.")
             
        k_coulomb = 1 / (4 * np.pi * const.epsilon_0) # Use scipy constants
        r_squared = sum((pos[d] - center[d])**2 for d in range(dims))
        return k_coulomb * Q * q / np.sqrt(r_squared + epsilon**2)

    def _infinite_barrier(self, pos: List[np.ndarray], dims: int, region_ranges: Optional[List[Tuple[float, float]]] = None, potential_height=1e6):
        """Infinite potential barrier outside specified region (N-D)."""
        if region_ranges is None: region_ranges = [(-1.0, 1.0)] * dims # Default box [-1, 1] in all dims
        if len(region_ranges) != dims:
             raise ValueError("Length of 'region_ranges' must match 'dims'.")
             
        condition = np.ones_like(pos[0], dtype=bool)
        for d in range(dims):
            condition &= (pos[d] >= region_ranges[d][0]) & (pos[d] <= region_ranges[d][1])
            
        potential = np.where(condition, 0.0, potential_height)
        return potential

    def _double_slit_potential_1d(self, pos: List[np.ndarray], dims: int, barrier_pos=0.0, barrier_thickness=0.1, slit_width=0.2, slit_separation=1.0, potential_height=1e6):
        """Double slit potential barrier (1D only)."""
        if dims != 1:
            warnings.warn("Double slit potential currently only implemented for 1D.")
            return np.zeros_like(pos[0])
            
        x = pos[0]
        potential = np.zeros_like(x)
        
        # Define barrier region
        barrier_mask = (x >= barrier_pos - barrier_thickness / 2) & (x <= barrier_pos + barrier_thickness / 2)
        
        # Define slit positions within the barrier
        slit1_center = -slit_separation / 2
        slit2_center = slit_separation / 2
        in_slit1 = (x >= slit1_center - slit_width / 2) & (x <= slit1_center + slit_width / 2)
        in_slit2 = (x >= slit2_center - slit_width / 2) & (x <= slit2_center + slit_width / 2)
        
        # Apply potential only within the barrier region, excluding slits
        potential[barrier_mask & ~in_slit1 & ~in_slit2] = potential_height
        
        return potential

    def _uniform_field_potential(self, pos: List[np.ndarray], dims: int, field_vector: Optional[List[float]] = None, charge=1.0):
        """Potential from a uniform electric field E: V = -q E · r"""
        if field_vector is None: field_vector = [1.0] + [0.0]*(dims-1) # Default E along x
        if len(field_vector) != dims:
             raise ValueError("Length of 'field_vector' must match 'dims'.")
             
        potential = -charge * sum(field_vector[d] * pos[d] for d in range(dims))
        return potential

# Example usage placeholder (remove or move to examples/tests)
# if __name__ == '__main__':
#     sim = QuantumSimulator()
#     # Define entities, interactions, params...
#     # results = sim.simulate(...)
#     # Process results...

# Example usage placeholder (remove or move to examples/tests)
# if __name__ == '__main__':
#     sim = QuantumSimulator()
#     # Define entities, interactions, params...
#     # results = sim.simulate(...)
#     # Process results... 