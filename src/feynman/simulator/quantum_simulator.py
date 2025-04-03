import numpy as np
from typing import Dict, List, Any, Callable
from scipy.sparse import diags, lil_matrix, eye
from scipy.sparse.linalg import expm_multiply
import scipy.constants as const

class QuantumSimulator:
    """Simulator for quantum physics"""
    
    def __init__(self):
        self.entities = {}
        self.interactions = []
        self.wavefunction_generators = {
            "gaussian": self._gaussian_wavefunction,
            "hydrogen_ground_state": self._hydrogen_ground_state,
            "plane_wave": self._plane_wave,
            "harmonic_oscillator": self._harmonic_oscillator_state,
        }
        self.potential_functions = {
            "harmonic": self._harmonic_potential,
            "coulomb": self._coulomb_potential,
            "infinite_barrier": self._infinite_barrier,
            "infinite_barrier_except": self._infinite_barrier_except,
            "double_slit": self._double_slit_potential,
        }
        
        # Physical constants (in SI units)
        self.hbar = const.hbar
        self.e = const.e  # electron charge
        self.m_e = const.m_e  # electron mass
    
    def simulate(self, entities: Dict[str, Any], interactions: List[Dict[str, Any]], 
                 time_start: float, time_end: float, time_step: float) -> Dict[str, Any]:
        """Run a quantum physics simulation"""
        self.entities = entities
        self.interactions = interactions
        
        # Filter to get just the quantum atoms/particles
        quantum_entities = {
            name: props for name, props in entities.items()
            if props["type"] == "atom"
        }
        
        if not quantum_entities:
            return {
                "time_points": np.arange(time_start, time_end + time_step, time_step),
                "entities": {}
            }
        
        # For simplicity, we'll simulate 1D quantum systems
        # More sophisticated implementations could use 2D or 3D grids
        
        # Set up spatial grid
        x_min, x_max = -10, 10  # default domain
        nx = 1000  # number of grid points
        x = np.linspace(x_min, x_max, nx)
        dx = x[1] - x[0]
        
        # Time points for simulation
        time_points = np.arange(time_start, time_end + time_step, time_step)
        
        # Results dictionary
        results = {"time_points": time_points, "entities": {}}
        
        # Simulate each quantum entity independently
        # (In a more sophisticated version, we would handle entanglement)
        for name, entity in quantum_entities.items():
            props = entity["properties"]
            mass = float(props.get("mass", self.m_e))  # Default to electron mass
            
            # Get initial state
            initial_state_info = props.get("initial_state", {})
            if "wavefunction" in initial_state_info:
                wf_info = initial_state_info["wavefunction"]
                if isinstance(wf_info, str):
                    # Simple wavefunction name
                    wf_name = wf_info
                    wf_params = {}
                elif isinstance(wf_info, dict) and "function" in wf_info:
                    # Function call with parameters
                    wf_name = wf_info["function"]
                    wf_params = {}
                    for arg in wf_info.get("args", []):
                        if isinstance(arg, dict) and arg.get("param") and arg.get("value"):
                            wf_params[arg["param"]] = arg["value"]
                        else:
                            # Handle positional args
                            wf_params[len(wf_params)] = arg
                else:
                    wf_name = "gaussian"  # Default
                    wf_params = {}
                
                # Generate initial wavefunction
                if wf_name in self.wavefunction_generators:
                    wf_gen = self.wavefunction_generators[wf_name]
                    psi_0 = wf_gen(x=x, **wf_params)
                else:
                    # Default to Gaussian
                    psi_0 = self._gaussian_wavefunction(x=x)
            else:
                # Default initial state is a Gaussian wavepacket
                psi_0 = self._gaussian_wavefunction(x=x)
            
            # Set up Hamiltonian
            # Kinetic energy operator: -ħ²/2m * d²/dx²
            # We use the finite difference approximation
            h_diag = np.ones(nx) * (2 * self.hbar**2 / (mass * dx**2))
            h_off_diag = np.ones(nx-1) * (-self.hbar**2 / (mass * dx**2))
            H_kinetic = diags([h_off_diag, h_diag, h_off_diag], [-1, 0, 1])
            
            # Potential energy operator: V(x)
            V = np.zeros(nx)
            
            # Apply potentials from interactions
            for interaction in self.interactions:
                if interaction["target"] != name:
                    continue
                
                props = interaction["properties"]
                if "potential" not in props:
                    continue
                
                pot_info = props["potential"]
                pot_name = ""
                pot_params = {}
                
                if isinstance(pot_info, str):
                    pot_name = pot_info
                elif isinstance(pot_info, dict) and "function" in pot_info:
                    pot_name = pot_info["function"]
                    for arg in pot_info.get("args", []):
                        if isinstance(arg, dict) and arg.get("param") and arg.get("value"):
                            pot_params[arg["param"]] = arg["value"]
                
                if pot_name in self.potential_functions:
                    pot_func = self.potential_functions[pot_name]
                    for i, xi in enumerate(x):
                        V[i] += pot_func(pos=xi, **pot_params)
            
            # Full Hamiltonian
            H = H_kinetic + diags([V], [0])
            
            # Time evolution
            # For small systems, we could use the full matrix exponential
            # For larger systems, split-operator or Crank-Nicolson method would be better
            
            # Convert to sparse matrix
            H_sparse = lil_matrix(H)
            
            # Time evolution operator: exp(-i*H*dt/ħ)
            dt = time_step
            evolution_op = -1j * H_sparse * dt / self.hbar
            
            # Evolve the system
            psi_t = np.zeros((len(time_points), nx), dtype=complex)
            psi_t[0] = psi_0
            
            for i in range(1, len(time_points)):
                # Apply the time evolution operator
                psi_t[i] = expm_multiply(evolution_op, psi_t[i-1])
                
                # Normalize (optional, as the evolution should preserve norm)
                norm = np.sqrt(np.sum(np.abs(psi_t[i])**2) * dx)
                psi_t[i] /= norm
            
            # Calculate observables
            # Probability density: |ψ|²
            prob_density = np.abs(psi_t)**2
            
            # Expected position: <x> = ∫ x |ψ|² dx
            expected_position = np.zeros(len(time_points))
            for i in range(len(time_points)):
                expected_position[i] = np.sum(x * prob_density[i]) * dx
            
            # Store results
            results["entities"][name] = {
                "wavefunction": psi_t,
                "probability_density": prob_density,
                "expected_position": expected_position,
                "grid": x,
                "type": "atom"
            }
        
        return results
    
    # Wavefunction generators
    def _gaussian_wavefunction(self, x, center=0.0, spread=1.0, k=0.0):
        """Gaussian wavepacket: ψ(x) ∝ exp(-(x-x₀)²/4σ²) exp(ikx)"""
        normalization = (2 * np.pi * spread**2)**(-0.25)
        gaussian = np.exp(-(x - center)**2 / (4 * spread**2))
        phase = np.exp(1j * k * x)
        return normalization * gaussian * phase
    
    def _hydrogen_ground_state(self, x, a0=1.0):
        """1D analogue of hydrogen ground state: ψ(x) ∝ exp(-|x|/a0)"""
        normalization = 1.0 / np.sqrt(2 * a0)
        wavefunction = np.exp(-np.abs(x) / a0)
        return normalization * wavefunction
    
    def _plane_wave(self, x, k=1.0):
        """Plane wave: ψ(x) ∝ exp(ikx)"""
        normalization = 1.0 / np.sqrt(x[-1] - x[0])
        return normalization * np.exp(1j * k * x)
    
    def _harmonic_oscillator_state(self, x, n=0, omega=1.0, m=1.0):
        """Harmonic oscillator eigenstate: ψₙ(x)"""
        # For simplicity, we'll just implement n=0, n=1 states
        # A more complete implementation would use Hermite polynomials
        alpha = np.sqrt(m * omega / self.hbar)
        if n == 0:
            normalization = (alpha / np.pi)**0.25
            return normalization * np.exp(-alpha * x**2 / 2)
        elif n == 1:
            normalization = (alpha / np.pi)**0.25 * np.sqrt(2 * alpha)
            return normalization * x * np.exp(-alpha * x**2 / 2)
        else:
            # Default to ground state if n > 1
            normalization = (alpha / np.pi)**0.25
            return normalization * np.exp(-alpha * x**2 / 2)
    
    # Potential functions
    def _harmonic_potential(self, pos, k=1.0, center=0.0):
        """Harmonic oscillator potential: V(x) = 1/2 k (x-x₀)²"""
        return 0.5 * k * (pos - center)**2
    
    def _coulomb_potential(self, pos, q1=1.0, q2=1.0, k=8.99e9, epsilon=1e-5):
        """Coulomb potential with regularization: V(x) = k q1 q2 / |x|"""
        # Regularized to avoid singularity at x=0
        return k * q1 * q2 / (np.abs(pos) + epsilon)
    
    def _infinite_barrier(self, pos, region=[-1, 1]):
        """Infinite potential barrier outside specified region"""
        if region[0] <= pos <= region[1]:
            return 0.0
        return 1e6  # Large but finite for numerical stability
    
    def _infinite_barrier_except(self, pos, holes=[]):
        """Infinite potential barrier except at specified holes"""
        for hole in holes:
            if len(hole) >= 2 and hole[0] <= pos <= hole[1]:
                return 0.0
        return 1e6  # Large but finite for numerical stability
    
    def _double_slit_potential(self, pos, position=0.0, slit_width=0.2, slit_separation=1.0):
        """Double slit potential barrier"""
        # Barrier at position with two slits
        if abs(pos - position) < 1e-6:  # At the barrier position
            # Check if within either slit
            slit1_center = -slit_separation / 2
            slit2_center = slit_separation / 2
            
            if (abs(pos - slit1_center) < slit_width / 2 or 
                abs(pos - slit2_center) < slit_width / 2):
                return 0.0  # Inside a slit
            else:
                return 1e6  # At barrier but not in a slit
        return 0.0  # Not at barrier position 