import numpy as np
from typing import Dict, List, Any, Callable
from scipy.integrate import solve_ivp

class ClassicalSimulator:
    """Simulator for classical physics"""
    
    def __init__(self):
        self.entities = {}
        self.interactions = []
        self.force_functions = {
            "coulomb": self._coulomb_force,
            "gravity": self._gravity_force,
            "spring": self._spring_force,
            "constant": self._constant_force,
        }
        self.potential_functions = {
            "harmonic": self._harmonic_potential,
            "gravitational": self._gravitational_potential,
            "coulomb": self._coulomb_potential,
            "infinite_barrier": self._infinite_barrier,
            "infinite_barrier_except": self._infinite_barrier_except,
        }
    
    def simulate(self, entities: Dict[str, Any], interactions: List[Dict[str, Any]], 
                 time_start: float, time_end: float, time_step: float) -> Dict[str, Any]:
        """Run a classical physics simulation"""
        self.entities = entities
        self.interactions = interactions
        
        # Filter to get just the classical objects
        classical_objects = {
            name: props for name, props in entities.items()
            if props["type"] == "object"
        }
        
        if not classical_objects:
            return {
                "time_points": np.arange(time_start, time_end + time_step, time_step),
                "entities": {}
            }
        
        # Set up initial state vector for all objects
        # [x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2, ...]
        initial_state = []
        object_names = []
        
        for name, entity in classical_objects.items():
            object_names.append(name)
            props = entity["properties"]
            pos = props.get("position", [0, 0, 0])
            vel = props.get("velocity", [0, 0, 0])
            initial_state.extend(pos)
            initial_state.extend(vel)
        
        initial_state = np.array(initial_state, dtype=float)
        
        # Set up ODE solver
        time_points = np.arange(time_start, time_end + time_step, time_step)
        
        # Solve the equations of motion
        solution = solve_ivp(
            self._compute_derivatives,
            [time_start, time_end],
            initial_state,
            t_eval=time_points,
            method="RK45"
        )
        
        # Extract results
        results = {"time_points": time_points, "entities": {}}
        
        for i, name in enumerate(object_names):
            base_idx = i * 6  # Each object has 6 values (3 position, 3 velocity)
            
            # Extract position and velocity trajectories
            positions = solution.y[base_idx:base_idx+3, :]
            velocities = solution.y[base_idx+3:base_idx+6, :]
            
            # Calculate kinetic energy
            mass = float(classical_objects[name]["properties"].get("mass", 1.0))
            kinetic_energy = 0.5 * mass * np.sum(velocities**2, axis=0)
            
            # Store results
            results["entities"][name] = {
                "positions": positions.T,  # Transpose to get [timestep, coordinate]
                "velocities": velocities.T,
                "kinetic_energy": kinetic_energy,
                "type": "object"
            }
        
        return results
    
    def _compute_derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute time derivatives for all objects (equations of motion)"""
        num_objects = len(state) // 6  # Each object has 6 values
        derivatives = np.zeros_like(state)
        
        # First, extract all positions and velocities
        positions = {}
        velocities = {}
        masses = {}
        
        object_names = list(filter(lambda n: self.entities[n]["type"] == "object", self.entities.keys()))
        
        for i, name in enumerate(object_names[:num_objects]):
            base_idx = i * 6
            positions[name] = state[base_idx:base_idx+3]
            velocities[name] = state[base_idx+3:base_idx+6]
            masses[name] = float(self.entities[name]["properties"].get("mass", 1.0))
        
        # For each object, set velocity derivatives (accelerations)
        for i, name in enumerate(object_names[:num_objects]):
            base_idx = i * 6
            
            # Position derivatives = velocities
            derivatives[base_idx:base_idx+3] = velocities[name]
            
            # Calculate forces on this object from all interactions
            net_force = np.zeros(3)
            
            for interaction in self.interactions:
                source = interaction["source"]
                target = interaction["target"]
                props = interaction["properties"]
                
                # Skip non-force interactions
                if "force" not in props:
                    continue
                
                # Skip interactions not involving this object
                if source != name and target != name:
                    continue
                
                # Determine the other object in the interaction
                other = target if source == name else source
                if other not in positions:
                    continue
                
                # Calculate the force
                force_info = props["force"]
                if isinstance(force_info, str):
                    # Handle simple force name
                    force_name = force_info
                    force_params = {}
                elif isinstance(force_info, dict) and "function" in force_info:
                    # Handle function call with parameters
                    force_name = force_info["function"]
                    force_params = {}
                    for arg in force_info.get("args", []):
                        if isinstance(arg, dict) and "param" in arg and "value" in arg:
                            force_params[arg["param"]] = arg["value"]
                else:
                    continue
                
                if force_name in self.force_functions:
                    force_func = self.force_functions[force_name]
                    force = force_func(
                        pos1=positions[name],
                        pos2=positions[other],
                        mass1=masses[name],
                        mass2=masses[other],
                        **force_params
                    )
                    
                    # If this object is the target, the force acts on it directly
                    # If it's the source, reverse the force (Newton's third law)
                    if name == target:
                        net_force += force
                    else:
                        net_force -= force
            
            # F = ma => a = F/m
            derivatives[base_idx+3:base_idx+6] = net_force / masses[name]
        
        return derivatives
    
    # Force functions
    def _coulomb_force(self, pos1, pos2, mass1=1.0, mass2=1.0, k=8.99e9, q1=1.0, q2=1.0):
        """Calculate Coulomb force between two charged particles"""
        r_vec = pos2 - pos1
        r = np.linalg.norm(r_vec)
        if r < 1e-10:  # Avoid division by zero
            return np.zeros(3)
        r_hat = r_vec / r
        force_magnitude = k * q1 * q2 / (r * r)
        return force_magnitude * r_hat
    
    def _gravity_force(self, pos1, pos2, mass1=1.0, mass2=1.0, G=6.67430e-11):
        """Calculate gravitational force between two masses"""
        r_vec = pos2 - pos1
        r = np.linalg.norm(r_vec)
        if r < 1e-10:  # Avoid division by zero
            return np.zeros(3)
        r_hat = r_vec / r
        force_magnitude = G * mass1 * mass2 / (r * r)
        return force_magnitude * r_hat
    
    def _spring_force(self, pos1, pos2, mass1=1.0, mass2=1.0, k=1.0, rest_length=0.0):
        """Calculate spring force between two objects"""
        r_vec = pos2 - pos1
        r = np.linalg.norm(r_vec)
        if r < 1e-10:  # Avoid division by zero
            return np.zeros(3)
        r_hat = r_vec / r
        force_magnitude = k * (r - rest_length)
        return force_magnitude * r_hat
    
    def _constant_force(self, pos1, pos2, mass1=1.0, mass2=1.0, value=[0, 0, 0]):
        """Apply a constant force"""
        return np.array(value)
    
    # Potential functions
    def _harmonic_potential(self, pos, k=1.0, center=[0, 0, 0]):
        """Harmonic (spring) potential: V = 1/2 k |r-r0|^2"""
        center = np.array(center)
        r_vec = pos - center
        return 0.5 * k * np.sum(r_vec**2)
    
    def _gravitational_potential(self, pos, mass=1.0, M=1.0, G=6.67430e-11, center=[0, 0, 0]):
        """Gravitational potential: V = -G M m / r"""
        center = np.array(center)
        r_vec = pos - center
        r = np.linalg.norm(r_vec)
        if r < 1e-10:
            return -1e10  # Large negative value for very small r
        return -G * M * mass / r
    
    def _coulomb_potential(self, pos, q=1.0, Q=1.0, k=8.99e9, center=[0, 0, 0]):
        """Coulomb potential: V = k Q q / r"""
        center = np.array(center)
        r_vec = pos - center
        r = np.linalg.norm(r_vec)
        if r < 1e-10:
            return 1e10 if q * Q > 0 else -1e10  # Large value for very small r
        return k * Q * q / r
    
    def _infinite_barrier(self, pos, region=[[-1, 1], [-1, 1], [-1, 1]]):
        """Infinite potential barrier outside specified region"""
        x, y, z = pos
        x_range, y_range, z_range = region
        
        if (x_range[0] <= x <= x_range[1] and
            y_range[0] <= y <= y_range[1] and
            z_range[0] <= z <= z_range[1]):
            return 0.0
        return float('inf')
    
    def _infinite_barrier_except(self, pos, holes=[]):
        """Infinite potential barrier except at specified holes"""
        for hole in holes:
            # Check if position is within any hole
            if all(hole[0][i] <= pos[i] <= hole[1][i] for i in range(min(len(hole[0]), len(pos)))):
                return 0.0
        return float('inf') 