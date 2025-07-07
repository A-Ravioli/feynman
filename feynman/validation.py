"""
Input validation and error handling for Feynman physics simulations.
Provides comprehensive validation of DSL inputs to prevent nonsensical physics.
"""
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import warnings


class ValidationError(Exception):
    """Raised when physics validation fails"""
    pass


class PhysicsValidator:
    """Validates physics parameters and catches common errors"""
    
    # Physical constants and reasonable ranges
    PHYSICAL_CONSTANTS = {
        'c': 299792458,  # speed of light m/s
        'h': 6.62607015e-34,  # Planck constant J⋅s
        'hbar': 1.054571817e-34,  # reduced Planck constant J⋅s
        'e': 1.602176634e-19,  # elementary charge C
        'm_e': 9.1093837015e-31,  # electron mass kg
        'm_p': 1.67262192369e-27,  # proton mass kg
        'k_B': 1.380649e-23,  # Boltzmann constant J/K
        'G': 6.67430e-11,  # gravitational constant m³/kg⋅s²
    }
    
    # Reasonable parameter ranges
    RANGES = {
        'mass': (1e-35, 1e50),  # kg - from elementary particles to supermassive black holes
        'position': (-1e20, 1e20),  # m - from subatomic to cosmic scales  
        'velocity': (0, 0.9 * PHYSICAL_CONSTANTS['c']),  # m/s - up to 90% light speed
        'energy': (-1e10, 1e10),  # J - reasonable energy range
        'time': (0, 1e20),  # s - from zero to age of universe
        'length': (-1e20, 1e20),  # m - from subatomic to cosmic, including negative coords
        'charge': (-1e10 * PHYSICAL_CONSTANTS['e'], 1e10 * PHYSICAL_CONSTANTS['e']),  # C
    }
    
    def __init__(self):
        self.warnings = []
        self.errors = []
    
    def validate_model(self, model_props: Dict[str, Any]) -> bool:
        """Validate model parameters"""
        try:
            model_type = model_props.get('type', 'classical')
            if model_type not in ['classical', 'quantum']:
                raise ValidationError(f"Invalid model type: {model_type}. Must be 'classical' or 'quantum'")
            
            # Validate time parameters
            time_range = model_props.get('time', '0..1')
            if isinstance(time_range, str) and '..' in time_range:
                start_str, end_str = time_range.split('..')
                try:
                    time_start = float(start_str)
                    time_end = float(end_str)
                    
                    if time_start < 0:
                        raise ValidationError(f"Negative start time: {time_start}")
                    if time_end <= time_start:
                        raise ValidationError(f"End time ({time_end}) must be greater than start time ({time_start})")
                    if not self._in_range(time_start, 'time') or not self._in_range(time_end, 'time'):
                        raise ValidationError(f"Time range [{time_start}, {time_end}] outside reasonable bounds")
                        
                except ValueError as e:
                    raise ValidationError(f"Invalid time format: {time_range}")
            
            # Validate resolution
            resolution = model_props.get('resolution', 0.01)
            if not isinstance(resolution, (int, float)) or resolution <= 0:
                raise ValidationError(f"Resolution must be positive number, got: {resolution}")
            
            # Validate quantum-specific parameters
            if model_type == 'quantum':
                self._validate_quantum_model(model_props)
                
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Unexpected error validating model: {e}")
    
    def _validate_quantum_model(self, model_props: Dict[str, Any]):
        """Validate quantum model specific parameters"""
        domain_settings = model_props.get('domain_settings')
        if domain_settings:
            dims = domain_settings.get('dimensions')
            points = domain_settings.get('points')
            ranges = domain_settings.get('ranges')
            
            if dims and not isinstance(dims, int):
                raise ValidationError(f"Dimensions must be integer, got: {dims}")
            if dims and not (1 <= dims <= 3):
                raise ValidationError(f"Dimensions must be 1, 2, or 3, got: {dims}")
                
            if points:
                if not isinstance(points, list) or len(points) != dims:
                    raise ValidationError(f"Points must be list of length {dims}, got: {points}")
                for i, p in enumerate(points):
                    if not isinstance(p, int) or p < 10:
                        raise ValidationError(f"points[{i}] must be integer >= 10, got: {p}")
                    if p > 1000:
                        self.warnings.append(f"Large grid size points[{i}]={p} may cause performance issues")
            
            if ranges:
                if not isinstance(ranges, list) or len(ranges) != dims:
                    raise ValidationError(f"Ranges must be list of length {dims}, got: {ranges}")
                for i, r in enumerate(ranges):
                    if not isinstance(r, (list, tuple)) or len(r) != 2:
                        raise ValidationError(f"ranges[{i}] must be [min, max], got: {r}")
                    if r[1] <= r[0]:
                        raise ValidationError(f"ranges[{i}] max ({r[1]}) must be > min ({r[0]})")
                    if not self._in_range(r[0], 'length') or not self._in_range(r[1], 'length'):
                        raise ValidationError(f"ranges[{i}] = {r} outside reasonable length bounds")
    
    def validate_entity(self, entity_type: str, entity_props: Dict[str, Any]) -> bool:
        """Validate entity (object/atom) parameters"""
        try:
            # Validate mass
            mass = entity_props.get('mass')
            if mass is not None:
                if not isinstance(mass, (int, float)) or mass <= 0:
                    raise ValidationError(f"Mass must be positive number, got: {mass}")
                if not self._in_range(mass, 'mass'):
                    if mass < self.RANGES['mass'][0]:
                        raise ValidationError(f"Mass {mass} kg too small (below {self.RANGES['mass'][0]} kg)")
                    else:
                        self.warnings.append(f"Very large mass: {mass} kg")
            
            # Validate position
            position = entity_props.get('position')
            if position is not None:
                if not isinstance(position, (list, tuple)) or len(position) != 3:
                    raise ValidationError(f"Position must be [x, y, z], got: {position}")
                for i, pos in enumerate(position):
                    if not isinstance(pos, (int, float)):
                        raise ValidationError(f"position[{i}] must be number, got: {pos}")
                    if not self._in_range(pos, 'position'):
                        self.warnings.append(f"Large position coordinate position[{i}] = {pos}")
            
            # Validate velocity  
            velocity = entity_props.get('velocity')
            if velocity is not None:
                if not isinstance(velocity, (list, tuple)) or len(velocity) != 3:
                    raise ValidationError(f"Velocity must be [vx, vy, vz], got: {velocity}")
                for i, vel in enumerate(velocity):
                    if not isinstance(vel, (int, float)):
                        raise ValidationError(f"velocity[{i}] must be number, got: {vel}")
                    if abs(vel) > self.RANGES['velocity'][1]:
                        raise ValidationError(f"velocity[{i}] = {vel} m/s exceeds speed of light")
                        
                # Check total velocity
                v_total = np.sqrt(sum(v**2 for v in velocity))
                if v_total > self.RANGES['velocity'][1]:
                    raise ValidationError(f"Total velocity {v_total} m/s exceeds speed of light")
            
            # Validate charge
            charge = entity_props.get('charge')
            if charge is not None:
                if not isinstance(charge, (int, float)):
                    raise ValidationError(f"Charge must be number, got: {charge}")
                if not self._in_range(charge, 'charge'):
                    self.warnings.append(f"Very large charge: {charge} C")
            
            # Validate atom-specific parameters
            if entity_type == 'atom':
                self._validate_atom_properties(entity_props)
                
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Unexpected error validating entity: {e}")
    
    def _validate_atom_properties(self, atom_props: Dict[str, Any]):
        """Validate atom-specific properties"""
        initial_state = atom_props.get('initial_state')
        if initial_state:
            wavefunction = initial_state.get('wavefunction')
            if wavefunction:
                if isinstance(wavefunction, dict):
                    wf_func = wavefunction.get('function')
                    if not wf_func:
                        raise ValidationError("Wavefunction must specify 'function'")
                    
                    # Validate wavefunction parameters
                    if wf_func == 'gaussian':
                        self._validate_gaussian_wavefunction(wavefunction)
    
    def _validate_gaussian_wavefunction(self, wf_dict: Dict[str, Any]):
        """Validate Gaussian wavefunction parameters"""
        kwargs = wf_dict.get('kwargs', {})
        
        center = kwargs.get('center')
        if center and isinstance(center, list):
            for i, c in enumerate(center):
                if not isinstance(c, (int, float)):
                    raise ValidationError(f"Gaussian center[{i}] must be number, got: {c}")
        
        spread = kwargs.get('spread')
        if spread and isinstance(spread, list):
            for i, s in enumerate(spread):
                if not isinstance(s, (int, float)) or s <= 0:
                    raise ValidationError(f"Gaussian spread[{i}] must be positive, got: {s}")
        
        k = kwargs.get('k')
        if k and isinstance(k, list):
            for i, ki in enumerate(k):
                if not isinstance(ki, (int, float)):
                    raise ValidationError(f"Gaussian k[{i}] must be number, got: {ki}")
    
    def validate_interaction(self, interaction: Dict[str, Any]) -> bool:
        """Validate interaction parameters"""
        try:
            source = interaction.get('source')
            target = interaction.get('target')
            
            if not source or not target:
                raise ValidationError("Interaction must specify source and target")
            
            props = interaction.get('properties', {})
            
            # Validate force interactions
            if 'force' in props:
                self._validate_force(props['force'])
            
            # Validate potential interactions  
            if 'potential' in props:
                self._validate_potential(props['potential'])
                
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Unexpected error validating interaction: {e}")
    
    def _validate_force(self, force_spec):
        """Validate force specifications"""
        if isinstance(force_spec, str):
            if force_spec not in ['gravity', 'coulomb', 'spring', 'constant']:
                self.warnings.append(f"Unknown force type: {force_spec}")
        elif isinstance(force_spec, dict):
            func_name = force_spec.get('function')
            if func_name not in ['gravity', 'coulomb', 'spring', 'constant']:
                self.warnings.append(f"Unknown force function: {func_name}")
    
    def _validate_potential(self, potential_spec):
        """Validate potential specifications"""
        if isinstance(potential_spec, str):
            if potential_spec not in ['harmonic', 'coulomb_reg', 'infinite_barrier', 'double_slit_1d', 'uniform_field']:
                self.warnings.append(f"Unknown potential type: {potential_spec}")
        elif isinstance(potential_spec, dict):
            func_name = potential_spec.get('function')
            if func_name not in ['harmonic', 'coulomb_reg', 'infinite_barrier', 'double_slit_1d', 'uniform_field']:
                self.warnings.append(f"Unknown potential function: {func_name}")
    
    def _in_range(self, value: float, param_type: str) -> bool:
        """Check if a value is in reasonable range for its type"""
        if param_type not in self.RANGES:
            return True
        min_val, max_val = self.RANGES[param_type]
        return min_val <= value <= max_val
    
    def validate_program(self, program) -> Tuple[bool, List[str], List[str]]:
        """
        Validate an entire program
        Returns: (is_valid, warnings, errors)
        """
        self.warnings = []
        self.errors = []
        
        try:
            # Validate models
            if hasattr(program, 'models'):
                for model_name, model in program.models.items():
                    try:
                        if hasattr(model, 'properties'):
                            self.validate_model(model.properties)
                        else:
                            self.validate_model(model)
                    except ValidationError as e:
                        self.errors.append(f"Model '{model_name}': {e}")
            
            # Validate objects
            if hasattr(program, 'objects'):
                for obj_name, obj in program.objects.items():
                    try:
                        if hasattr(obj, 'properties'):
                            self.validate_entity('object', obj.properties)
                        else:
                            self.validate_entity('object', obj.get('properties', {}))
                    except ValidationError as e:
                        self.errors.append(f"Object '{obj_name}': {e}")
            
            # Validate atoms
            if hasattr(program, 'atoms'):
                for atom_name, atom in program.atoms.items():
                    try:
                        if hasattr(atom, 'properties'):
                            self.validate_entity('atom', atom.properties)
                        else:
                            self.validate_entity('atom', atom.get('properties', {}))
                    except ValidationError as e:
                        self.errors.append(f"Atom '{atom_name}': {e}")
            
            # Validate interactions
            if hasattr(program, 'interactions'):
                for i, interaction in enumerate(program.interactions):
                    try:
                        if hasattr(interaction, 'source'):
                            # AST object
                            interaction_dict = {
                                'source': interaction.source,
                                'target': interaction.target,
                                'properties': interaction.properties if hasattr(interaction, 'properties') else {}
                            }
                        else:
                            # Dict object
                            interaction_dict = interaction
                        self.validate_interaction(interaction_dict)
                    except ValidationError as e:
                        self.errors.append(f"Interaction {i}: {e}")
            
            return len(self.errors) == 0, self.warnings, self.errors
            
        except Exception as e:
            self.errors.append(f"Validation failed: {e}")
            return False, self.warnings, self.errors


def validate_and_report(program) -> bool:
    """
    Convenience function to validate a program and print results
    Returns True if validation passes
    """
    validator = PhysicsValidator()
    is_valid, warnings, errors = validator.validate_program(program)
    
    if warnings:
        print("Validation warnings:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
    
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  ❌ {error}")
        return False
    
    if not warnings and not errors:
        print("✅ Validation passed - no issues found")
    
    return is_valid