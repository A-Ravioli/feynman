"""
Utility functions for Feynman physics DSL
"""

import numpy as np
import warnings
from typing import Dict, List, Any, Union, Tuple, Optional

# Physical constants
HBAR = 1.0545718e-34  # Planck's constant / 2π
C = 299792458  # Speed of light
K_E = 8.9875517923e9  # Coulomb's constant
G = 6.67430e-11  # Gravitational constant

def validate_array_input(arr: Any, name: str, expected_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """Validate and convert input to numpy array with optional shape checking."""
    if not isinstance(arr, (list, tuple, np.ndarray)):
        raise TypeError(f"{name} must be array-like (list, tuple, or numpy array)")
    
    arr = np.asarray(arr, dtype=float)
    
    if expected_shape is not None and arr.shape != expected_shape:
        raise ValueError(f"{name} has shape {arr.shape}, expected {expected_shape}")
    
    return arr

def validate_time_parameters(time_start: float, time_end: float, time_step: float) -> None:
    """Validate time parameters for simulations."""
    if time_end <= time_start:
        raise ValueError("time_end must be greater than time_start")
    if time_step <= 0:
        raise ValueError("time_step must be positive")
    if time_step > (time_end - time_start):
        raise ValueError("time_step too large for simulation time range")

def safe_normalize(r_vec: np.ndarray, threshold: float = 1e-10) -> Tuple[float, np.ndarray]:
    """Safely normalize a vector, returning magnitude and unit vector."""
    r = np.linalg.norm(r_vec)
    if r < threshold:
        return 0.0, np.zeros_like(r_vec)
    return r, r_vec / r

def parse_function_call(func_info: Any) -> Tuple[str, Dict[str, Any]]:
    """Parse function call information into name and parameters."""
    if isinstance(func_info, str):
        return func_info, {}
    elif isinstance(func_info, dict) and "function" in func_info:
        func_name = func_info["function"]
        params = {}
        for arg in func_info.get("args", []):
            if isinstance(arg, dict) and "param" in arg and "value" in arg:
                params[arg["param"]] = arg["value"]
        return func_name, params
    else:
        raise ValueError(f"Invalid function format: {func_info}")

def create_grid_1d_to_3d(dims: int, points: List[int], ranges: List[Tuple[float, float]]) -> Tuple[List[np.ndarray], List[float], List[int]]:
    """Create coordinate grids for 1D, 2D, or 3D simulations."""
    if not 1 <= dims <= 3:
        raise ValueError("dims must be 1, 2, or 3")
    if len(points) != dims or len(ranges) != dims:
        raise ValueError("points and ranges must have length equal to dims")
    
    grid_coords = []
    deltas = []
    n_points = []
    
    for i in range(dims):
        if points[i] < 10:
            raise ValueError(f"points[{i}] must be >= 10")
        if ranges[i][1] <= ranges[i][0]:
            raise ValueError(f"ranges[{i}] must have max > min")
            
        coords, delta = np.linspace(ranges[i][0], ranges[i][1], points[i], retstep=True)
        grid_coords.append(coords)
        deltas.append(delta)
        n_points.append(points[i])
    
    return grid_coords, deltas, n_points

def compute_kinetic_energy(mass: float, velocity: np.ndarray) -> float:
    """Compute kinetic energy for given mass and velocity."""
    return 0.5 * mass * np.sum(velocity**2)

def format_complex_for_json(obj: Any) -> Any:
    """Convert complex numbers and numpy arrays to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {
                "__complex_array__": True,
                "real": obj.real.tolist(),
                "imag": obj.imag.tolist()
            }
        return obj.tolist()
    elif isinstance(obj, (np.complex128, np.complex64)):
        return {
            "__complex__": True,
            "real": float(obj.real),
            "imag": float(obj.imag)
        }
    elif isinstance(obj, dict):
        return {k: format_complex_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [format_complex_for_json(item) for item in obj]
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def restore_complex_from_json(obj: Any) -> Any:
    """Restore complex numbers and numpy arrays from JSON format."""
    if isinstance(obj, dict):
        if obj.get("__complex_array__"):
            real = np.array(obj["real"])
            imag = np.array(obj["imag"])
            return real + 1j * imag
        elif obj.get("__complex__"):
            return complex(obj["real"], obj["imag"])
        else:
            return {k: restore_complex_from_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [restore_complex_from_json(item) for item in obj]
    return obj 