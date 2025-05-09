from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable
import numpy as np

# Base node class
class Node:
    pass

@dataclass
class Model(Node):
    name: str
    properties: Dict[str, Any]
    
    def __init__(self, name=None, properties=None):
        self.name = name
        self.properties = properties or {}
    
    def get_type(self) -> str:
        return self.properties.get("type", "classical")
    
    def get_time_range(self):
        """Get the time range as (start, end)"""
        time_str = self.properties.get("time", "0..1")
        
        # Handle if time is already a tuple/list
        if isinstance(time_str, (list, tuple)) and len(time_str) == 2:
            return float(time_str[0]), float(time_str[1])
        
        # Handle if time is already a float (single value)
        if isinstance(time_str, (int, float)):
            return 0.0, float(time_str)
        
        # Handle string formats like "0..1" or "0 to 1"
        if isinstance(time_str, str):
            if ".." in time_str:
                start, end = time_str.split("..")
                return float(start.strip()), float(end.strip())
            elif " to " in time_str:
                start, end = time_str.split(" to ")
                return float(start.strip()), float(end.strip())
            else:
                # If it's just a single number like "5s", treat it as end time
                # Remove any units (like 's' for seconds)
                time_val = time_str.strip()
                for unit in ["s", "ms", "us", "ns"]:
                    if time_val.endswith(unit):
                        time_val = time_val[:-len(unit)]
                        break
                try:
                    end = float(time_val)
                    return 0.0, end
                except ValueError:
                    # Default if parsing fails
                    return 0.0, 1.0
        
        # Default
        return 0.0, 1.0
    
    def get_resolution(self):
        """Get the time step resolution"""
        res = self.properties.get("resolution", 0.01)
        
        # Handle if resolution is already a float
        if isinstance(res, (int, float)):
            return float(res)
        
        # Handle string format
        if isinstance(res, str):
            # Remove any units (like 's' for seconds)
            res_val = res.strip()
            for unit in ["s", "ms", "us", "ns"]:
                if res_val.endswith(unit):
                    res_val = res_val[:-len(unit)]
                    break
            try:
                return float(res_val)
            except ValueError:
                # Default if parsing fails
                return 0.01
        
        # Default
        return 0.01

@dataclass
class Object(Node):
    name: str
    properties: Dict[str, Any]
    
    def get_mass(self) -> float:
        return float(self.properties.get("mass", 1.0))
    
    def get_position(self) -> np.ndarray:
        pos = self.properties.get("position", [0, 0, 0])
        return np.array(pos, dtype=float)
    
    def get_velocity(self) -> np.ndarray:
        vel = self.properties.get("velocity", [0, 0, 0])
        return np.array(vel, dtype=float)

@dataclass
class Atom(Node):
    name: str
    properties: Dict[str, Any]
    
    def get_mass(self) -> float:
        return float(self.properties.get("mass", 0.0))
    
    def get_initial_state(self) -> Dict[str, Any]:
        return self.properties.get("initial_state", {})

@dataclass
class Field(Node):
    name: str
    properties: Dict[str, Any]
    
    def get_type(self) -> str:
        return self.properties.get("type", "classical")
    
    def get_formula(self) -> str:
        return self.properties.get("formula", "")

@dataclass
class Interaction(Node):
    source: str
    target: str
    properties: Dict[str, Any]
    
    def get_type(self) -> str:
        if "force" in self.properties:
            return "force"
        elif "potential" in self.properties:
            return "potential"
        elif "action" in self.properties:
            return "action"
        return "unknown"
    
    def get_value(self) -> Any:
        if "force" in self.properties:
            return self.properties["force"]
        elif "potential" in self.properties:
            return self.properties["potential"]
        elif "action" in self.properties:
            return self.properties["action"]
        return None

@dataclass
class Simulate(Node):
    model_name: str

@dataclass
class Visualize(Node):
    entity_name: str
    target_name: Optional[str] = None
    visualization_type: Optional[str] = None

@dataclass
class Program(Node):
    models: Dict[str, Model] = field(default_factory=dict)
    objects: Dict[str, Object] = field(default_factory=dict)
    atoms: Dict[str, Atom] = field(default_factory=dict)
    fields: Dict[str, Field] = field(default_factory=dict)
    interactions: List[Interaction] = field(default_factory=list)
    simulations: List[Simulate] = field(default_factory=list)
    visualizations: List[Visualize] = field(default_factory=list)
    
    def add_model(self, model: Model):
        self.models[model.name] = model
    
    def add_object(self, obj: Object):
        self.objects[obj.name] = obj
    
    def add_atom(self, atom: Atom):
        self.atoms[atom.name] = atom
    
    def add_field(self, field: Field):
        self.fields[field.name] = field
    
    def add_interaction(self, interaction: Interaction):
        self.interactions.append(interaction)
    
    def add_simulate(self, simulate: Simulate):
        self.simulations.append(simulate)
    
    def add_visualize(self, visualize: Visualize):
        self.visualizations.append(visualize) 