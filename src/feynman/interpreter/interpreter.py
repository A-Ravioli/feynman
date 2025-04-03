from typing import Dict, List, Any, Optional
import numpy as np

from ..parser.parser import PhysicaParser
from ..parser.simple_parser import SimpleParser
from .ast_builder import ASTBuilder
from .ast import Program, Model, Object, Atom, Field, Interaction
from ..simulator.classical_simulator import ClassicalSimulator
from ..simulator.quantum_simulator import QuantumSimulator
from ..visualizer.visualizer import Visualizer

class Interpreter:
    def __init__(self):
        self.lark_parser = PhysicaParser()
        self.simple_parser = SimpleParser()
        self.ast_builder = ASTBuilder()
        self.program = None
        self.simulation_results = {}
        self.classical_simulator = ClassicalSimulator()
        self.quantum_simulator = QuantumSimulator()
        self.visualizer = Visualizer()
    
    def interpret(self, code: str) -> Dict[str, Any]:
        """Interpret PhysicaLang code and return results"""
        try:
            # First try our simple parser which is more robust
            self.program = self.simple_parser.parse(code)
        except Exception as e:
            # Fall back to Lark parser if simple parser fails
            print(f"Simple parser failed: {e}. Falling back to Lark parser.")
            parse_tree = self.lark_parser.parse(code)
            self.program = self.ast_builder.transform(parse_tree)
        
        # Process and execute simulations
        for simulate in self.program.simulations:
            self._run_simulation(simulate.model_name)
        
        # Process visualizations
        vis_results = []
        for visualize in self.program.visualizations:
            vis_result = self._create_visualization(
                visualize.entity_name, 
                visualize.target_name
            )
            vis_results.append(vis_result)
        
        return {
            "simulation_results": self.simulation_results,
            "visualization_results": vis_results
        }
    
    def _run_simulation(self, model_name: str) -> Dict[str, Any]:
        """Run a simulation for the given model"""
        if model_name not in self.program.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.program.models[model_name]
        model_type = model.get_type()
        time_start, time_end = model.get_time_range()
        time_step = model.get_resolution()
        
        # Collect entities and interactions for the simulation
        entities = {}
        for obj_name, obj in self.program.objects.items():
            entities[obj_name] = {
                "type": "object",
                "properties": obj.properties
            }
        
        for atom_name, atom in self.program.atoms.items():
            entities[atom_name] = {
                "type": "atom", 
                "properties": atom.properties
            }
        
        for field_name, field in self.program.fields.items():
            entities[field_name] = {
                "type": "field",
                "properties": field.properties
            }
        
        interactions = []
        for interaction in self.program.interactions:
            interactions.append({
                "source": interaction.source,
                "target": interaction.target,
                "properties": interaction.properties
            })
        
        # Choose simulator based on model type
        if model_type == "classical":
            result = self.classical_simulator.simulate(
                entities=entities,
                interactions=interactions,
                time_start=time_start,
                time_end=time_end,
                time_step=time_step
            )
        elif model_type == "quantum":
            result = self.quantum_simulator.simulate(
                entities=entities,
                interactions=interactions,
                time_start=time_start,
                time_end=time_end,
                time_step=time_step
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.simulation_results[model_name] = result
        return result
    
    def _create_visualization(self, entity_name: str, target_name: Optional[str] = None) -> Dict[str, Any]:
        """Create a visualization for the given entity"""
        # Find which simulation contains this entity
        sim_data = None
        for model_name, sim_result in self.simulation_results.items():
            if entity_name in sim_result["entities"]:
                sim_data = sim_result
                break
        
        if not sim_data:
            raise ValueError(f"Entity '{entity_name}' not found in any simulation results")
        
        # Get entity data
        entity_data = sim_data["entities"][entity_name]
        
        # Get target data if specified
        target_data = None
        if target_name and target_name in sim_data["entities"]:
            target_data = sim_data["entities"][target_name]
        
        # Create visualization
        vis_result = self.visualizer.visualize(
            entity_name=entity_name,
            entity_data=entity_data,
            target_name=target_name,
            target_data=target_data,
            time_points=sim_data["time_points"]
        )
        
        return vis_result 