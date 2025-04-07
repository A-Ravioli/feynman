from typing import Dict, List, Any, Optional
import numpy as np

from feynman.parser.parser import PhysicaParser
from feynman.parser.simple_parser import SimpleParser
from feynman.interpreter.ast_builder import ASTBuilder
from feynman.interpreter.ast import Program, Model, Object, Atom, Field, Interaction
from feynman.simulator.classical_simulator import ClassicalSimulator
from feynman.simulator.quantum_simulator import QuantumSimulator
from feynman.visualizer.visualizer3d import Visualizer3D
from feynman.visualizer.enhanced_visualizer import EnhancedVisualizer

class Interpreter:
    def __init__(self):
        self.lark_parser = PhysicaParser()
        self.simple_parser = SimpleParser()
        self.ast_builder = ASTBuilder()
        self.program = None
        self.simulation_results = {}
        self.classical_simulator = ClassicalSimulator()
        self.quantum_simulator = QuantumSimulator()
        self.visualizer3d = Visualizer3D()
        self.enhanced_visualizer = EnhancedVisualizer()
    
    def interpret(self, code: str) -> Dict[str, Any]:
        """Interpret PhysicaLang code and return simulation results only.
           Visualization is now handled separately after interpretation.
        """
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
        
        # Return only simulation results
        # The structure might need adjusting depending on how results are used
        # For now, assume the calling code expects this structure
        # If the structure included visualization results previously, 
        # that needs to be handled by the caller now. 
        return {
            "simulation_results": self.simulation_results,
            # Add other necessary keys if the structure was different, e.g.:
            "time_points": self.simulation_results.get(list(self.simulation_results.keys())[0], {}).get('time_points', []) if self.simulation_results else [],
            "entities": self.simulation_results.get(list(self.simulation_results.keys())[0], {}).get('entities', {}) if self.simulation_results else {},
            "simulation_params": self.simulation_results.get(list(self.simulation_results.keys())[0], {}).get('simulation_params', {}) if self.simulation_results else {}
        }
    
    def _run_simulation(self, model_name: str) -> Dict[str, Any]:
        """Run a simulation for the given model"""
        # Find the model
        model = None
        if isinstance(self.program.models, dict):
            # Handle case where models are stored in a dictionary
            if model_name in self.program.models:
                model = self.program.models[model_name]
        else:
            # Handle case where models are stored in a list
            for m in self.program.models:
                if isinstance(m, str):
                    # If the model is a string, use it directly
                    if m == model_name:
                        model = {"name": model_name, "properties": {}}
                        break
                elif hasattr(m, 'name') and m.name == model_name:
                    model = m
                    break
        
        if model is None:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Get model properties
        if isinstance(model, dict):
            model_type = model.get("properties", {}).get("type", "classical")
        else:
            model_type = model.properties.get("type", "classical")
        
        # Get time range and resolution
        time_start, time_end = model.get_time_range() if hasattr(model, 'get_time_range') else (0.0, 1.0)
        time_step = model.get_resolution() if hasattr(model, 'get_resolution') else 0.01
        
        # Collect entities for this model
        entities = {}
        
        # Add objects (classical entities)
        for obj in self.program.objects:
            # Handle different object representations
            if isinstance(obj, str):
                # If the object is a string, use it directly with default properties
                entities[obj] = {
                    "type": "object",
                    "properties": {}
                }
            elif hasattr(obj, 'name') and hasattr(obj, 'properties'):
                entities[obj.name] = {
                    "type": "object",
                    "properties": obj.properties
                }
            elif isinstance(obj, dict) and "name" in obj:
                entities[obj["name"]] = {
                    "type": "object",
                    "properties": obj.get("properties", {})
                }
        
        # Add atoms (quantum entities)
        for atom in self.program.atoms:
            # Handle different atom representations
            if isinstance(atom, str):
                # If the atom is a string, use it directly with default properties
                entities[atom] = {
                    "type": "atom",
                    "properties": {}
                }
            elif hasattr(atom, 'name') and hasattr(atom, 'properties'):
                entities[atom.name] = {
                    "type": "atom",
                    "properties": atom.properties
                }
            elif isinstance(atom, dict) and "name" in atom:
                entities[atom["name"]] = {
                    "type": "atom",
                    "properties": atom.get("properties", {})
                }
        
        # Add fields
        for field in self.program.fields:
            # Handle different field representations
            if isinstance(field, str):
                # If the field is a string, use it directly with default properties
                entities[field] = {
                    "type": "field",
                    "properties": {}
                }
            elif hasattr(field, 'name') and hasattr(field, 'properties'):
                entities[field.name] = {
                    "type": "field",
                    "properties": field.properties
                }
            elif isinstance(field, dict) and "name" in field:
                entities[field["name"]] = {
                    "type": "field",
                    "properties": field.get("properties", {})
                }
        
        interactions = []
        for interaction in self.program.interactions:
            # Handle different interaction representations
            if isinstance(interaction, dict):
                interactions.append(interaction)
            elif hasattr(interaction, 'source') and hasattr(interaction, 'target'):
                interactions.append({
                    "source": interaction.source,
                    "target": interaction.target,
                    "properties": getattr(interaction, 'properties', {})
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

    def _create_3d_visualization(self, ast, sim_result, entity_name):
        """Create a 3D visualization from simulation results."""
        if sim_result is None:
            return {"error": f"No simulation results available for visualization"}

        # Get the entity from the simulation result
        entities = sim_result.get("entities", {})
        if entity_name and entity_name not in entities:
            return {"error": f"Entity '{entity_name}' not found in simulation results"}

        # Get interactions
        interactions = []
        for interaction in ast.interactions:
            interactions.append({
                "source": interaction.source,
                "target": interaction.target,
                "properties": interaction.get_properties()
            })

        # Create visualization using the advanced 3D visualizer
        all_entities = {}
        time_points = sim_result.get("time_points", [])
        
        for name, entity_data in entities.items():
            all_entities[name] = entity_data
        
        use_enhanced = ast.get_property("enhanced_visuals", "false").lower() == "true"
        
        if use_enhanced:
            # Use the enhanced visualizer with interactive features
            html = self.enhanced_visualizer.create_visualization(all_entities, time_points, interactions, entity_name)
        else:
            # Use the standard 3D visualizer
            html = self.visualizer3d.visualize_scene(all_entities, entity_name, interactions=interactions)
            
            # If animation is desired, create it
            if ast.get_property("animate", "false").lower() == "true":
                animation_html = self.visualizer3d.create_animation(all_entities, entity_name, interactions=interactions)
                # Combine the static and animated visualizations
                html = f"""
                <h2>3D Visualization</h2>
                <div style="display: flex; justify-content: center;">
                    <div style="margin-right: 20px;">
                        <h3>Static View</h3>
                        {html}
                    </div>
                    <div>
                        <h3>Animation</h3>
                        {animation_html}
                    </div>
                </div>
                """

        return {"html": html} 