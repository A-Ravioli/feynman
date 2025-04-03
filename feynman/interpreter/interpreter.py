from typing import Dict, List, Any, Optional
import numpy as np

from feynman.parser.parser import PhysicaParser
from feynman.parser.simple_parser import SimpleParser
from feynman.interpreter.ast_builder import ASTBuilder
from feynman.interpreter.ast import Program, Model, Object, Atom, Field, Interaction
from feynman.simulator.classical_simulator import ClassicalSimulator
from feynman.simulator.quantum_simulator import QuantumSimulator
from feynman.visualizer.visualizer import Visualizer
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
        self.visualizer = Visualizer()
        self.visualizer3d = Visualizer3D()
        self.enhanced_visualizer = EnhancedVisualizer()
    
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
        vis3d_results = []
        
        for visualize in self.program.visualizations:
            # Create standard 2D visualizations
            vis_result = self._create_visualization(
                visualize.entity_name, visualize.target_name
            )
            vis_results.append(vis_result)
            
            # Create 3D visualization if applicable
            vis3d_result = self._create_3d_visualization(
                visualize.entity_name, visualize.target_name
            )
            if vis3d_result:
                vis3d_results.append(vis3d_result)
        
        return {
            "simulation_results": self.simulation_results,
            "visualization_results": vis_results,
            "visualization3d_results": vis3d_results
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
    
    def _create_visualization(self, entity_name: str, target_name: Optional[str] = None) -> Dict[str, Any]:
        """Create a visualization for the given entity"""
        # Find which simulation contains this entity
        sim_data = None
        for model_name, sim_result in self.simulation_results.items():
            if entity_name in sim_result["entities"]:
                sim_data = sim_result
                break
        
        if not sim_data:
            # Entity not found in simulation results - return empty result with warning message
            return {entity_name: {"error": f"Entity '{entity_name}' not found in simulation results"}}
        
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
            time_points=sim_data["time_points"],
            target_name=target_name,
            target_data=target_data
        )
        
        return {entity_name: vis_result}
    
    def _create_3d_visualization(self, entity_name: str, target_name: Optional[str] = None) -> Dict[str, Any]:
        """Create a 3D visualization for the given entity"""
        # Find which simulation contains this entity
        sim_data = None
        for model_name, sim_result in self.simulation_results.items():
            if "entities" in sim_result and entity_name in sim_result["entities"]:
                sim_data = sim_result
                break
        
        if not sim_data:
            # Entity not found in simulation results - return empty result with warning message
            return {entity_name: {"error": f"Entity '{entity_name}' not found in 3D visualization"}}
        
        # Get entity data 
        entities = sim_data["entities"]
        time_points = sim_data["time_points"]
        
        # Additional entity check
        if entity_name not in entities:
            return {entity_name: {"error": f"Entity '{entity_name}' not found in entities"}}
        
        entity_data = entities[entity_name]
        
        try:
            # Create static 3D visualization at final time step
            final_frame = self.visualizer3d.visualize_scene(
                entities=entities,
                time_points=time_points,
                time_index=-1  # Final time step
            )
            
            # Create animation if we have trajectory data
            has_trajectory = False
            if entity_data.get("type") == "object" and len(entity_data.get("positions", [])) > 1:
                has_trajectory = True
            elif entity_data.get("type") == "atom" and len(entity_data.get("expected_position", [])) > 1:
                has_trajectory = True
            
            animation_data = None
            if has_trajectory:
                animation_data = self.visualizer3d.create_animation(
                    entities=entities,
                    time_points=time_points,
                    num_frames=30
                )
            
            vis_result = {
                "scene_3d": final_frame["scene_3d"],
                "time": final_frame["time"]
            }
            
            if animation_data:
                vis_result["animation_3d"] = animation_data
                vis_result["animation_type"] = "gif"
            
            return {entity_name: vis_result}
        except Exception as e:
            # Return error message if visualization fails
            import traceback
            error_info = traceback.format_exc()
            print(f"Error in 3D visualization: {error_info}")
            return {entity_name: {"error": f"Error creating 3D visualization: {str(e)}"}}
        
        return None 

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