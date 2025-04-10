from typing import Dict, List, Any, Optional
import numpy as np

from feynman.parser.parser import PhysicaParser
from feynman.parser.simple_parser import SimpleParser
from feynman.interpreter.ast_builder import ASTBuilder
from feynman.interpreter.ast import Program, Model, Object, Atom, Field, Interaction
from feynman.simulator.classical_simulator import ClassicalSimulator
from feynman.simulator.quantum_simulator import QuantumSimulator

class Interpreter:
    def __init__(self):
        self.lark_parser = PhysicaParser()
        self.simple_parser = SimpleParser()
        self.ast_builder = ASTBuilder()
        self.program = None
        self._initial_entity_properties = {}
        self.simulation_results = {}
        self.classical_simulator = ClassicalSimulator()
        self.quantum_simulator = QuantumSimulator()
    
    def interpret(self, code: str) -> Dict[str, Any]:
        """Interpret PhysicaLang code, run simulations, and return structured results.

        Returns:
            Dict[str, Any]: A dictionary containing simulation results structured
                           for visualization (time_points, entities with initial_properties
                           and time_series data, simulation_parameters). Returns data
                           for the first simulation run specified in the code.
        """
        try:
            # First try our simple parser which is more robust
            self.program = self.simple_parser.parse(code)
        except Exception as e:
            # Fall back to Lark parser if simple parser fails
            print(f"Simple parser failed: {e}. Falling back to Lark parser.")
            parse_tree = self.lark_parser.parse(code)
            self.program = self.ast_builder.transform(parse_tree)
        
        # Clear previous state
        self._initial_entity_properties = {}
        self.simulation_results = {}

        # --- Collect Initial Properties First ---
        # We need the initial properties *before* running the simulation
        # The _run_simulation method will be adapted to use this stored data
        self._collect_initial_properties()

        # --- Process and execute simulations ---
        # Run only the first simulation defined in the .phys file
        executed_sim_name = None
        if self.program.simulations:
            first_simulation = self.program.simulations[0]
            model_name_to_run = first_simulation.model_name
            self._run_simulation(model_name_to_run)
            executed_sim_name = model_name_to_run
        else:
            # Return an empty structure if no simulation is run
            return {
                "time_points": np.array([]),
                "entities": {},
                "simulation_parameters": {}
            }

        # --- Structure the results ---
        if executed_sim_name and executed_sim_name in self.simulation_results:
            sim_output = self.simulation_results[executed_sim_name]
            structured_results = {
                "time_points": sim_output.get("time_points", np.array([])),
                "entities": {},
                "simulation_parameters": sim_output.get("simulation_parameters", {}) # Store params here
            }

            # Merge initial properties with time-series data
            for name, initial_props in self._initial_entity_properties.items():
                # Ensure the entity exists in the simulation output's entities
                if name in sim_output.get("entities", {}):
                    structured_results["entities"][name] = {
                        "initial_properties": initial_props,
                        "time_series": sim_output["entities"][name] # This now contains pos, vel, energy etc.
                    }
                else:
                    # Handle entities defined but not part of the simulation output if necessary
                    # For now, only include entities that were actually simulated
                    pass
            
            return structured_results
        else:
            # Return empty structure if simulation failed or didn't produce results
            return {
                "time_points": np.array([]),
                "entities": {},
                "simulation_parameters": {}
            }
    
    def _collect_initial_properties(self):
        """Helper method to gather initial properties from the parsed program."""
        self._initial_entity_properties = {}
        # Add objects (classical entities)
        for obj in self.program.objects:
            if hasattr(obj, 'name') and hasattr(obj, 'properties'):
                self._initial_entity_properties[obj.name] = obj.properties
            elif isinstance(obj, dict) and "name" in obj: # Handle simple parser dict format
                self._initial_entity_properties[obj["name"]] = obj.get("properties", {})
                
        # Add atoms (quantum entities) - Add structure if needed
        for atom in self.program.atoms:
            if hasattr(atom, 'name') and hasattr(atom, 'properties'):
                self._initial_entity_properties[atom.name] = {
                    "type": "atom", # Explicitly add type if not in properties
                    **atom.properties
                }
            elif isinstance(atom, dict) and "name" in atom:
                self._initial_entity_properties[atom["name"]] = {
                    "type": "atom",
                    **atom.get("properties", {})
                }

        # Add fields - Add structure if needed
        for field in self.program.fields:
            if hasattr(field, 'name') and hasattr(field, 'properties'):
                self._initial_entity_properties[field.name] = {
                    "type": "field",
                    **field.properties
                }
            elif isinstance(field, dict) and "name" in field:
                self._initial_entity_properties[field["name"]] = {
                    "type": "field",
                    **field.get("properties", {})
                }

    def _run_simulation(self, model_name: str):
        """Run a simulation for the given model and store results in self.simulation_results"""
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
        
        # --- Use pre-collected initial properties ---
        # Create the 'entities' dictionary needed by the simulator
        entities_for_simulator = {}
        for name, props in self._initial_entity_properties.items():
            # Determine entity type (object, atom, field) based on stored props or defaults
            entity_type = "object" # Default assumption, adjust if props contain type info
            if 'type' in props:
                entity_type = props['type']
            # Or infer based on which list it came from (objects, atoms, fields) if needed
            # For now, assume props might contain a 'type' hint or default to 'object'

            entities_for_simulator[name] = {
                "type": entity_type, # Pass type to simulator
                "properties": props # Pass all initial properties
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
                entities=entities_for_simulator,
                interactions=interactions,
                time_start=time_start,
                time_end=time_end,
                time_step=time_step
            )
        elif model_type == "quantum":
            result = self.quantum_simulator.simulate(
                entities=entities_for_simulator,
                interactions=interactions,
                time_start=time_start,
                time_end=time_end,
                time_step=time_step
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Store results along with simulation parameters
        result["simulation_parameters"] = {
            "time_start": time_start,
            "time_end": time_end,
            "time_step": time_step,
            "model_type": model_type
        }
        self.simulation_results[model_name] = result
        # No return needed, results stored in self.simulation_results 