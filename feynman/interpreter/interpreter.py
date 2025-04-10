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
        
        # No longer need to remove this, relying on _collect_initial_properties to clear
        # self._initial_entity_properties = {} 
        self.simulation_results = {} # Still need to clear sim results

        # --- Collect Initial Properties First ---
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
                "simulation_parameters": {},
                "interactions": [] # Add empty interactions list
            }

        # --- Structure the results ---
        if executed_sim_name and executed_sim_name in self.simulation_results:
            sim_output = self.simulation_results[executed_sim_name]
            # Retrieve the interactions associated with this specific simulation run
            # Need to get them from the program based on the model, or store them during _run_simulation
            # Let's retrieve them from the program object now
            relevant_interactions = self._get_interactions_for_model(executed_sim_name)
            
            structured_results = {
                "time_points": sim_output.get("time_points", np.array([])),
                "entities": {},
                "simulation_parameters": sim_output.get("simulation_parameters", {}), # Store params here
                "interactions": relevant_interactions # Include interactions
            }

            # Merge initial properties with time-series data
            # print(f"DEBUG: Merging entities. Simulator output keys: {list(sim_output.get('entities', {}).keys())}") # Remove debug
            for name, initial_props in self._initial_entity_properties.items():
                # print(f"DEBUG: Checking entity '{name}'...") # Remove debug
                # Ensure the entity exists in the simulation output's entities
                sim_output_entities = sim_output.get("entities", {})
                if name in sim_output_entities:
                    # print(f"DEBUG:   Found '{name}' in simulator output. Adding to results.") # Remove debug
                    structured_results["entities"][name] = {
                        "initial_properties": initial_props,
                        "time_series": sim_output_entities[name] # Use the temp variable
                    }
                # else:
                    # print(f"DEBUG:   '{name}' NOT found in simulator output entities.") # Remove debug
                    # Handle entities defined but not part of the simulation output if necessary
                    # For now, only include entities that were actually simulated
            
            return structured_results
        else:
            # Return empty structure if simulation failed or didn't produce results
            return {
                "time_points": np.array([]),
                "entities": {},
                "simulation_parameters": {},
                "interactions": [] # Add empty interactions list
            }
    
    def _collect_initial_properties(self):
        """Helper method to gather initial properties from the parsed program."""
        self._initial_entity_properties = {}
        # print("\nDEBUG (Interpreter): Collecting initial properties...") # Remove debug
        program_objects = getattr(self.program, 'objects', None)
        # print(f"DEBUG: Type of self.program.objects: {type(program_objects)}") # Remove debug
        # print(f"DEBUG: self.program.objects keys: {list(program_objects.keys()) if isinstance(program_objects, dict) else 'N/A'}") # Remove debug
        
        # Try iterating and printing here
        # print("DEBUG: Iterating self.program.objects.items() in _collect_initial_properties:") # Remove debug
        # try: # Remove debug
        #     if hasattr(program_objects, 'items'): # Remove debug
        #          for name, obj in program_objects.items(): # Remove debug
        #               print(f"  Item: '{name}' -> Type: {type(obj)}") # Remove debug
        #     else: # Remove debug
        #          print("  self.program.objects has no 'items' method or is None.") # Remove debug
        # except Exception as e: # Remove debug
        #     print(f"  Error iterating: {e}") # Remove debug
        # print("DEBUG: Finished iterating.") # Remove debug

        # Add objects (classical entities)
        # Iterate through VALUES (Object instances), not keys!
        for obj in self.program.objects.values(): 
            if hasattr(obj, 'name') and hasattr(obj, 'properties'):
                self._initial_entity_properties[obj.name] = obj.properties
            elif isinstance(obj, dict) and "name" in obj: # Keep handling dict format just in case
                self._initial_entity_properties[obj["name"]] = obj.get("properties", {})
                
        # Add atoms (quantum entities) - Add structure if needed
        # Iterate through VALUES here too!
        for atom in self.program.atoms.values(): 
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
        # Iterate through VALUES here too!
        for field in self.program.fields.values(): 
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

    def _get_interactions_for_model(self, model_name: str) -> List[Dict[str, Any]]:
        """Helper method to extract interaction definitions relevant to the model.
           Assumes interactions in self.program apply globally for now.
           A more complex system might associate interactions with models.
        """
        # Currently, the AST/program structure seems to list interactions globally.
        # We return all defined interactions.
        # If interactions were model-specific, logic would be needed here.
        interactions_list = []
        for interaction in self.program.interactions:
            # Handle different interaction representations from parsers
            if isinstance(interaction, dict):
                # Already in the expected dict format
                interactions_list.append(interaction)
            elif hasattr(interaction, 'source') and hasattr(interaction, 'target'):
                # Convert AST node object to dict
                interactions_list.append({
                    "source": interaction.source,
                    "target": interaction.target,
                    "properties": getattr(interaction, 'properties', {})
                })
            # Add handling for other potential formats if necessary
            
        return interactions_list

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
        
        # --- Extract Quantum Simulation Parameters from Model --- 
        simulation_params = {}
        if model_type == "quantum":
             model_props = model.properties if hasattr(model, 'properties') else model.get("properties", {})
             # Domain settings (required)
             # Assume they are directly in model properties or under a 'domain' key
             domain_settings = model_props.get('domain_settings', model_props.get('domain')) 
             if domain_settings:
                  simulation_params['domain_settings'] = domain_settings
             else: # Try to construct from top-level properties if missing
                 dims = model_props.get('dimensions')
                 points = model_props.get('points')
                 ranges = model_props.get('ranges')
                 if dims and points and ranges:
                      simulation_params['domain_settings'] = {'dimensions': dims, 'points': points, 'ranges': ranges}
                 else:
                      # Default if completely missing - might need adjustment
                      print("Warning: Quantum model missing domain settings. Using default 1D grid.")
                      simulation_params['domain_settings'] = {'dimensions': 1, 'points': [100], 'ranges': [(-10, 10)]}
             
             # Solver method (optional)
             solver = model_props.get('solver', model_props.get('solver_method'))
             if solver: simulation_params['solver_method'] = solver
             
             # Eigenstate calculation (optional)
             eigen_calc = model_props.get('calculate_eigenstates')
             if eigen_calc: simulation_params['calculate_eigenstates'] = eigen_calc
             
        # --- Use pre-collected initial properties ---
        # Create the 'entities' dictionary needed by the simulator
        entities_for_simulator = {}
        # print(f"DEBUG (Interpreter): Initial entities_for_simulator (id: {id(entities_for_simulator)}): {list(entities_for_simulator.keys())}") # Remove debug
        for i, (name, props) in enumerate(self._initial_entity_properties.items()): 
            # print(f"DEBUG (Interpreter): Loop {i}, adding '{name}'") # Remove debug
            # Determine entity type (object, atom, field) based on stored props or defaults
            entity_type = "object" # Default assumption, adjust if props contain type info
            if 'type' in props:
                entity_type = props['type']
            # Or infer based on which list it came from (objects, atoms, fields) if needed
            # For now, assume props might contain a 'type' hint or default to 'object'

            value_to_assign = {
                "type": entity_type, # Pass type to simulator
                "properties": props # Pass all initial properties
            }
            entities_for_simulator[name] = value_to_assign
            # print(f"DEBUG (Interpreter):   After adding '{name}', keys: {list(entities_for_simulator.keys())}") # Remove debug

        # print(f"DEBUG (Interpreter): Final entities_for_simulator (id: {id(entities_for_simulator)}): {list(entities_for_simulator.keys())}") # Remove debug
        
        # --- Prepare interactions for the simulator --- 
        interactions_for_simulator = self._get_interactions_for_model(model_name)

        # --- DEBUG PRINT: Entities being passed to simulator (Detailed) ---
        print(f"\nDEBUG (Interpreter): Passing entities to simulator:") 
        for name, data in entities_for_simulator.items():
             print(f"  '{name}': type='{data.get('type')}', properties={data.get('properties')}") # Print full properties
        # --- END DEBUG PRINT ---

        # Choose simulator based on model type
        if model_type == "classical":
            # --- Pass a copy to isolate from potential side effects ---
            entities_copy = entities_for_simulator.copy()
            # print(f"DEBUG (Interpreter): Passing COPY of entities (orig id: {id(entities_for_simulator)}, copy id: {id(entities_copy)}) with keys: {list(entities_copy.keys())}") # Remove debug
            result = self.classical_simulator.simulate(
                entities=entities_copy, # Pass the copy
                interactions=interactions_for_simulator,
                time_start=time_start,
                time_end=time_end,
                time_step=time_step
            )
        elif model_type == "quantum":
            # Now pass the gathered simulation_params
            result = self.quantum_simulator.simulate(
                entities=entities_for_simulator,
                interactions=interactions_for_simulator, # Pass the prepared list
                time_start=time_start,
                time_end=time_end,
                time_step=time_step,
                simulation_params=simulation_params # Pass the new dict
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