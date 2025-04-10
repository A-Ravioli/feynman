#!/usr/bin/env python3
import argparse
import sys
import os
import json
import numpy as np
from typing import Dict, Any

from .interpreter.interpreter import Interpreter
# Import the new main visualizer
from .visualizer.main_visualizer import FeynmanVisualizer

# Helper function to convert loaded JSON lists back to numpy arrays if needed
# This might be necessary if the simulator saves arrays directly
# Adjust based on how results are actually saved/loaded
def _load_results_with_numpy(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Recursively convert lists back to numpy arrays where appropriate
    # This is a basic example; might need refinement based on actual data structure
    def convert_to_numpy(item):
        if isinstance(item, dict):
            # Handle complex number dicts saved by old _make_json_serializable
            if 'real' in item and 'imag' in item and len(item) == 2:
                 real_part = np.array(item['real'])
                 imag_part = np.array(item['imag'])
                 return real_part + 1j * imag_part
            # Recursively check other dicts
            return {k: convert_to_numpy(v) for k, v in item.items()}
        elif isinstance(item, list):
            # Attempt conversion if list contains numbers
            try:
                arr = np.array(item)
                # Only convert if it looks like numerical data (e.g., not list of strings)
                if np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.complexfloating):
                    return arr
                else:
                    return [convert_to_numpy(sub_item) for sub_item in item] # Convert list items recursively
            except (ValueError, TypeError):
                 # If conversion fails, return list with items converted recursively
                 return [convert_to_numpy(sub_item) for sub_item in item]
        return item
        
    return convert_to_numpy(data)

class PhysicaLangCLI:
    """Command-line interface for PhysicaLang"""
    
    def __init__(self):
        self.interpreter = Interpreter()
    
    def run(self, args=None):
        """Run the CLI with the given arguments"""
        parser = self.create_argument_parser()
        parsed_args = parser.parse_args(args)
        
        if parsed_args.subcommand == "run":
            self.run_simulation(parsed_args.file[0], parsed_args.output, parsed_args.visualize)
        elif parsed_args.subcommand == "visualize":
            self.visualize_results(parsed_args.results[0])
        else:
            parser.print_help()
    
    def create_argument_parser(self):
        """Create the command-line argument parser"""
        parser = argparse.ArgumentParser(
            description="PhysicaLang - A universal DSL for classical and quantum physics simulations"
        )
        subparsers = parser.add_subparsers(dest="subcommand", help="Subcommands")
        
        # Run simulation command
        run_parser = subparsers.add_parser("run", help="Run a physics simulation from a PhysicaLang script")
        run_parser.add_argument("file", nargs=1, help="PhysicaLang script file")
        run_parser.add_argument(
            "--output", "-o", default="results.json",
            help="Output file for simulation results (default: results.json)"
        )
        run_parser.add_argument(
            "--visualize", "-v", action="store_true",
            help="Launch interactive visualizer after simulation"
        )
        
        # Visualize results command
        vis_parser = subparsers.add_parser("visualize", help="Visualize existing simulation results")
        vis_parser.add_argument("results", nargs=1, help="Simulation results JSON file")
        
        return parser
    
    def run_simulation(self, script_file: str, output_file: str, visualize=False):
        """Run a simulation from a PhysicaLang script file"""
        if not os.path.exists(script_file):
            print(f"Error: File '{script_file}' not found")
            sys.exit(1)
        
        print(f"Running simulation from {script_file}...")
        
        try:
            with open(script_file, "r") as f:
                code = f.read()
            
            # Run the interpreter
            results = self.interpreter.interpret(code)
            
            # --- DEBUG PRINT: Results from Interpreter ---
            # print("\nDEBUG: Results directly from interpreter:") # Remove debug
            # Basic structure print to avoid huge output
            # print(f"  Keys: {list(results.keys())}") # Remove debug
            # if 'time_points' in results: print(f"  Time Points Length: {len(results['time_points'])}") # Remove debug
            # if 'entities' in results: print(f"  Entities: {list(results['entities'].keys())}") # Remove debug
            # if 'interactions' in results: print(f"  Interactions Count: {len(results['interactions'])}") # Remove debug
            # print(results) # Uncomment for full data if needed
            # --- END DEBUG PRINT ---
            
            # Save results
            print(f"Saving simulation results to {output_file}...")
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            with open(output_file, "w") as f:
                # Use the existing serializer for saving
                serializable_results = self._make_json_serializable(results)
                json.dump(serializable_results, f, indent=2)
            
            print(f"Simulation completed. Results saved to {output_file}")
            
            # Launch visualization if requested
            if visualize:
                # Load results back (potentially converting arrays)
                loaded_results = _load_results_with_numpy(output_file)
                
                # --- DEBUG PRINT: Results after loading from JSON ---
                # print("\nDEBUG: Results after loading from JSON:") # Remove debug
                # print(f"  Keys: {list(loaded_results.keys())}") # Remove debug
                # if 'time_points' in loaded_results: print(f"  Time Points Type: {type(loaded_results['time_points'])}") # Remove debug
                # if 'entities' in loaded_results: print(f"  Entities: {list(loaded_results['entities'].keys())}") # Remove debug
                # Check type of a position array to see if numpy conversion worked
                # if 'entities' in loaded_results and loaded_results['entities']: # Remove debug
                #      first_entity_name = list(loaded_results['entities'].keys())[0] # Remove debug
                #      first_entity = loaded_results['entities'][first_entity_name] # Remove debug
                #      if 'time_series' in first_entity and 'positions' in first_entity['time_series']: # Remove debug
                #           pos_type = type(first_entity['time_series']['positions']) # Remove debug
                #           print(f"  Position Array Type (first entity): {pos_type}") # Remove debug
                # print(loaded_results) # Uncomment for full data if needed
                # --- END DEBUG PRINT ---
                
                self.launch_visualizer(loaded_results)
            
        except Exception as e:
            print(f"Error running simulation: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def visualize_results(self, results_file: str):
        """Visualize simulation results from a JSON file"""
        if not os.path.exists(results_file):
            print(f"Error: Results file '{results_file}' not found")
            sys.exit(1)
        
        print(f"Loading results from {results_file} for visualization...")
        
        try:
            # Load results (potentially converting arrays)
            loaded_results = _load_results_with_numpy(results_file)
            self.launch_visualizer(loaded_results)
            
        except Exception as e:
            print(f"Error visualizing results: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def launch_visualizer(self, results: Dict[str, Any]):
        """Instantiates and runs the new main Dash visualizer."""
        print("Launching interactive visualizer...")
        try:
            # Ensure results are properly structured (e.g., with NumPy arrays)
            # _load_results_with_numpy should handle this if called before
            visualizer = FeynmanVisualizer(results) # Instantiate the main visualizer
            # Call the method (which now internally calls app.run)
            visualizer.run_server(debug=False) # Set debug=True for development
            print("Visualizer closed.")
        except Exception as e:
             print(f"Error launching visualizer from CLI: {e}")
             import traceback
             traceback.print_exc()
             # Optionally exit, or let the CLI continue
             # sys.exit(1)
    
    def _make_json_serializable(self, obj):
        """Convert NumPy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            # Handle complex arrays separately
            if np.iscomplexobj(obj):
                # Save complex as dict of real/imag lists
                return {
                    "__complex__": True, # Add marker for easier loading
                    "real": obj.real.tolist(),
                    "imag": obj.imag.tolist()
                }
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex128, np.complex64, np.complex128)):
             # Save complex scalars as dict
             return {"__complex__": True, "real": obj.real, "imag": obj.imag}
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)): # For structured arrays, if any
             return None # Or handle appropriately
        return obj

def main():
    """Main entry point for the CLI"""
    cli = PhysicaLangCLI()
    cli.run()

if __name__ == "__main__":
    main() 